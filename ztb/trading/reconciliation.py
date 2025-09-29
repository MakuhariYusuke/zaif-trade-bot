"""
Reconciliation Framework Skeleton

Provides framework for reconciling trading operations between
internal state and external systems (exchanges, brokers).
Ensures consistency and detects discrepancies.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ztb.utils.observability import get_logger

logger = get_logger(__name__)


@dataclass
class ReconciliationItem:
    """Item to be reconciled."""
    item_id: str
    internal_state: Dict[str, Any]
    external_state: Dict[str, Any]
    timestamp: float


@dataclass
class ReconciliationResult:
    """Result of reconciliation check."""
    item_id: str
    is_consistent: bool
    discrepancies: List[str]
    actions_taken: List[str]
    timestamp: float


class ReconciliationStrategy(ABC):
    """Abstract base class for reconciliation strategies."""

    @abstractmethod
    def reconcile_item(self, item: ReconciliationItem) -> ReconciliationResult:
        """
        Reconcile a single item.

        Args:
            item: Item to reconcile

        Returns:
            Reconciliation result
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class OrderReconciliationStrategy(ReconciliationStrategy):
    """Reconciliation strategy for orders."""

    def get_strategy_name(self) -> str:
        return "order_reconciliation"

    def reconcile_item(self, item: ReconciliationItem) -> ReconciliationResult:
        """Reconcile order states."""
        discrepancies = []
        actions = []

        internal = item.internal_state
        external = item.external_state

        # Check order status
        if internal.get('status') != external.get('status'):
            discrepancies.append(
                f"Status mismatch: internal={internal.get('status')}, external={external.get('status')}"
            )

        # Check filled quantity
        internal_filled = internal.get('filled_quantity', 0)
        external_filled = external.get('filled_quantity', 0)
        if abs(internal_filled - external_filled) > 0.0001:  # Allow small floating point differences
            discrepancies.append(
                f"Filled quantity mismatch: internal={internal_filled}, external={external_filled}"
            )

        # Check remaining quantity
        internal_remaining = internal.get('remaining_quantity', 0)
        external_remaining = external.get('remaining_quantity', 0)
        if abs(internal_remaining - external_remaining) > 0.0001:
            discrepancies.append(
                f"Remaining quantity mismatch: internal={internal_remaining}, external={external_remaining}"
            )

        # If discrepancies found, log and potentially trigger corrective actions
        if discrepancies:
            logger.warning(f"Reconciliation discrepancies for order {item.item_id}: {discrepancies}")
            # TODO: Implement corrective actions (e.g., update internal state, alert, etc.)
            actions.append("logged_discrepancies")
        else:
            actions.append("no_action_needed")

        return ReconciliationResult(
            item_id=item.item_id,
            is_consistent=len(discrepancies) == 0,
            discrepancies=discrepancies,
            actions_taken=actions,
            timestamp=item.timestamp
        )


class PositionReconciliationStrategy(ReconciliationStrategy):
    """Reconciliation strategy for positions."""

    def get_strategy_name(self) -> str:
        return "position_reconciliation"

    def reconcile_item(self, item: ReconciliationItem) -> ReconciliationResult:
        """Reconcile position states."""
        discrepancies = []
        actions = []

        internal = item.internal_state
        external = item.external_state

        # Check position size
        internal_size = internal.get('size', 0)
        external_size = external.get('size', 0)
        if abs(internal_size - external_size) > 0.0001:
            discrepancies.append(
                f"Position size mismatch: internal={internal_size}, external={external_size}"
            )

        # Check average price
        internal_avg_price = internal.get('average_price', 0)
        external_avg_price = external.get('average_price', 0)
        if abs(internal_avg_price - external_avg_price) > 0.01:  # Allow 1% difference
            discrepancies.append(
                f"Average price mismatch: internal={internal_avg_price}, external={external_avg_price}"
            )

        if discrepancies:
            logger.warning(f"Reconciliation discrepancies for position {item.item_id}: {discrepancies}")
            actions.append("logged_discrepancies")
        else:
            actions.append("no_action_needed")

        return ReconciliationResult(
            item_id=item.item_id,
            is_consistent=len(discrepancies) == 0,
            discrepancies=discrepancies,
            actions_taken=actions,
            timestamp=item.timestamp
        )


class ReconciliationEngine:
    """Engine for running reconciliation processes."""

    def __init__(self):
        """Initialize reconciliation engine."""
        self.strategies: Dict[str, ReconciliationStrategy] = {}
        self.register_default_strategies()

    def register_default_strategies(self):
        """Register default reconciliation strategies."""
        self.register_strategy(OrderReconciliationStrategy())
        self.register_strategy(PositionReconciliationStrategy())

    def register_strategy(self, strategy: ReconciliationStrategy):
        """Register a reconciliation strategy."""
        self.strategies[strategy.get_strategy_name()] = strategy
        logger.info(f"Registered reconciliation strategy: {strategy.get_strategy_name()}")

    def reconcile_items(self, items: List[ReconciliationItem], strategy_name: str) -> List[ReconciliationResult]:
        """
        Reconcile multiple items using specified strategy.

        Args:
            items: Items to reconcile
            strategy_name: Name of strategy to use

        Returns:
            List of reconciliation results
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown reconciliation strategy: {strategy_name}")

        strategy = self.strategies[strategy_name]
        results = []

        for item in items:
            try:
                result = strategy.reconcile_item(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error reconciling item {item.item_id}: {e}")
                # Create error result
                results.append(ReconciliationResult(
                    item_id=item.item_id,
                    is_consistent=False,
                    discrepancies=[f"Reconciliation error: {str(e)}"],
                    actions_taken=["error_logged"],
                    timestamp=item.timestamp
                ))

        return results

    def get_reconciliation_summary(self, results: List[ReconciliationResult]) -> Dict[str, Any]:
        """
        Generate summary of reconciliation results.

        Args:
            results: Reconciliation results

        Returns:
            Summary statistics
        """
        total_items = len(results)
        consistent_items = sum(1 for r in results if r.is_consistent)
        inconsistent_items = total_items - consistent_items

        total_discrepancies = sum(len(r.discrepancies) for r in results)

        return {
            "total_items": total_items,
            "consistent_items": consistent_items,
            "inconsistent_items": inconsistent_items,
            "consistency_rate": consistent_items / total_items if total_items > 0 else 0,
            "total_discrepancies": total_discrepancies,
            "results": results
        }


# Global instance
_reconciliation_engine: Optional[ReconciliationEngine] = None


def get_reconciliation_engine() -> ReconciliationEngine:
    """Get global reconciliation engine instance."""
    global _reconciliation_engine
    if _reconciliation_engine is None:
        _reconciliation_engine = ReconciliationEngine()
    return _reconciliation_engine