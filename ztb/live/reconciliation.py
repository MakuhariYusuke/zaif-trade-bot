"""
Reconciliation utilities for position and balance verification.

This module provides reconciliation functionality to verify trading positions,
balances, and order states against external sources for consistency and accuracy.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ztb.utils.observability import get_logger

logger = get_logger(__name__)


@dataclass
class PositionRecord:
    """Position record for reconciliation."""

    symbol: str
    quantity: float
    average_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    timestamp: float


@dataclass
class BalanceRecord:
    """Balance record for reconciliation."""

    asset: str
    free: float
    locked: float
    total: float
    timestamp: float


@dataclass
class OrderRecord:
    """Order record for reconciliation."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    filled_quantity: float
    status: str
    timestamp: float


@dataclass
class ReconciliationResult:
    """Result of reconciliation process."""

    timestamp: float
    positions_match: bool
    balances_match: bool
    orders_match: bool
    discrepancies: Dict[str, Any]
    summary: Dict[str, Any]


class ReconciliationError(Exception):
    """Exception raised when reconciliation fails."""

    pass


class BaseReconciler:
    """Base class for reconciliation operations."""

    def __init__(self, name: str):
        """Initialize reconciler.

        Args:
            name: Name identifier for this reconciler
        """
        self.name = name
        self.last_reconciliation = 0.0
        self.reconciliation_count = 0

    async def reconcile(self) -> ReconciliationResult:
        """Perform reconciliation.

        Returns:
            Reconciliation result
        """
        raise NotImplementedError

    def get_last_result(self) -> Optional[ReconciliationResult]:
        """Get last reconciliation result."""
        return None

    def should_reconcile(self, interval_seconds: float = 300) -> bool:
        """Check if reconciliation should be performed.

        Args:
            interval_seconds: Minimum interval between reconciliations

        Returns:
            True if reconciliation should run
        """
        return time.time() - self.last_reconciliation >= interval_seconds


class PositionReconciler(BaseReconciler):
    """Reconciler for trading positions."""

    def __init__(self, name: str = "positions"):
        """Initialize position reconciler."""
        super().__init__(name)
        self._last_result: Optional[ReconciliationResult] = None

    async def reconcile(
        self,
        local_positions: List[PositionRecord],
        external_positions: List[PositionRecord],
        tolerance: float = 0.001,
    ) -> ReconciliationResult:
        """Reconcile local vs external positions.

        Args:
            local_positions: Positions from local system
            external_positions: Positions from external source
            tolerance: Acceptable difference tolerance

        Returns:
            Reconciliation result
        """
        self.last_reconciliation = time.time()
        self.reconciliation_count += 1

        discrepancies = {}
        matches = 0
        total = 0

        # Convert to dicts for easier comparison
        local_dict = {p.symbol: p for p in local_positions}
        external_dict = {p.symbol: p for p in external_positions}

        all_symbols = set(local_dict.keys()) | set(external_dict.keys())

        for symbol in all_symbols:
            total += 1
            local_pos = local_dict.get(symbol)
            external_pos = external_dict.get(symbol)

            if local_pos is None:
                discrepancies[f"missing_local_{symbol}"] = {
                    "external": external_pos,
                    "difference": "missing locally",
                }
                continue

            if external_pos is None:
                discrepancies[f"missing_external_{symbol}"] = {
                    "local": local_pos,
                    "difference": "missing externally",
                }
                continue

            # Compare quantities
            qty_diff = abs(local_pos.quantity - external_pos.quantity)
            if qty_diff > tolerance:
                discrepancies[f"quantity_{symbol}"] = {
                    "local": local_pos.quantity,
                    "external": external_pos.quantity,
                    "difference": qty_diff,
                }

            # Compare market values
            value_diff = abs(local_pos.market_value - external_pos.market_value)
            if value_diff > tolerance:
                discrepancies[f"value_{symbol}"] = {
                    "local": local_pos.market_value,
                    "external": external_pos.market_value,
                    "difference": value_diff,
                }

            if qty_diff <= tolerance and value_diff <= tolerance:
                matches += 1

        positions_match = len(discrepancies) == 0

        result = ReconciliationResult(
            timestamp=self.last_reconciliation,
            positions_match=positions_match,
            balances_match=True,  # Not checking balances here
            orders_match=True,  # Not checking orders here
            discrepancies=discrepancies,
            summary={
                "total_positions": total,
                "matching_positions": matches,
                "discrepancies": len(discrepancies),
                "match_rate": matches / total if total > 0 else 0,
            },
        )

        self._last_result = result

        if not positions_match:
            logger.warning(
                f"Position reconciliation found {len(discrepancies)} discrepancies"
            )
        else:
            logger.info("Position reconciliation passed")

        return result

    def get_last_result(self) -> Optional[ReconciliationResult]:
        """Get last reconciliation result."""
        return self._last_result


class BalanceReconciler(BaseReconciler):
    """Reconciler for account balances."""

    def __init__(self, name: str = "balances"):
        """Initialize balance reconciler."""
        super().__init__(name)
        self._last_result: Optional[ReconciliationResult] = None

    async def reconcile(
        self,
        local_balances: List[BalanceRecord],
        external_balances: List[BalanceRecord],
        tolerance: float = 0.001,
    ) -> ReconciliationResult:
        """Reconcile local vs external balances.

        Args:
            local_balances: Balances from local system
            external_balances: Balances from external source
            tolerance: Acceptable difference tolerance

        Returns:
            Reconciliation result
        """
        self.last_reconciliation = time.time()
        self.reconciliation_count += 1

        discrepancies = {}
        matches = 0
        total = 0

        # Convert to dicts
        local_dict = {b.asset: b for b in local_balances}
        external_dict = {b.asset: b for b in external_balances}

        all_assets = set(local_dict.keys()) | set(external_dict.keys())

        for asset in all_assets:
            total += 1
            local_bal = local_dict.get(asset)
            external_bal = external_dict.get(asset)

            if local_bal is None:
                discrepancies[f"missing_local_{asset}"] = {
                    "external": external_bal,
                    "difference": "missing locally",
                }
                continue

            if external_bal is None:
                discrepancies[f"missing_external_{asset}"] = {
                    "local": external_bal,
                    "difference": "missing externally",
                }
                continue

            # Compare total balances
            total_diff = abs(local_bal.total - external_bal.total)
            if total_diff > tolerance:
                discrepancies[f"total_{asset}"] = {
                    "local": local_bal.total,
                    "external": external_bal.total,
                    "difference": total_diff,
                }

            # Compare free balances
            free_diff = abs(local_bal.free - external_bal.free)
            if free_diff > tolerance:
                discrepancies[f"free_{asset}"] = {
                    "local": local_bal.free,
                    "external": external_bal.free,
                    "difference": free_diff,
                }

            if total_diff <= tolerance and free_diff <= tolerance:
                matches += 1

        balances_match = len(discrepancies) == 0

        result = ReconciliationResult(
            timestamp=self.last_reconciliation,
            positions_match=True,  # Not checking positions here
            balances_match=balances_match,
            orders_match=True,  # Not checking orders here
            discrepancies=discrepancies,
            summary={
                "total_balances": total,
                "matching_balances": matches,
                "discrepancies": len(discrepancies),
                "match_rate": matches / total if total > 0 else 0,
            },
        )

        self._last_result = result

        if not balances_match:
            logger.warning(
                f"Balance reconciliation found {len(discrepancies)} discrepancies"
            )
        else:
            logger.info("Balance reconciliation passed")

        return result

    def get_last_result(self) -> Optional[ReconciliationResult]:
        """Get last reconciliation result."""
        return self._last_result


class OrderReconciler(BaseReconciler):
    """Reconciler for order states."""

    def __init__(self, name: str = "orders"):
        """Initialize order reconciler."""
        super().__init__(name)
        self._last_result: Optional[ReconciliationResult] = None

    async def reconcile(
        self, local_orders: List[OrderRecord], external_orders: List[OrderRecord]
    ) -> ReconciliationResult:
        """Reconcile local vs external orders.

        Args:
            local_orders: Orders from local system
            external_orders: Orders from external source

        Returns:
            Reconciliation result
        """
        self.last_reconciliation = time.time()
        self.reconciliation_count += 1

        discrepancies = {}
        matches = 0
        total = 0

        # Convert to dicts
        local_dict = {o.order_id: o for o in local_orders}
        external_dict = {o.order_id: o for o in external_orders}

        all_order_ids = set(local_dict.keys()) | set(external_dict.keys())

        for order_id in all_order_ids:
            total += 1
            local_order = local_dict.get(order_id)
            external_order = external_dict.get(order_id)

            if local_order is None:
                discrepancies[f"missing_local_{order_id}"] = {
                    "external": external_order,
                    "difference": "missing locally",
                }
                continue

            if external_order is None:
                discrepancies[f"missing_external_{order_id}"] = {
                    "local": local_order,
                    "difference": "missing externally",
                }
                continue

            # Compare filled quantities
            fill_diff = abs(
                local_order.filled_quantity - external_order.filled_quantity
            )
            if fill_diff > 0.001:  # Small tolerance for fills
                discrepancies[f"fill_{order_id}"] = {
                    "local": local_order.filled_quantity,
                    "external": external_order.filled_quantity,
                    "difference": fill_diff,
                }

            # Compare statuses
            if local_order.status != external_order.status:
                discrepancies[f"status_{order_id}"] = {
                    "local": local_order.status,
                    "external": external_order.status,
                    "difference": "status mismatch",
                }

            if fill_diff <= 0.001 and local_order.status == external_order.status:
                matches += 1

        orders_match = len(discrepancies) == 0

        result = ReconciliationResult(
            timestamp=self.last_reconciliation,
            positions_match=True,  # Not checking positions here
            balances_match=True,  # Not checking balances here
            orders_match=orders_match,
            discrepancies=discrepancies,
            summary={
                "total_orders": total,
                "matching_orders": matches,
                "discrepancies": len(discrepancies),
                "match_rate": matches / total if total > 0 else 0,
            },
        )

        self._last_result = result

        if not orders_match:
            logger.warning(
                f"Order reconciliation found {len(discrepancies)} discrepancies"
            )
        else:
            logger.info("Order reconciliation passed")

        return result

    def get_last_result(self) -> Optional[ReconciliationResult]:
        """Get last reconciliation result."""
        return self._last_result


class ComprehensiveReconciler:
    """Comprehensive reconciler that combines multiple reconciliation types."""

    def __init__(self):
        """Initialize comprehensive reconciler."""
        self.position_reconciler = PositionReconciler()
        self.balance_reconciler = BalanceReconciler()
        self.order_reconciler = OrderReconciler()
        self._last_result: Optional[ReconciliationResult] = None

    async def reconcile_all(
        self,
        local_positions: List[PositionRecord] = None,
        external_positions: List[PositionRecord] = None,
        local_balances: List[BalanceRecord] = None,
        external_balances: List[BalanceRecord] = None,
        local_orders: List[OrderRecord] = None,
        external_orders: List[OrderRecord] = None,
    ) -> ReconciliationResult:
        """Perform comprehensive reconciliation.

        Args:
            local_positions: Local position records
            external_positions: External position records
            local_balances: Local balance records
            external_balances: External balance records
            local_orders: Local order records
            external_orders: External order records

        Returns:
            Comprehensive reconciliation result
        """
        timestamp = time.time()

        # Perform individual reconciliations
        position_result = None
        balance_result = None
        order_result = None

        if local_positions is not None and external_positions is not None:
            position_result = await self.position_reconciler.reconcile(
                local_positions, external_positions
            )

        if local_balances is not None and external_balances is not None:
            balance_result = await self.balance_reconciler.reconcile(
                local_balances, external_balances
            )

        if local_orders is not None and external_orders is not None:
            order_result = await self.order_reconciler.reconcile(
                local_orders, external_orders
            )

        # Combine results
        all_discrepancies = {}
        summary = {}

        if position_result:
            all_discrepancies.update(position_result.discrepancies)
            summary.update(
                {f"positions_{k}": v for k, v in position_result.summary.items()}
            )

        if balance_result:
            all_discrepancies.update(balance_result.discrepancies)
            summary.update(
                {f"balances_{k}": v for k, v in balance_result.summary.items()}
            )

        if order_result:
            all_discrepancies.update(order_result.discrepancies)
            summary.update({f"orders_{k}": v for k, v in order_result.summary.items()})

        comprehensive_result = ReconciliationResult(
            timestamp=timestamp,
            positions_match=position_result.positions_match
            if position_result
            else True,
            balances_match=balance_result.balances_match if balance_result else True,
            orders_match=order_result.orders_match if order_result else True,
            discrepancies=all_discrepancies,
            summary=summary,
        )

        self._last_result = comprehensive_result

        total_discrepancies = len(all_discrepancies)
        if total_discrepancies > 0:
            logger.warning(
                f"Comprehensive reconciliation found {total_discrepancies} total discrepancies"
            )
        else:
            logger.info("Comprehensive reconciliation passed")

        return comprehensive_result

    def get_last_result(self) -> Optional[ReconciliationResult]:
        """Get last comprehensive reconciliation result."""
        return self._last_result


# Global reconciler instance
_reconciler = ComprehensiveReconciler()


def get_reconciler() -> ComprehensiveReconciler:
    """Get global reconciler instance."""
    return _reconciler


# Coincheck-specific hooks for future integration
def coincheck_position_reconciliation_hook(
    local_positions: List[PositionRecord], broker_positions: List[PositionRecord]
) -> List[PositionRecord]:
    """Position reconciliation hook for Coincheck (placeholder for future implementation)."""
    # TODO: Implement Coincheck-specific position reconciliation
    logger.debug("Coincheck position reconciliation hook called")
    return broker_positions


def coincheck_balance_reconciliation_hook(
    local_balances: List[BalanceRecord], broker_balances: List[BalanceRecord]
) -> List[BalanceRecord]:
    """Balance reconciliation hook for Coincheck (placeholder for future implementation)."""
    # TODO: Implement Coincheck-specific balance reconciliation
    logger.debug("Coincheck balance reconciliation hook called")
    return broker_balances


def coincheck_order_reconciliation_hook(
    local_orders: List[OrderRecord], broker_orders: List[OrderRecord]
) -> Tuple[List[OrderRecord], List[str]]:
    """Order reconciliation hook for Coincheck (placeholder for future implementation)."""
    # TODO: Implement Coincheck-specific order reconciliation
    logger.debug("Coincheck order reconciliation hook called")
    return broker_orders, []
