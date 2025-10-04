"""
Order state machine and idempotency management.

This module provides state management for trading orders with idempotency
guarantees to prevent duplicate orders and ensure reliable execution.
"""

import enum
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional, Set

from ztb.trading.live.precision_policy import quantize_price, quantize_quantity
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OrderState(enum.Enum):
    """Order states in the lifecycle."""

    CREATED = "created"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderEvent(enum.Enum):
    """Events that can trigger order state transitions."""

    SUBMIT = "submit"
    ACCEPT = "accept"
    FILL = "fill"
    PARTIAL_FILL = "partial_fill"
    CANCEL = "cancel"
    REJECT = "reject"
    EXPIRE = "expire"
    FAIL = "fail"
    RESET = "reset"


@dataclass
class OrderData:
    """Order data structure."""

    order_id: str
    client_order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""  # buy/sell
    order_type: str = "market"
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderRecord:
    """Complete order record with state."""

    data: OrderData
    state: OrderState = OrderState.CREATED
    filled_quantity: float = 0.0
    average_price: float = 0.0
    fees: float = 0.0
    last_update: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    external_order_id: Optional[str] = None
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        """Generate idempotency key if not provided."""
        if not self.idempotency_key:
            self.idempotency_key = self._generate_idempotency_key()

    def _generate_idempotency_key(self) -> str:
        """Generate idempotency key from order data."""
        key_data = f"{self.data.symbol}:{self.data.side}:{self.data.quantity}:{self.data.price or 0}:{self.data.timestamp}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def is_terminal_state(self) -> bool:
        """Check if order is in a terminal state."""
        return self.state in {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.FAILED,
        }

    def can_transition_to(self, new_state: OrderState) -> bool:
        """Check if transition to new state is valid."""
        # Define valid transitions
        valid_transitions = {
            OrderState.CREATED: {
                OrderState.SUBMITTED,
                OrderState.CANCELLED,
                OrderState.FAILED,
            },
            OrderState.SUBMITTED: {
                OrderState.ACCEPTED,
                OrderState.REJECTED,
                OrderState.CANCELLED,
                OrderState.FAILED,
            },
            OrderState.ACCEPTED: {
                OrderState.PARTIAL_FILL,
                OrderState.FILLED,
                OrderState.CANCELLED,
                OrderState.EXPIRED,
                OrderState.FAILED,
            },
            OrderState.PARTIAL_FILL: {
                OrderState.PARTIAL_FILL,
                OrderState.FILLED,
                OrderState.CANCELLED,
                OrderState.EXPIRED,
                OrderState.FAILED,
            },
            OrderState.FILLED: set(),  # Terminal
            OrderState.CANCELLED: set(),  # Terminal
            OrderState.REJECTED: set(),  # Terminal
            OrderState.EXPIRED: set(),  # Terminal
            OrderState.FAILED: set(),  # Terminal
        }

        return new_state in valid_transitions.get(self.state, set())


class OrderStateMachine:
    """State machine for managing order lifecycle."""

    def __init__(self) -> None:
        """Initialize state machine."""
        super().__init__()
        self.orders: Dict[str, OrderRecord] = {}
        self.idempotency_map: Dict[str, str] = {}  # idempotency_key -> order_id

    def create_order(self, order_data: OrderData) -> OrderRecord:
        """Create a new order.

        Args:
            order_data: Order data

        Returns:
            Order record

        Raises:
            ValueError: If order with same idempotency key exists
        """
        record = OrderRecord(data=order_data)

        # Check idempotency
        if record.idempotency_key in self.idempotency_map:
            existing_id = self.idempotency_map[record.idempotency_key]
            existing_order = self.orders.get(existing_id)
            if existing_order and not existing_order.is_terminal_state():
                raise ValueError(
                    f"Order with idempotency key {record.idempotency_key} already exists"
                )

        self.orders[record.data.order_id] = record
        self.idempotency_map[record.idempotency_key] = record.data.order_id

        logger.info(
            f"Created order {record.data.order_id} with state {record.state.value}"
        )
        return record

    def transition_order(self, order_id: str, event: OrderEvent, **kwargs: Any) -> bool:
        """Transition order to new state based on event.

        Args:
            order_id: Order ID
            event: State transition event
            **kwargs: Additional data for transition

        Returns:
            True if transition successful
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False

        record = self.orders[order_id]

        # Determine new state based on event
        state_transitions = {
            OrderEvent.SUBMIT: OrderState.SUBMITTED,
            OrderEvent.ACCEPT: OrderState.ACCEPTED,
            OrderEvent.FILL: OrderState.FILLED,
            OrderEvent.PARTIAL_FILL: OrderState.PARTIAL_FILL,
            OrderEvent.CANCEL: OrderState.CANCELLED,
            OrderEvent.REJECT: OrderState.REJECTED,
            OrderEvent.EXPIRE: OrderState.EXPIRED,
            OrderEvent.FAIL: OrderState.FAILED,
            OrderEvent.RESET: OrderState.CREATED,
        }

        if event not in state_transitions:
            logger.error(f"Unknown event: {event}")
            return False

        new_state = state_transitions[event]

        # Validate transition
        if not record.can_transition_to(new_state):
            logger.warning(
                f"Invalid transition from {record.state.value} to {new_state.value} for order {order_id}"
            )
            return False

        # Apply transition
        old_state = record.state
        record.state = new_state
        record.last_update = time.time()

        # Update additional data
        if event == OrderEvent.FILL or event == OrderEvent.PARTIAL_FILL:
            if "filled_quantity" in kwargs:
                record.filled_quantity = kwargs["filled_quantity"]
            if "average_price" in kwargs:
                record.average_price = kwargs["average_price"]
            if "fees" in kwargs:
                record.fees = kwargs["fees"]

        if "external_order_id" in kwargs:
            record.external_order_id = kwargs["external_order_id"]

        if "error_message" in kwargs:
            record.error_message = kwargs["error_message"]

        logger.info(
            f"Order {order_id} transitioned from {old_state.value} to {new_state.value}"
        )
        return True

    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        """Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order record or None
        """
        return self.orders.get(order_id)

    def get_order_by_idempotency_key(self, key: str) -> Optional[OrderRecord]:
        """Get order by idempotency key.

        Args:
            key: Idempotency key

        Returns:
            Order record or None
        """
        order_id = self.idempotency_map.get(key)
        if order_id:
            return self.orders.get(order_id)
        return None

    def list_orders(
        self, state_filter: Optional[Set[OrderState]] = None
    ) -> list[OrderRecord]:
        """List orders, optionally filtered by state.

        Args:
            state_filter: Set of states to include

        Returns:
            List of order records
        """
        if state_filter is None:
            return list(self.orders.values())
        return [order for order in self.orders.values() if order.state in state_filter]

    def cleanup_terminal_orders(self, max_age_seconds: float = 86400) -> int:
        """Clean up terminal orders older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of orders cleaned up
        """
        cutoff_time = time.time() - max_age_seconds
        to_remove = []

        for order_id, record in self.orders.items():
            if record.is_terminal_state() and record.last_update < cutoff_time:
                to_remove.append(order_id)

        for order_id in to_remove:
            record = self.orders[order_id]
            del self.idempotency_map[record.idempotency_key]
            del self.orders[order_id]

        logger.info(f"Cleaned up {len(to_remove)} terminal orders")
        return len(to_remove)

    def get_order_summary(self) -> Dict[str, int]:
        """Get summary of orders by state.

        Returns:
            Dictionary of state counts
        """
        summary = {}
        for state in OrderState:
            summary[state.value] = 0

        for record in self.orders.values():
            summary[record.state.value] += 1

        return summary


# Global state machine instance
_order_state_machine = OrderStateMachine()


def get_order_state_machine() -> OrderStateMachine:
    """Get global order state machine instance."""
    return _order_state_machine


def generate_order_id() -> str:
    """Generate a unique order ID.

    Returns:
        Order ID string
    """
    return f"ord_{uuid.uuid4().hex[:12]}"


def create_order_with_idempotency(
    symbol: str,
    side: str,
    quantity: float,
    price: Optional[float] = None,
    venue: str = "coincheck",
    **kwargs: Any,
) -> OrderRecord:
    """Create order with automatic idempotency handling.

    Args:
        symbol: Trading symbol
        side: Order side (buy/sell)
        quantity: Order quantity
        price: Order price (None for market orders)
        venue: Trading venue for precision policy
        **kwargs: Additional order parameters

    Returns:
        Order record
    """
    # Apply precision policy
    if price is not None:
        price = float(quantize_price(venue, symbol, Decimal(str(price))))
    quantity = float(quantize_quantity(venue, symbol, Decimal(str(quantity))))

    order_data = OrderData(
        order_id=generate_order_id(),
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        **kwargs,
    )

    return _order_state_machine.create_order(order_data)


def find_existing_order(
    symbol: str,
    side: str,
    quantity: float,
    price: Optional[float] = None,
    timestamp: Optional[float] = None,
) -> Optional[OrderRecord]:
    """Find existing order by key parameters.

    Args:
        symbol: Trading symbol
        side: Order side
        quantity: Order quantity
        price: Order price
        timestamp: Order timestamp

    Returns:
        Existing order record or None
    """
    if timestamp is None:
        timestamp = time.time()

    key_data = f"{symbol}:{side}:{quantity}:{price or 0}:{timestamp}"
    idempotency_key = hashlib.sha256(key_data.encode()).hexdigest()[:16]

    return _order_state_machine.get_order_by_idempotency_key(idempotency_key)


# Coincheck-specific hooks for future integration
def coincheck_order_pre_submit_hook(order_data: OrderData) -> OrderData:
    """Pre-submit hook for Coincheck orders (placeholder for future implementation)."""
    # TODO: Add Coincheck-specific validation or transformation
    logger.debug(f"Coincheck pre-submit hook called for order {order_data.order_id}")
    return order_data


def coincheck_order_post_submit_hook(
    order_data: OrderData, _broker_response: Any
) -> None:
    """Post-submit hook for Coincheck orders (placeholder for future implementation)."""
    # TODO: Handle Coincheck-specific response processing
    logger.debug(f"Coincheck post-submit hook called for order {order_data.order_id}")


def coincheck_reconciliation_hook(
    order_data: OrderData, broker_state: Dict[str, Any]
) -> Dict[str, Any]:
    """Reconciliation hook for Coincheck orders (placeholder for future implementation)."""
    # TODO: Implement Coincheck-specific reconciliation logic
    logger.debug(
        f"Coincheck reconciliation hook called for order {order_data.order_id}"
    )
    return broker_state


# Global order state machine instance
_order_state_machine = OrderStateMachine()
