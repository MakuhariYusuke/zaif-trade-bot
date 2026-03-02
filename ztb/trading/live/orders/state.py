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
from typing import Any

from ztb.trading.live.orders.compat import OrderData
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class OrderState(enum.Enum):
    """Order states in the lifecycle."""

    CREATED = "created"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
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
class OrderRecord:
    """Complete order record with state."""

    data: OrderData
    state: OrderState = OrderState.CREATED
    filled_quantity: float = 0.0
    average_price: float = 0.0
    fees: float = 0.0
    last_update: float = field(default_factory=time.time)
    error_message: str | None = None
    external_order_id: str | None = None
    idempotency_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate idempotency key if not provided."""
        if not self.idempotency_key:
            self.idempotency_key = self._generate_idempotency_key()

    def _generate_idempotency_key(self) -> str:
        """Generate idempotency key from order data."""
        key_data = (
            f"{self.data.client_order_id}_{self.data.symbol}_{self.data.timestamp}"
        )
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def generate_order_id(self) -> str:
        return f"ord_{uuid.uuid4().hex[:12]}"

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
            OrderState.CREATED: {OrderState.SUBMITTED, OrderState.CANCELLED},
            OrderState.SUBMITTED: {
                OrderState.FILLED,
                OrderState.PARTIAL,
                OrderState.CANCELLED,
                OrderState.REJECTED,
                OrderState.EXPIRED,
            },
            OrderState.PARTIAL: {OrderState.FILLED, OrderState.CANCELLED},
            OrderState.FILLED: set(),
            OrderState.CANCELLED: set(),
            OrderState.REJECTED: set(),
            OrderState.EXPIRED: set(),
            OrderState.FAILED: set(),
        }
        return new_state in valid_transitions.get(self.state, set())

class OrderStateMachine:
    """State machine for managing multiple orders."""

    def __init__(self) -> None:
        self.orders: dict[str, OrderRecord] = {}

    def transition_order(self, order_id: str, event: OrderEvent, **kwargs: Any) -> bool:
        """Transition order to new state based on event.

        Args:
            order_id: Order ID
            event: State transition event
            **kwargs: Additional data for transition

        Returns:
            True if transition was successful, False otherwise
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False

        record = self.orders[order_id]

        # Determine new state based on event
        state_transitions = {
            OrderEvent.SUBMIT: OrderState.SUBMITTED,
            OrderEvent.FILL: OrderState.FILLED,
            OrderEvent.PARTIAL_FILL: OrderState.PARTIAL,
            OrderEvent.CANCEL: OrderState.CANCELLED,
            OrderEvent.REJECT: OrderState.REJECTED,
            OrderEvent.EXPIRE: OrderState.EXPIRED,
            OrderEvent.FAIL: OrderState.FAILED,
        }

        new_state = state_transitions.get(event)
        if new_state is None:
            logger.warning(f"Unknown event {event} for order {order_id}")
            return False

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

# Global order state machine instance
_order_state_machine = OrderStateMachine()
