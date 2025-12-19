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

from ztb.trading.live.core.precision_policy import quantize_price, quantize_quantity
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)




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
            return False

        record = self.orders[order_id]

        # Determine new state based on event
            OrderEvent.PARTIAL_FILL: OrderState.PARTIAL_FILL,
            OrderEvent.CANCEL: OrderState.CANCELLED,
            OrderEvent.REJECT: OrderState.REJECTED,
            OrderEvent.EXPIRE: OrderState.EXPIRED,
            OrderEvent.FAIL: OrderState.FAILED,
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

        f"Coincheck reconciliation hook called for order {order_data.order_id}"
    )
    return broker_state


# Global order state machine instance
_order_state_machine = OrderStateMachine()
