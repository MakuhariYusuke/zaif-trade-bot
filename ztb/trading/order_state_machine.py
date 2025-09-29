"""
Order State Machine with Idempotency

Provides state machine for order lifecycle management with idempotency
to prevent duplicate orders and ensure reliable execution.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from ztb.utils.observability import get_logger

logger = get_logger(__name__)


class OrderState(Enum):
    """Order states in the lifecycle."""
    PENDING = "pending"      # Order submitted but not confirmed
    CONFIRMED = "confirmed"  # Order confirmed by exchange
    PARTIAL = "partial"      # Partially filled
    FILLED = "filled"        # Fully filled
    CANCELLED = "cancelled"  # Cancelled by user
    REJECTED = "rejected"    # Rejected by exchange
    EXPIRED = "expired"      # Expired
    FAILED = "failed"        # Failed to submit


class OrderEvent(Enum):
    """Events that can trigger state transitions."""
    SUBMIT = "submit"
    CONFIRM = "confirm"
    FILL = "fill"
    PARTIAL_FILL = "partial_fill"
    CANCEL = "cancel"
    REJECT = "reject"
    EXPIRE = "expire"
    FAIL = "fail"


@dataclass
class OrderData:
    """Order data structure."""
    order_id: str
    client_order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float] = None
    order_type: str = "market"
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class OrderStateMachine:
    """State machine for order lifecycle with idempotency."""

    # State transition table
    TRANSITIONS = {
        OrderState.PENDING: {
            OrderEvent.CONFIRM: OrderState.CONFIRMED,
            OrderEvent.REJECT: OrderState.REJECTED,
            OrderEvent.FAIL: OrderState.FAILED,
            OrderEvent.CANCEL: OrderState.CANCELLED,
            OrderEvent.EXPIRE: OrderState.EXPIRED,
        },
        OrderState.CONFIRMED: {
            OrderEvent.FILL: OrderState.FILLED,
            OrderEvent.PARTIAL_FILL: OrderState.PARTIAL,
            OrderEvent.CANCEL: OrderState.CANCELLED,
            OrderEvent.EXPIRE: OrderState.EXPIRED,
        },
        OrderState.PARTIAL: {
            OrderEvent.FILL: OrderState.FILLED,
            OrderEvent.PARTIAL_FILL: OrderState.PARTIAL,  # Stay in partial
            OrderEvent.CANCEL: OrderState.CANCELLED,
            OrderEvent.EXPIRE: OrderState.EXPIRED,
        },
        OrderState.FILLED: {},  # Terminal state
        OrderState.CANCELLED: {},  # Terminal state
        OrderState.REJECTED: {},  # Terminal state
        OrderState.EXPIRED: {},  # Terminal state
        OrderState.FAILED: {},  # Terminal state
    }

    def __init__(self, order_data: OrderData):
        """
        Initialize state machine.

        Args:
            order_data: Order data
        """
        self.order_data = order_data
        self.current_state = OrderState.PENDING
        self.state_history: list[tuple[OrderState, float]] = [(OrderState.PENDING, time.time())]
        self.idempotency_key = f"{order_data.client_order_id}_{order_data.symbol}"

    def transition(self, event: OrderEvent, **kwargs) -> bool:
        """
        Attempt state transition.

        Args:
            event: Event triggering transition
            **kwargs: Additional event data

        Returns:
            True if transition successful, False otherwise
        """
        if event not in self.TRANSITIONS[self.current_state]:
            logger.warning(f"Invalid transition {event} from {self.current_state} for order {self.order_data.order_id}")
            return False

        new_state = self.TRANSITIONS[self.current_state][event]
        old_state = self.current_state
        self.current_state = new_state
        self.state_history.append((new_state, time.time()))

        logger.info(f"Order {self.order_data.order_id} transitioned: {old_state} -> {new_state} ({event})")
        return True

    def can_transition(self, event: OrderEvent) -> bool:
        """Check if transition is valid."""
        return event in self.TRANSITIONS[self.current_state]

    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        return len(self.TRANSITIONS[self.current_state]) == 0

    def get_state(self) -> OrderState:
        """Get current state."""
        return self.current_state

    def get_state_history(self) -> list[tuple[OrderState, float]]:
        """Get state transition history."""
        return self.state_history.copy()


class IdempotencyManager:
    """Manages order idempotency to prevent duplicates."""

    def __init__(self):
        """Initialize idempotency manager."""
        self._processed_keys: set[str] = set()
        self._order_states: dict[str, OrderStateMachine] = {}

    def is_idempotent(self, idempotency_key: str) -> bool:
        """
        Check if operation is idempotent.

        Args:
            idempotency_key: Unique key for operation

        Returns:
            True if operation can proceed, False if duplicate
        """
        if idempotency_key in self._processed_keys:
            logger.warning(f"Duplicate operation detected for key: {idempotency_key}")
            return False
        return True

    def mark_processed(self, idempotency_key: str):
        """Mark operation as processed."""
        self._processed_keys.add(idempotency_key)

    def get_order_state_machine(self, order_id: str) -> Optional[OrderStateMachine]:
        """Get state machine for order."""
        return self._order_states.get(order_id)

    def register_order(self, state_machine: OrderStateMachine):
        """Register order state machine."""
        self._order_states[state_machine.order_data.order_id] = state_machine

    def cleanup_expired(self, max_age_seconds: float = 3600):
        """Clean up expired state machines."""
        cutoff = time.time() - max_age_seconds
        expired_orders = []

        for order_id, sm in self._order_states.items():
            if sm.state_history[-1][1] < cutoff and sm.is_terminal():
                expired_orders.append(order_id)

        for order_id in expired_orders:
            del self._order_states[order_id]

        if expired_orders:
            logger.info(f"Cleaned up {len(expired_orders)} expired order state machines")


# Global instance
_idempotency_manager: Optional[IdempotencyManager] = None


def get_idempotency_manager() -> IdempotencyManager:
    """Get global idempotency manager instance."""
    global _idempotency_manager
    if _idempotency_manager is None:
        _idempotency_manager = IdempotencyManager()
    return _idempotency_manager