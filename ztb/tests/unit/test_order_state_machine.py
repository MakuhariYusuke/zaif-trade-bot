"""
Unit tests for order state machine
注文ステートマシンの単体テスト
"""

import time
from unittest.mock import patch

from ztb.trading.orders.state_machine import (
    IdempotencyManager,
    OrderData,
    OrderEvent,
    OrderState,
    OrderStateMachine,
    get_idempotency_manager,
)


class TestOrderState:
    def test_order_state_enum_values(self):
        """Test OrderState enum has expected values"""
        assert OrderState.PENDING.value == "pending"
        assert OrderState.CONFIRMED.value == "confirmed"
        assert OrderState.PARTIAL.value == "partial"
        assert OrderState.FILLED.value == "filled"
        assert OrderState.CANCELLED.value == "cancelled"
        assert OrderState.REJECTED.value == "rejected"
        assert OrderState.EXPIRED.value == "expired"
        assert OrderState.FAILED.value == "failed"


class TestOrderEvent:
    def test_order_event_enum_values(self):
        """Test OrderEvent enum has expected values"""
        assert OrderEvent.SUBMIT.value == "submit"
        assert OrderEvent.CONFIRM.value == "confirm"
        assert OrderEvent.FILL.value == "fill"
        assert OrderEvent.PARTIAL_FILL.value == "partial_fill"
        assert OrderEvent.CANCEL.value == "cancel"
        assert OrderEvent.REJECT.value == "reject"
        assert OrderEvent.EXPIRE.value == "expire"
        assert OrderEvent.FAIL.value == "fail"


class TestOrderData:
    def test_order_data_creation(self):
        """Test OrderData dataclass creation"""
        order_data = OrderData(
            order_id="order_123",
            client_order_id="client_123",
            symbol="BTC_JPY",
            side="buy",
            quantity=1.0,
            price=5000000.0,
            order_type="limit",
        )

        assert order_data.order_id == "order_123"
        assert order_data.client_order_id == "client_123"
        assert order_data.symbol == "BTC_JPY"
        assert order_data.side == "buy"
        assert order_data.quantity == 1.0
        assert order_data.price == 5000000.0
        assert order_data.order_type == "limit"
        assert order_data.timestamp is not None

    def test_order_data_default_timestamp(self):
        """Test OrderData sets default timestamp"""
        with patch("time.time", return_value=1234567890.0):
            order_data = OrderData(
                order_id="order_123",
                client_order_id="client_123",
                symbol="BTC_JPY",
                side="buy",
                quantity=1.0,
            )

            assert order_data.timestamp == 1234567890.0


class TestOrderStateMachine:
    def test_initialization(self):
        """Test OrderStateMachine initialization"""
        order_data = OrderData(
            order_id="order_123",
            client_order_id="client_123",
            symbol="BTC_JPY",
            side="buy",
            quantity=1.0,
        )

        sm = OrderStateMachine(order_data)

        assert sm.order_data == order_data
        assert sm.current_state == OrderState.PENDING
        assert len(sm.state_history) == 1
        assert sm.state_history[0][0] == OrderState.PENDING
        assert sm.idempotency_key == "client_123_BTC_JPY"

    def test_valid_transitions_from_pending(self):
        """Test valid state transitions from PENDING"""
        order_data = OrderData(
            order_id="order_123",
            client_order_id="client_123",
            symbol="BTC_JPY",
            side="buy",
            quantity=1.0,
        )

        sm = OrderStateMachine(order_data)

        # Test CONFIRM transition
        assert sm.transition(OrderEvent.CONFIRM)
        assert sm.current_state == OrderState.CONFIRMED

        # Reset and test REJECT transition
        sm.current_state = OrderState.PENDING
        assert sm.transition(OrderEvent.REJECT)
        assert sm.current_state == OrderState.REJECTED

        # Reset and test CANCEL transition
        sm.current_state = OrderState.PENDING
        assert sm.transition(OrderEvent.CANCEL)
        assert sm.current_state == OrderState.CANCELLED

    def test_invalid_transition(self):
        """Test invalid state transition"""
        order_data = OrderData(
            order_id="order_123",
            client_order_id="client_123",
            symbol="BTC_JPY",
            side="buy",
            quantity=1.0,
        )

        sm = OrderStateMachine(order_data)

        # Try to fill from pending (invalid)
        assert not sm.transition(OrderEvent.FILL)
        assert sm.current_state == OrderState.PENDING

    def test_can_transition(self):
        """Test can_transition method"""
        order_data = OrderData(
            order_id="order_123",
            client_order_id="client_123",
            symbol="BTC_JPY",
            side="buy",
            quantity=1.0,
        )

        sm = OrderStateMachine(order_data)

        assert sm.can_transition(OrderEvent.CONFIRM)
        assert not sm.can_transition(OrderEvent.FILL)

    def test_is_terminal(self):
        """Test is_terminal method"""
        order_data = OrderData(
            order_id="order_123",
            client_order_id="client_123",
            symbol="BTC_JPY",
            side="buy",
            quantity=1.0,
        )

        sm = OrderStateMachine(order_data)

        # PENDING is not terminal
        assert not sm.is_terminal()

        # Move to FILLED (terminal)
        sm.transition(OrderEvent.CONFIRM)
        sm.transition(OrderEvent.FILL)
        assert sm.is_terminal()

    def test_get_state_and_history(self):
        """Test get_state and get_state_history methods"""
        order_data = OrderData(
            order_id="order_123",
            client_order_id="client_123",
            symbol="BTC_JPY",
            side="buy",
            quantity=1.0,
        )

        sm = OrderStateMachine(order_data)

        assert sm.get_state() == OrderState.PENDING
        history = sm.get_state_history()
        assert len(history) == 1
        assert history[0][0] == OrderState.PENDING


class TestIdempotencyManager:
    def test_initialization(self):
        """Test IdempotencyManager initialization"""
        manager = IdempotencyManager()

        assert len(manager._processed_keys) == 0
        assert len(manager._order_states) == 0

    def test_is_idempotent(self):
        """Test is_idempotent method"""
        manager = IdempotencyManager()

        # First call should be idempotent
        assert manager.is_idempotent("key1")

        # Mark as processed
        manager.mark_processed("key1")

        # Second call should not be idempotent
        assert not manager.is_idempotent("key1")

        # Different key should be idempotent
        assert manager.is_idempotent("key2")

    def test_order_registration(self):
        """Test order state machine registration"""
        manager = IdempotencyManager()

        order_data = OrderData(
            order_id="order_123",
            client_order_id="client_123",
            symbol="BTC_JPY",
            side="buy",
            quantity=1.0,
        )

        sm = OrderStateMachine(order_data)
        manager.register_order(sm)

        retrieved_sm = manager.get_order_state_machine("order_123")
        assert retrieved_sm == sm

        # Non-existent order should return None
        assert manager.get_order_state_machine("non_existent") is None

    def test_cleanup_expired(self):
        """Test cleanup of expired orders"""
        manager = IdempotencyManager()

        # Create order and register
        order_data = OrderData(
            order_id="order_123",
            client_order_id="client_123",
            symbol="BTC_JPY",
            side="buy",
            quantity=1.0,
        )

        sm = OrderStateMachine(order_data)
        sm.transition(OrderEvent.CONFIRM)
        sm.transition(OrderEvent.FILL)  # Terminal state

        manager.register_order(sm)

        # Mock old timestamp
        sm.state_history[-1] = (OrderState.FILLED, time.time() - 7200)  # 2 hours ago

        # Cleanup should remove expired orders
        manager.cleanup_expired(max_age_seconds=3600)  # 1 hour max age

        assert manager.get_order_state_machine("order_123") is None


class TestGlobalIdempotencyManager:
    def test_get_idempotency_manager(self):
        """Test global idempotency manager getter"""
        manager1 = get_idempotency_manager()
        manager2 = get_idempotency_manager()

        # Should return the same instance
        assert manager1 is manager2
        assert isinstance(manager1, IdempotencyManager)
