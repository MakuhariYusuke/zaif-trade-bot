"""Unit tests for idempotency store."""

import os
import tempfile

import pytest

from ztb.live.idempotency_store import IdempotencyStore


class TestIdempotencyStore:
    """Test idempotency store functionality."""

    def test_check_and_store_new_order(self):
        """Test storing a new order."""
        store = IdempotencyStore(":memory:")

        order_data = {
            "symbol": "BTC_JPY",
            "side": "buy",
            "quantity": 0.001,
            "price": 5000000.0,
        }

        # Should succeed for new order
        assert store.check_and_store("order_123", order_data) is True

        # Should fail for duplicate
        assert store.check_and_store("order_123", order_data) is False

    def test_get_order_data(self):
        """Test retrieving stored order data."""
        store = IdempotencyStore(":memory:")

        order_data = {
            "symbol": "BTC_JPY",
            "side": "buy",
            "quantity": 0.001,
            "price": 5000000.0,
        }

        store.check_and_store("order_456", order_data)
        retrieved = store.get_order_data("order_456")

        assert retrieved == order_data

        # Non-existent order
        assert store.get_order_data("nonexistent") is None

    def test_update_status(self):
        """Test updating order status."""
        store = IdempotencyStore(":memory:")

        order_data = {"symbol": "BTC_JPY", "side": "buy", "quantity": 0.001}
        store.check_and_store("order_789", order_data)

        # Update status
        assert store.update_status("order_789", "filled") is True

        # Verify status in list_orders
        orders = store.list_orders()
        assert len(orders) == 1
        assert orders[0]["status"] == "filled"

        # Update non-existent order
        assert store.update_status("nonexistent", "cancelled") is False

    def test_list_orders(self):
        """Test listing orders with filters."""
        store = IdempotencyStore(":memory:")

        # Add multiple orders
        store.check_and_store("order_1", {"symbol": "BTC_JPY"})
        store.check_and_store("order_2", {"symbol": "ETH_JPY"})
        store.update_status("order_1", "filled")

        # List all
        all_orders = store.list_orders()
        assert len(all_orders) == 2

        # List by status
        filled_orders = store.list_orders(status="filled")
        assert len(filled_orders) == 1
        assert filled_orders[0]["client_order_id"] == "order_1"

        pending_orders = store.list_orders(status="pending")
        assert len(pending_orders) == 1
        assert pending_orders[0]["client_order_id"] == "order_2"

    def test_cleanup_old_orders(self):
        """Test cleaning up old orders."""
        store = IdempotencyStore(":memory:")

        # Add an order
        store.check_and_store("old_order", {"symbol": "BTC_JPY"})

        # Cleanup with 0 days (should remove the order)
        deleted = store.cleanup_old_orders(days_old=0)
        assert deleted >= 1

        # Verify order is gone
        assert store.get_order_data("old_order") is None

    def test_get_stats(self):
        """Test getting store statistics."""
        store = IdempotencyStore(":memory:")

        # Add orders with different statuses
        store.check_and_store("order_1", {"symbol": "BTC_JPY"})
        store.check_and_store("order_2", {"symbol": "ETH_JPY"})
        store.update_status("order_1", "filled")

        stats = store.get_stats()

        assert stats["total_orders"] == 2
        assert stats["status_breakdown"]["pending"] == 1
        assert stats["status_breakdown"]["filled"] == 1

    def test_persistent_storage(self):
        """Test persistence across store instances."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        try:
            # Create store and add data
            store1 = IdempotencyStore(db_path)
            store1.check_and_store("persistent_order", {"symbol": "BTC_JPY"})

            # Create new store instance
            store2 = IdempotencyStore(db_path)
            retrieved = store2.get_order_data("persistent_order")

            assert retrieved is not None
            assert retrieved["symbol"] == "BTC_JPY"

        finally:
            os.unlink(db_path)

    def test_duplicate_order_error_handling(self):
        """Test proper error handling for duplicate orders."""
        from ztb.live.order_state import create_order_with_idempotency

        # First order should succeed
        try:
            result1 = create_order_with_idempotency(
                symbol="BTC_JPY",
                side="buy",
                quantity=0.001,
                client_order_id="duplicate_test",
            )
            # Should not raise error
        except ValueError:
            pytest.fail("First order with unique ID should not raise error")

        # Second order with same ID should raise ValueError
        with pytest.raises(ValueError, match="already exists"):
            create_order_with_idempotency(
                symbol="BTC_JPY",
                side="sell",  # Different side
                quantity=0.002,  # Different quantity
                client_order_id="duplicate_test",  # Same ID
            )
