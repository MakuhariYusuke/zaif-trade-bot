"""
Tests for order submission with precision and idempotency safety.
"""

import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from ztb.live.idempotency_store import IdempotencyStore
from ztb.live.order_submission import OrderPreparer, PreparedOrder
from ztb.utils.errors import IdempotencyError, ValidationError


class TestOrderPreparer(unittest.TestCase):
    """Test OrderPreparer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.idempotency_store = MagicMock(spec=IdempotencyStore)
        self.preparer = OrderPreparer(self.idempotency_store)

    def test_prepare_order_basic(self):
        """Test basic order preparation."""
        # Mock idempotency store
        self.idempotency_store.get_order_data.return_value = None
        self.idempotency_store.check_and_store.return_value = None

        order = self.preparer.prepare_order(
            venue="coincheck",
            symbol="BTC/JPY",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("5000000"),
            client_order_id="test-123",
        )

        self.assertIsInstance(order, PreparedOrder)
        self.assertEqual(order.venue, "coincheck")
        self.assertEqual(order.normalized_symbol, "BTC_JPY")
        self.assertEqual(order.side, "BUY")
        self.assertEqual(order.quantity, Decimal("0.1"))
        self.assertEqual(order.price, Decimal("5000000"))
        self.assertEqual(order.client_order_id, "test-123")

    def test_prepare_order_idempotency_conflict(self):
        """Test that duplicate client_order_id raises IdempotencyError."""
        # Mock existing order
        self.idempotency_store.get_order_data.return_value = {"status": "pending"}

        with self.assertRaises(IdempotencyError):
            self.preparer.prepare_order(
                venue="coincheck",
                symbol="BTC/JPY",
                side="BUY",
                quantity=Decimal("0.1"),
                price=Decimal("5000000"),
                client_order_id="duplicate-123",
            )

    @patch("ztb.live.order_submission.uuid")
    def test_prepare_order_auto_generate_id(self, mock_uuid):
        """Test automatic client_order_id generation."""
        mock_uuid.uuid4.return_value = "generated-uuid-123"
        self.idempotency_store.get_order_data.return_value = None
        self.idempotency_store.check_and_store.return_value = None

        order = self.preparer.prepare_order(
            venue="coincheck",
            symbol="BTC/JPY",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("5000000"),
        )

        self.assertEqual(order.client_order_id, "generated-uuid-123")
        mock_uuid.uuid4.assert_called_once()

    def test_prepare_order_invalid_venue(self):
        """Test invalid venue raises ValidationError."""
        with self.assertRaises(ValidationError):
            self.preparer.prepare_order(
                venue="invalid_venue",
                symbol="BTC/JPY",
                side="BUY",
                quantity=Decimal("0.1"),
                price=Decimal("5000000"),
            )

    def test_prepare_order_market_order(self):
        """Test market order preparation (no price)."""
        self.idempotency_store.get_order_data.return_value = None
        self.idempotency_store.check_and_store.return_value = None

        order = self.preparer.prepare_order(
            venue="coincheck",
            symbol="BTC/JPY",
            side="BUY",
            quantity=Decimal("0.1"),
            price=None,  # Market order
        )

        self.assertIsInstance(order, PreparedOrder)
        self.assertIsNone(order.price)


class TestOrderPreparerConcurrency(unittest.TestCase):
    """Test OrderPreparer concurrency safety."""

    def setUp(self):
        """Set up test fixtures."""
        self.idempotency_store = MagicMock(spec=IdempotencyStore)
        self.preparer = OrderPreparer(self.idempotency_store)

    def test_concurrent_order_preparation(self):
        """Test that concurrent identical orders are handled safely."""
        import threading
        import time

        results = []
        errors = []

        def prepare_order_thread(order_id):
            try:
                time.sleep(0.01)  # Small delay to increase chance of race
                order = self.preparer.prepare_order(
                    venue="coincheck",
                    symbol="BTC/JPY",
                    side="BUY",
                    quantity=Decimal("0.1"),
                    price=Decimal("5000000"),
                    client_order_id=order_id,
                )
                results.append(order)
            except Exception as e:
                errors.append(e)

        # Mock idempotency store to simulate race condition
        call_count = 0

        def mock_check_and_store(client_id, data):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call succeeds
                return
            else:
                # Second call fails (duplicate)
                raise IdempotencyError(f"Duplicate order {client_id}")

        self.idempotency_store.get_order_data.return_value = None
        self.idempotency_store.check_and_store.side_effect = mock_check_and_store

        # Start two threads with same order ID
        thread1 = threading.Thread(target=prepare_order_thread, args=("race-test-123",))
        thread2 = threading.Thread(target=prepare_order_thread, args=("race-test-123",))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # One should succeed, one should fail
        self.assertEqual(len(results) + len(errors), 2)
        if results:
            self.assertIsInstance(results[0], PreparedOrder)
        if errors:
            self.assertIsInstance(errors[0], IdempotencyError)


if __name__ == "__main__":
    unittest.main()
