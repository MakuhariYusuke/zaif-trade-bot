"""
Unit tests for reconciliation framework
調整フレームワークの単体テスト
"""

import time
import pytest
from unittest.mock import patch
from ztb.trading.core.reconciliation import (
    ReconciliationItem, ReconciliationResult, ReconciliationStrategy,
    OrderReconciliationStrategy, PositionReconciliationStrategy,
    ReconciliationEngine, get_reconciliation_engine
)


class TestReconciliationItem:
    def test_reconciliation_item_creation(self):
        """Test ReconciliationItem dataclass creation"""
        item = ReconciliationItem(
            item_id="item_123",
            internal_state={"status": "filled", "quantity": 1.0},
            external_state={"status": "filled", "quantity": 1.0},
            timestamp=1234567890.0
        )

        assert item.item_id == "item_123"
        assert item.internal_state["status"] == "filled"
        assert item.external_state["status"] == "filled"
        assert item.timestamp == 1234567890.0


class TestReconciliationResult:
    def test_reconciliation_result_creation(self):
        """Test ReconciliationResult dataclass creation"""
        result = ReconciliationResult(
            item_id="item_123",
            is_consistent=True,
            discrepancies=[],
            actions_taken=["no_action_needed"],
            timestamp=1234567890.0
        )

        assert result.item_id == "item_123"
        assert result.is_consistent is True
        assert len(result.discrepancies) == 0
        assert result.actions_taken == ["no_action_needed"]
        assert result.timestamp == 1234567890.0


class TestReconciliationStrategy:
    def test_reconciliation_strategy_is_abstract(self):
        """Test that ReconciliationStrategy cannot be instantiated directly"""
        with pytest.raises(TypeError):
            ReconciliationStrategy()


class TestOrderReconciliationStrategy:
    def test_get_strategy_name(self):
        """Test OrderReconciliationStrategy strategy name"""
        strategy = OrderReconciliationStrategy()
        assert strategy.get_strategy_name() == "order_reconciliation"

    def test_reconcile_consistent_order(self):
        """Test reconciling a consistent order"""
        item = ReconciliationItem(
            item_id="order_123",
            internal_state={
                "status": "filled",
                "filled_quantity": 1.0,
                "remaining_quantity": 0.0
            },
            external_state={
                "status": "filled",
                "filled_quantity": 1.0,
                "remaining_quantity": 0.0
            },
            timestamp=time.time()
        )

        strategy = OrderReconciliationStrategy()
        result = strategy.reconcile_item(item)

        assert result.item_id == "order_123"
        assert result.is_consistent is True
        assert len(result.discrepancies) == 0
        assert result.actions_taken == ["no_action_needed"]

    def test_reconcile_inconsistent_order_status(self):
        """Test reconciling an order with status mismatch"""
        item = ReconciliationItem(
            item_id="order_123",
            internal_state={
                "status": "filled",
                "filled_quantity": 1.0,
                "remaining_quantity": 0.0
            },
            external_state={
                "status": "pending",
                "filled_quantity": 1.0,
                "remaining_quantity": 0.0
            },
            timestamp=time.time()
        )

        strategy = OrderReconciliationStrategy()
        result = strategy.reconcile_item(item)

        assert result.item_id == "order_123"
        assert result.is_consistent is False
        assert len(result.discrepancies) == 1
        assert "Status mismatch" in result.discrepancies[0]
        assert result.actions_taken == ["logged_discrepancies"]

    def test_reconcile_inconsistent_order_quantity(self):
        """Test reconciling an order with quantity mismatch"""
        item = ReconciliationItem(
            item_id="order_123",
            internal_state={
                "status": "filled",
                "filled_quantity": 1.0,
                "remaining_quantity": 0.0
            },
            external_state={
                "status": "filled",
                "filled_quantity": 0.5,
                "remaining_quantity": 0.5
            },
            timestamp=time.time()
        )

        strategy = OrderReconciliationStrategy()
        result = strategy.reconcile_item(item)

        assert result.item_id == "order_123"
        assert result.is_consistent is False
        assert len(result.discrepancies) == 2  # filled and remaining quantity mismatch
        assert any("Filled quantity mismatch" in d for d in result.discrepancies)
        assert any("Remaining quantity mismatch" in d for d in result.discrepancies)


class TestPositionReconciliationStrategy:
    def test_get_strategy_name(self):
        """Test PositionReconciliationStrategy strategy name"""
        strategy = PositionReconciliationStrategy()
        assert strategy.get_strategy_name() == "position_reconciliation"

    def test_reconcile_consistent_position(self):
        """Test reconciling a consistent position"""
        item = ReconciliationItem(
            item_id="position_BTC",
            internal_state={
                "size": 1.0,
                "average_price": 50000.0
            },
            external_state={
                "size": 1.0,
                "average_price": 50000.0
            },
            timestamp=time.time()
        )

        strategy = PositionReconciliationStrategy()
        result = strategy.reconcile_item(item)

        assert result.item_id == "position_BTC"
        assert result.is_consistent is True
        assert len(result.discrepancies) == 0
        assert result.actions_taken == ["no_action_needed"]

    def test_reconcile_inconsistent_position_size(self):
        """Test reconciling a position with size mismatch"""
        item = ReconciliationItem(
            item_id="position_BTC",
            internal_state={
                "size": 1.0,
                "average_price": 50000.0
            },
            external_state={
                "size": 1.5,
                "average_price": 50000.0
            },
            timestamp=time.time()
        )

        strategy = PositionReconciliationStrategy()
        result = strategy.reconcile_item(item)

        assert result.item_id == "position_BTC"
        assert result.is_consistent is False
        assert len(result.discrepancies) == 1
        assert "Position size mismatch" in result.discrepancies[0]

    def test_reconcile_inconsistent_position_price(self):
        """Test reconciling a position with price mismatch"""
        item = ReconciliationItem(
            item_id="position_BTC",
            internal_state={
                "size": 1.0,
                "average_price": 50000.0
            },
            external_state={
                "size": 1.0,
                "average_price": 51000.0  # 2% difference
            },
            timestamp=time.time()
        )

        strategy = PositionReconciliationStrategy()
        result = strategy.reconcile_item(item)

        assert result.item_id == "position_BTC"
        assert result.is_consistent is False
        assert len(result.discrepancies) == 1
        assert "Average price mismatch" in result.discrepancies[0]


class TestReconciliationEngine:
    def test_initialization_registers_default_strategies(self):
        """Test ReconciliationEngine initialization registers default strategies"""
        engine = ReconciliationEngine()

        assert "order_reconciliation" in engine.strategies
        assert "position_reconciliation" in engine.strategies
        assert isinstance(engine.strategies["order_reconciliation"], OrderReconciliationStrategy)
        assert isinstance(engine.strategies["position_reconciliation"], PositionReconciliationStrategy)

    def test_register_strategy(self):
        """Test registering a custom strategy"""
        engine = ReconciliationEngine()

        # Create a custom strategy
        class CustomStrategy(ReconciliationStrategy):
            def get_strategy_name(self):
                return "custom_strategy"

            def reconcile_item(self, item):
                return ReconciliationResult(
                    item_id=item.item_id,
                    is_consistent=True,
                    discrepancies=[],
                    actions_taken=["custom_action"],
                    timestamp=item.timestamp
                )

        custom_strategy = CustomStrategy()
        engine.register_strategy(custom_strategy)

        assert "custom_strategy" in engine.strategies
        assert engine.strategies["custom_strategy"] == custom_strategy

    def test_reconcile_items_with_valid_strategy(self):
        """Test reconciling items with valid strategy"""
        engine = ReconciliationEngine()

        items = [
            ReconciliationItem(
                item_id="order_1",
                internal_state={"status": "filled", "filled_quantity": 1.0, "remaining_quantity": 0.0},
                external_state={"status": "filled", "filled_quantity": 1.0, "remaining_quantity": 0.0},
                timestamp=time.time()
            ),
            ReconciliationItem(
                item_id="order_2",
                internal_state={"status": "pending", "filled_quantity": 0.0, "remaining_quantity": 1.0},
                external_state={"status": "filled", "filled_quantity": 1.0, "remaining_quantity": 0.0},
                timestamp=time.time()
            )
        ]

        results = engine.reconcile_items(items, "order_reconciliation")

        assert len(results) == 2
        assert results[0].is_consistent is True
        assert results[1].is_consistent is False

    def test_reconcile_items_with_invalid_strategy(self):
        """Test reconciling items with invalid strategy name"""
        engine = ReconciliationEngine()

        items = [
            ReconciliationItem(
                item_id="order_1",
                internal_state={"status": "filled"},
                external_state={"status": "filled"},
                timestamp=time.time()
            )
        ]

        with pytest.raises(ValueError, match="Unknown reconciliation strategy"):
            engine.reconcile_items(items, "invalid_strategy")

    def test_get_reconciliation_summary(self):
        """Test generating reconciliation summary"""
        engine = ReconciliationEngine()

        results = [
            ReconciliationResult(
                item_id="item_1",
                is_consistent=True,
                discrepancies=[],
                actions_taken=["no_action_needed"],
                timestamp=time.time()
            ),
            ReconciliationResult(
                item_id="item_2",
                is_consistent=False,
                discrepancies=["mismatch"],
                actions_taken=["logged_discrepancies"],
                timestamp=time.time()
            ),
            ReconciliationResult(
                item_id="item_3",
                is_consistent=True,
                discrepancies=[],
                actions_taken=["no_action_needed"],
                timestamp=time.time()
            )
        ]

        summary = engine.get_reconciliation_summary(results)

        assert summary["total_items"] == 3
        assert summary["consistent_items"] == 2
        assert summary["inconsistent_items"] == 1
        assert summary["consistency_rate"] == 2/3
        assert summary["total_discrepancies"] == 1
        assert len(summary["results"]) == 3


class TestGlobalReconciliationEngine:
    def test_get_reconciliation_engine(self):
        """Test global reconciliation engine getter"""
        engine1 = get_reconciliation_engine()
        engine2 = get_reconciliation_engine()

        # Should return the same instance
        assert engine1 is engine2
        assert isinstance(engine1, ReconciliationEngine)