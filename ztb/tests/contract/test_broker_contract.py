"""
Contract Tests for Broker Implementations

Tests broker implementations against a common interface contract.
Ensures sim and skeleton brokers behave consistently.
"""

import pytest
from typing import Dict, Any

from ztb.live.broker_registry import get_broker_registry, BrokerProtocol, CoincheckSkeletonBroker


class TestBrokerContract:
    """Contract tests for broker implementations."""

    @pytest.fixture(params=["sim", "coincheck_skeleton"])
    def broker(self, request) -> BrokerProtocol:
        """Parametrized fixture for broker instances."""
        registry = get_broker_registry()
        broker_name = request.param

        if broker_name == "coincheck_skeleton":
            # Skeleton should work without credentials
            return registry.get_broker(broker_name)
        else:
            return registry.get_broker(broker_name)

    @pytest.fixture
    def sim_broker(self) -> BrokerProtocol:
        """Sim broker instance."""
        registry = get_broker_registry()
        return registry.get_broker("sim")

    def test_get_balance_contract(self, broker):
        """Test get_balance method contract."""
        if isinstance(broker, CoincheckSkeletonBroker):
            # Skeleton should raise NotImplementedError
            with pytest.raises(NotImplementedError):
                broker.get_balance("JPY")
        else:
            # Should return a float
            balance = broker.get_balance("JPY")
            assert isinstance(balance, float)
            assert balance >= 0

    def test_place_order_contract(self, broker):
        """Test place_order method contract."""
        if isinstance(broker, CoincheckSkeletonBroker):
            # Skeleton should raise NotImplementedError
            with pytest.raises(NotImplementedError):
                broker.place_order("btc_jpy", "buy", 0.01, 30000.0)
        else:
            # Sim broker should work
            order_id = broker.place_order("btc_jpy", "buy", 0.01, 30000.0)
            assert isinstance(order_id, str)
            assert len(order_id) > 0

    def test_cancel_order_contract(self, sim_broker):
        """Test cancel_order method contract."""
        # Place an order first
        order_id = sim_broker.place_order("btc_jpy", "buy", 0.01, 30000.0)

        # Cancel should work
        result = sim_broker.cancel_order(order_id)
        assert isinstance(result, bool)

    def test_get_order_status_contract(self, sim_broker):
        """Test get_order_status method contract."""
        # Place an order first
        order_id = sim_broker.place_order("btc_jpy", "buy", 0.01, 30000.0)

        # Get status
        status = sim_broker.get_order_status(order_id)
        assert isinstance(status, dict)
        assert "status" in status

    def test_get_open_orders_contract(self, broker):
        """Test get_open_orders method contract."""
        if isinstance(broker, CoincheckSkeletonBroker):
            # Skeleton should raise NotImplementedError
            with pytest.raises(NotImplementedError):
                broker.get_open_orders()
        else:
            # Should return a list
            orders = broker.get_open_orders()
            assert isinstance(orders, list)
            for order in orders:
                assert isinstance(order, dict)

    def test_sim_broker_state_consistency(self, sim_broker):
        """Test that sim broker maintains consistent state."""
        initial_jpy = sim_broker.get_balance("JPY")
        initial_btc = sim_broker.get_balance("BTC")

        # Place a buy order
        order_id = sim_broker.place_order("btc_jpy", "buy", 0.01, 30000.0)

        # Check order was recorded
        status = sim_broker.get_order_status(order_id)
        assert status["status"] == "filled"
        assert status["symbol"] == "btc_jpy"
        assert status["side"] == "buy"
        assert status["quantity"] == 0.01

        # Balances should be updated (simplified - in real broker this would be more complex)
        # For sim broker, we don't update balances, but the contract is that balances are accessible

    def test_coincheck_skeleton_graceful_failure(self):
        """Test that coincheck skeleton fails gracefully without credentials."""
        registry = get_broker_registry()
        broker = registry.get_broker("coincheck_skeleton")

        # All methods should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            broker.get_balance("JPY")

        with pytest.raises(NotImplementedError):
            broker.place_order("btc_jpy", "buy", 0.01)

        with pytest.raises(NotImplementedError):
            broker.cancel_order("test_id")

        with pytest.raises(NotImplementedError):
            broker.get_order_status("test_id")

        with pytest.raises(NotImplementedError):
            broker.get_open_orders()

    def test_broker_registry_list(self):
        """Test broker registry lists available brokers."""
        registry = get_broker_registry()
        brokers = registry.list_brokers()

        assert isinstance(brokers, list)
        assert "sim" in brokers
        assert "coincheck_skeleton" in brokers

    def test_broker_registry_get_unknown(self):
        """Test broker registry raises error for unknown broker."""
        registry = get_broker_registry()

        with pytest.raises(ValueError, match="Unknown broker"):
            registry.get_broker("unknown_broker")