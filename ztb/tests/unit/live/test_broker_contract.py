"""
Contract tests for broker implementations.

Tests that all broker implementations conform to the IBroker interface
and behave correctly in dry-run mode.
"""

import pytest

from ztb.live.broker_interfaces import Balance, Order, Position
from ztb.live.broker_registry import get_broker_registry
from ztb.live.coincheck_adapter import CoincheckAdapter


class TestBrokerContract:
    """Test broker implementations against IBroker contract."""

    @pytest.fixture
    def coincheck_broker(self) -> CoincheckAdapter:
        """Create Coincheck broker in dry-run mode."""
        return CoincheckAdapter(dry_run=True)

    @pytest.fixture
    def broker_registry(self):
        """Get broker registry."""
        return get_broker_registry()

    def test_coincheck_adapter_creation(self, coincheck_broker: CoincheckAdapter):
        """Test CoincheckAdapter can be created."""
        assert coincheck_broker is not None
        assert coincheck_broker.dry_run is True

    def test_broker_registry_has_coincheck(self, broker_registry):
        """Test broker registry includes coincheck."""
        brokers = broker_registry.list_brokers()
        assert "coincheck" in brokers

    def test_broker_registry_can_create_coincheck(self, broker_registry):
        """Test broker registry can create coincheck instance."""
        broker = broker_registry.get_broker("coincheck", dry_run=True)
        assert isinstance(broker, CoincheckAdapter)
        assert broker.dry_run is True

    @pytest.mark.asyncio
    async def test_place_order_contract(self, coincheck_broker: CoincheckAdapter):
        """Test place_order conforms to contract."""
        order = await coincheck_broker.place_order(
            symbol="btc_jpy",
            side="buy",
            quantity=0.001,
            price=5000000.0,
            order_type="limit",
        )

        assert isinstance(order, Order)
        assert order.order_id is not None
        assert order.symbol == "btc_jpy"
        assert order.side == "buy"
        assert order.quantity == 0.001
        assert order.order_type == "limit"

    @pytest.mark.asyncio
    async def test_cancel_order_contract(self, coincheck_broker: CoincheckAdapter):
        """Test cancel_order conforms to contract."""
        # First place an order
        order = await coincheck_broker.place_order(
            symbol="btc_jpy", side="buy", quantity=0.001
        )

        # Try to cancel it
        cancelled = await coincheck_broker.cancel_order(order.order_id)
        assert isinstance(cancelled, bool)

    @pytest.mark.asyncio
    async def test_get_order_status_contract(self, coincheck_broker: CoincheckAdapter):
        """Test get_order_status conforms to contract."""
        # Place order
        order = await coincheck_broker.place_order(
            symbol="btc_jpy", side="buy", quantity=0.001
        )

        # Get status
        status_order = await coincheck_broker.get_order_status(order.order_id)
        assert status_order is not None
        assert isinstance(status_order, Order)
        assert status_order.order_id == order.order_id

    @pytest.mark.asyncio
    async def test_get_open_orders_contract(self, coincheck_broker: CoincheckAdapter):
        """Test get_open_orders conforms to contract."""
        # Place some orders
        await coincheck_broker.place_order("btc_jpy", "buy", 0.001)
        await coincheck_broker.place_order("btc_jpy", "sell", 0.001)

        # Get open orders
        open_orders = await coincheck_broker.get_open_orders()
        assert isinstance(open_orders, list)
        for order in open_orders:
            assert isinstance(order, Order)

        # Test filtering by symbol
        btc_orders = await coincheck_broker.get_open_orders("btc_jpy")
        assert isinstance(btc_orders, list)
        assert all(order.symbol == "btc_jpy" for order in btc_orders)

    @pytest.mark.asyncio
    async def test_get_positions_contract(self, coincheck_broker: CoincheckAdapter):
        """Test get_positions conforms to contract."""
        positions = await coincheck_broker.get_positions()
        assert isinstance(positions, list)
        for position in positions:
            assert isinstance(position, Position)

    @pytest.mark.asyncio
    async def test_get_balance_contract(self, coincheck_broker: CoincheckAdapter):
        """Test get_balance conforms to contract."""
        balances = await coincheck_broker.get_balance()
        assert isinstance(balances, list)
        assert len(balances) > 0
        for balance in balances:
            assert isinstance(balance, Balance)

        # Test filtering by currency
        jpy_balances = await coincheck_broker.get_balance("JPY")
        assert isinstance(jpy_balances, list)
        assert all(balance.currency == "JPY" for balance in jpy_balances)

    @pytest.mark.asyncio
    async def test_get_current_price_contract(self, coincheck_broker: CoincheckAdapter):
        """Test get_current_price conforms to contract."""
        price = await coincheck_broker.get_current_price("btc_jpy")
        assert isinstance(price, (float, type(None)))
        if price is not None:
            assert price > 0

    @pytest.mark.asyncio
    async def test_dry_run_simulation(self, coincheck_broker: CoincheckAdapter):
        """Test that dry-run simulates realistic behavior."""
        # Check initial balance
        balances = await coincheck_broker.get_balance("JPY")
        initial_jpy = balances[0].free if balances else 0

        # Place buy order
        order = await coincheck_broker.place_order(
            symbol="btc_jpy",
            side="buy",
            quantity=0.001,
            price=5000000.0,
            order_type="market",
        )

        # Check balance changed (simulated)
        balances_after = await coincheck_broker.get_balance("JPY")
        final_jpy = balances_after[0].free if balances_after else 0

        # Balance should decrease (cost of order)
        assert final_jpy <= initial_jpy

        # Check positions
        positions = await coincheck_broker.get_positions()
        btc_positions = [p for p in positions if p.symbol == "btc_jpy"]
        if order.status == "filled":
            assert len(btc_positions) > 0

    @pytest.mark.asyncio
    async def test_real_trading_not_implemented(
        self, coincheck_broker: CoincheckAdapter
    ):
        """Test that real trading raises NotImplementedError."""
        coincheck_broker.dry_run = False

        with pytest.raises(NotImplementedError):
            await coincheck_broker.place_order("btc_jpy", "buy", 0.001)

        with pytest.raises(NotImplementedError):
            await coincheck_broker.get_balance()

        # Reset for other tests
        coincheck_broker.dry_run = True
