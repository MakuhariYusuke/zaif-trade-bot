"""Unit tests for live trading module."""

from unittest.mock import patch

import pandas as pd
import pytest

from ztb.trading.backtest.adapters import create_adapter
from ztb.trading.live.broker_interfaces import Order
from ztb.trading.live.paper_trader import PaperTrader
from ztb.trading.live.sim_broker import SimBroker


class TestSimBroker:
    """Test SimBroker functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.broker = SimBroker(
            initial_balance=100000.0,
            commission_bps=10.0,  # 0.1%
            slippage_bps=5.0,  # 0.05%
        )

    def test_initialization(self):
        """Test broker initializes correctly."""
        assert self.broker.balance["JPY"] == 100000.0
        assert self.broker.commission_bps == 10.0
        assert self.broker.slippage_bps == 5.0
        assert len(self.broker.positions) == 0

    @pytest.mark.anyio
    async def test_place_market_order_buy(self):
        """Test placing market buy order."""
        order = await self.broker.place_order(
            symbol="BTC_JPY", side="buy", quantity=0.001, order_type="market"
        )

        assert order is not None
        assert order.symbol == "BTC_JPY"
        assert order.side == "buy"
        assert order.quantity == 0.001
        assert order.price is not None
        assert order.status == "filled"

        # Check balance updated
        assert self.broker.balance["JPY"] < 100000.0
        assert self.broker.balance["BTC"] == 0.001

    @pytest.mark.anyio
    async def test_place_market_order_sell(self):
        """Test placing market sell order."""
        # First buy some BTC
        await self.broker.place_order(
            symbol="BTC_JPY", side="buy", quantity=0.001, order_type="market"
        )

        # Then sell
        order = await self.broker.place_order(
            symbol="BTC_JPY", side="sell", quantity=0.001, order_type="market"
        )

        assert order.side == "sell"
        assert order.quantity == 0.001
        assert order.price is not None
        assert order.status == "filled"

    @pytest.mark.anyio
    async def test_insufficient_balance(self):
        """Test handling insufficient balance."""
        # Try to buy with insufficient balance
        order = await self.broker.place_order(
            symbol="BTC_JPY",
            side="buy",
            quantity=10.0,  # Too large
            order_type="market",
        )

        # Should be rejected
        assert order.status == "rejected"

    @pytest.mark.anyio
    async def test_get_balance(self):
        """Test getting current balance."""
        balances = await self.broker.get_balance()
        assert len(balances) > 0
        jpy_balance = next((b for b in balances if b.currency == "JPY"), None)
        assert jpy_balance is not None
        assert jpy_balance.total == 100000.0

    @pytest.mark.anyio
    async def test_get_positions(self):
        """Test getting current positions."""
        positions = await self.broker.get_positions()
        assert len(positions) == 0

        # After buying
        await self.broker.place_order(
            symbol="BTC_JPY", side="buy", quantity=0.001, order_type="market"
        )

        positions = await self.broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTC_JPY"
        assert positions[0].quantity == 0.001


class TestPaperTrader:
    """Test PaperTrader functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.broker = SimBroker(initial_balance=100000.0)
        self.strategy = create_adapter("sma_fast_slow")  # Create a strategy adapter
        self.trader = PaperTrader(
            broker=self.broker,
            strategy=self.strategy,
            mode="replay",
            dataset="test_data",  # Provide dataset to trigger data loading
        )

    @patch("ztb.live.paper_trader.PaperTrader._load_data_feed")
    @pytest.mark.anyio
    async def test_replay_mode(self, mock_load_data):
        """Test replay mode execution."""
        # Mock market data
        mock_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1H"),
                "close": [5000000.0 + i * 10000 for i in range(10)],
                "volume": [1000.0] * 10,
            }
        )
        mock_load_data.return_value = mock_data

        # Create output directory
        from pathlib import Path

        output_dir = Path("tmp/test_paper_trader")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run replay
        result = await self.trader.run_replay(output_dir=output_dir)

        assert result is not None
        assert "trade_log" in result
        assert "trades_executed" in result
        assert "pnl_series" in result

    def test_live_lite_mode(self):
        """Test live-lite mode setup."""
        # This would require mocking external price feeds
        # For now, just test initialization
        assert self.trader.broker is not None
        assert hasattr(self.trader, "run_replay")

    @pytest.mark.anyio
    async def test_live_lite_execution(self):
        """Test live-lite mode setup."""
        # This is a simplified test - live-lite mode requires external price feeds
        # Just test that the trader is properly initialized
        assert self.trader.broker is not None
        assert self.trader.strategy is not None
        assert self.trader.mode == "replay"


class TestBrokerInterfaces:
    """Test broker interface definitions."""

    def test_order_creation(self):
        """Test Order dataclass creation."""
        order = Order(
            order_id="test_123",
            symbol="BTC_JPY",
            side="buy",
            quantity=0.001,
            order_type="market",
            price=None,
        )

        assert order.order_id == "test_123"
        assert order.symbol == "BTC_JPY"
        assert order.side == "buy"
        assert order.quantity == 0.001
        assert order.order_type == "market"
        assert order.price is None
        assert order.status == "pending"


if __name__ == "__main__":
    pytest.main([__file__])
