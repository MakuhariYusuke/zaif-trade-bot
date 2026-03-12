"""Unit tests for live trading module."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ztb.trading.backtest.adapters import create_adapter
from ztb.trading.live.broker_interfaces import Order
from ztb.trading.live.paper_trader import PaperTrader
from ztb.trading.live.sim_broker import SimBroker

_SYMBOL_SPECS = [
    {
        "symbol": "BTC_JPY",
        "base_asset": "BTC",
        "quote_asset": "JPY",
        "min_order_size": 0.0001,
        "max_order_size": 10.0,
        "price_precision": 0,
        "quantity_precision": 8,
        "min_price": 1.0,
        "max_price": 100000000.0,
    },
    {
        "symbol": "btc_jpy",
        "base_asset": "BTC",
        "quote_asset": "JPY",
        "min_order_size": 0.0001,
        "max_order_size": 10.0,
        "price_precision": 0,
        "quantity_precision": 8,
        "min_price": 1.0,
        "max_price": 100000000.0,
    },
]
VENUE_CONFIG = {"symbols": _SYMBOL_SPECS}


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


async def _fast_sleep(_: float) -> None:
    return None


def _mock_replay_data() -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01", periods=8, freq="1h")
    return pd.DataFrame(
        {
            "open": [5000000.0 + i * 10000 for i in range(8)],
            "high": [5010000.0 + i * 10000 for i in range(8)],
            "low": [4990000.0 + i * 10000 for i in range(8)],
            "close": [5005000.0 + i * 10000 for i in range(8)],
            "volume": [1000.0] * 8,
        },
        index=timestamps,
    )


class TestSimBroker:
    """Test SimBroker functionality."""

    def setup_method(self) -> None:
        self.broker = SimBroker(
            initial_balance=100000.0,
            commission_bps=10.0,
            slippage_bps=5.0,
            venue_config=VENUE_CONFIG,
        )

    def test_initialization(self) -> None:
        assert self.broker.balance["JPY"] == 100000.0
        assert self.broker.commission_bps == 10.0
        assert self.broker.slippage_bps == 5.0
        assert len(self.broker.positions) == 0

    def test_place_market_order_buy(self) -> None:
        order = _run(
            self.broker.place_order(
                symbol="BTC_JPY", side="buy", quantity=0.001, order_type="market"
            )
        )

        assert order is not None
        assert order.symbol == "BTC_JPY"
        assert order.side == "buy"
        assert order.quantity == 0.001
        assert order.price is not None
        assert order.status == "filled"
        assert self.broker.balance["JPY"] < 100000.0
        assert self.broker.balance["BTC"] == 0.001

    def test_place_market_order_sell(self) -> None:
        _run(
            self.broker.place_order(
                symbol="BTC_JPY", side="buy", quantity=0.001, order_type="market"
            )
        )
        order = _run(
            self.broker.place_order(
                symbol="BTC_JPY", side="sell", quantity=0.001, order_type="market"
            )
        )

        assert order.side == "sell"
        assert order.quantity == 0.001
        assert order.price is not None
        assert order.status == "filled"

    def test_insufficient_balance(self) -> None:
        order = _run(
            self.broker.place_order(
                symbol="BTC_JPY", side="buy", quantity=10.0, order_type="market"
            )
        )

        assert order.status == "rejected"

    def test_get_balance(self) -> None:
        balances = _run(self.broker.get_balance())
        assert len(balances) > 0
        jpy_balance = next((b for b in balances if b.currency == "JPY"), None)
        assert jpy_balance is not None
        assert jpy_balance.total == 100000.0

    def test_get_positions(self) -> None:
        positions = _run(self.broker.get_positions())
        assert len(positions) == 0

        _run(
            self.broker.place_order(
                symbol="BTC_JPY", side="buy", quantity=0.001, order_type="market"
            )
        )

        positions = _run(self.broker.get_positions())
        assert len(positions) == 1
        assert positions[0].symbol == "BTC_JPY"
        assert positions[0].quantity == 0.001


class TestPaperTrader:
    """Test PaperTrader functionality."""

    def setup_method(self) -> None:
        self.strategy = create_adapter("sma_fast_slow")

    def _make_broker(self) -> SimBroker:
        return SimBroker(initial_balance=100000.0, venue_config=VENUE_CONFIG)

    def _make_trader(self, mode: str = "replay", dataset: str | None = "test_data") -> PaperTrader:
        return PaperTrader(
            broker=self._make_broker(),
            strategy=self.strategy,
            mode=mode,
            dataset=dataset,
            venue_config=VENUE_CONFIG,
        )

    def test_replay_mode(self, tmp_path: Path) -> None:
        mock_data = _mock_replay_data()
        with (
            patch(
                "ztb.trading.live.simulation.paper_trader.PaperTrader._load_data_feed",
                return_value=mock_data,
            ) as mock_load_data,
            patch(
                "ztb.trading.live.simulation.paper_trader.asyncio.sleep",
                side_effect=_fast_sleep,
            ),
        ):
            trader = self._make_trader(mode="replay", dataset="test_data")
            result = _run(trader.run_replay(output_dir=tmp_path))

        assert result is not None
        assert "trade_log" in result
        assert "trades_executed" in result
        assert "pnl_series" in result
        mock_load_data.assert_called_once_with("test_data")

    def test_live_lite_mode(self) -> None:
        trader = self._make_trader(mode="live-lite", dataset=None)
        assert trader.broker is not None
        assert hasattr(trader, "run_replay")

    def test_live_lite_execution(self) -> None:
        trader = self._make_trader(mode="replay", dataset="test_data")
        assert trader.broker is not None
        assert trader.strategy is not None
        assert trader.mode == "replay"


class TestBrokerInterfaces:
    """Test broker interface definitions."""

    def test_order_creation(self) -> None:
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
