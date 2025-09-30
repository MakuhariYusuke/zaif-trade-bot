"""Integration tests for paper trader with streaming data source."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ztb.trading.backtest.adapters import create_adapter
from ztb.trading.live.paper_trader import PaperTrader
from ztb.trading.live.sim_broker import SimBroker


class MockStreamingPipeline:
    """Mock streaming pipeline for testing."""

    def __init__(self, trades_data=None):
        self.trades_data = trades_data or []
        self.current_index = 0

    async def get_next_batch(self, batch_size=10):
        """Mock get_next_batch method."""
        if self.current_index >= len(self.trades_data):
            return []

        batch = self.trades_data[self.current_index : self.current_index + batch_size]
        self.current_index += len(batch)
        return batch

    async def get_current_price(self, symbol="BTC_JPY"):
        """Mock current price."""
        return 5000000.0  # 5M JPY


@pytest.mark.asyncio
class TestPaperFromStreaming:
    """Test paper trader with streaming data source."""

    async def test_paper_trader_from_streaming_executes_trades(self):
        """Test that paper trader executes trades when using streaming data."""
        # Create mock streaming pipeline with sample trades
        trades_data = [
            {
                "timestamp": "2023-01-01T10:00:00Z",
                "price": 5000000.0,
                "quantity": 0.001,
                "side": "buy",
            },
            {
                "timestamp": "2023-01-01T10:01:00Z",
                "price": 5010000.0,
                "quantity": 0.001,
                "side": "sell",
            },
            {
                "timestamp": "2023-01-01T10:02:00Z",
                "price": 4990000.0,
                "quantity": 0.001,
                "side": "buy",
            },
        ]

        mock_pipeline = MockStreamingPipeline(trades_data)

        # Create paper trader with from_streaming=True
        broker = SimBroker(initial_balance=10000.0)
        strategy = create_adapter("sma_fast_slow")

        venue_config = {
            "symbols": [
                {
                    "symbol": "btc_jpy",
                    "base_asset": "BTC",
                    "quote_asset": "JPY",
                    "min_order_size": 0.0001,
                    "max_order_size": 1.0,
                    "price_precision": 0,
                    "quantity_precision": 4,
                    "min_price": 1000,
                    "max_price": 10000000,
                }
            ]
        }

        trader = PaperTrader(
            broker=broker,
            strategy=strategy,
            mode="live-lite",
            duration_minutes=5,
            venue_config=venue_config,
            from_streaming=True,
        )

        # Mock the streaming pipeline connection
        with patch(
            "ztb.live.paper_trader.StreamingPipeline", return_value=mock_pipeline
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)

                # Run the trader
                results = await trader.run(output_dir)

                # Verify trades were executed
                assert results["trades_executed"] > 0, "No trades were executed"
                assert len(results["trade_log"]) > 0, "Trade log is empty"

                # Verify P&L series exists
                assert "pnl_series" in results

                # Verify output files were created
                assert (output_dir / "trade_log.json").exists()
                assert (output_dir / "orders.csv").exists()
                assert (output_dir / "stats.json").exists()

    async def test_paper_trader_from_streaming_validation(self):
        """Test that orders are validated against venue constraints."""
        mock_pipeline = MockStreamingPipeline()

        broker = SimBroker(initial_balance=10000.0)
        strategy = create_adapter("sma_fast_slow")

        venue_config = {
            "symbols": [
                {
                    "symbol": "btc_jpy",
                    "base_asset": "BTC",
                    "quote_asset": "JPY",
                    "min_order_size": 0.01,  # Large minimum
                    "max_order_size": 1.0,
                    "price_precision": 0,
                    "quantity_precision": 4,
                    "min_price": 1000,
                    "max_price": 10000000,
                }
            ]
        }

        trader = PaperTrader(
            broker=broker,
            strategy=strategy,
            mode="live-lite",
            duration_minutes=5,
            venue_config=venue_config,
            from_streaming=True,
        )

        # Try to place order below minimum size
        with pytest.raises(ValueError, match="below minimum"):
            trader.validate_order("btc_jpy", "buy", 0.001)  # Below 0.01 minimum

    async def test_paper_trader_fallback_to_cached_data(self):
        """Test that trader falls back to cached data when from_streaming=False."""
        broker = SimBroker(initial_balance=10000.0)
        strategy = create_adapter("sma_fast_slow")

        venue_config = {
            "symbols": [
                {
                    "symbol": "btc_jpy",
                    "base_asset": "BTC",
                    "quote_asset": "JPY",
                    "min_order_size": 0.0001,
                    "max_order_size": 1.0,
                    "price_precision": 0,
                    "quantity_precision": 4,
                    "min_price": 1000,
                    "max_price": 10000000,
                }
            ]
        }

        trader = PaperTrader(
            broker=broker,
            strategy=strategy,
            mode="replay",
            dataset="btc_jpy_1m",
            duration_minutes=5,
            venue_config=venue_config,
            from_streaming=False,  # Explicitly false
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Should work with cached data (replay mode)
            results = await trader.run(output_dir)
            assert "trades_executed" in results
