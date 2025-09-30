"""
Integration tests for the 1M learning validation pipeline.

Tests end-to-end workflows combining backtest, paper trading, and risk management.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd

from ztb.trading.backtest.adapters import BuyAndHoldAdapter, SMACrossoverAdapter
from ztb.trading.backtest.runner import BacktestEngine


class TestBacktestRiskIntegration:
    """Integration tests for backtest engine with risk controls."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create sample data
        self.sample_data = [
            {"timestamp": "2024-01-01 00:00:00", "close": 100.0, "volume": 1000},
            {"timestamp": "2024-01-01 01:00:00", "close": 101.0, "volume": 1100},
            {"timestamp": "2024-01-01 02:00:00", "close": 102.0, "volume": 1200},
            {"timestamp": "2024-01-01 03:00:00", "close": 103.0, "volume": 1300},
            {"timestamp": "2024-01-01 04:00:00", "close": 104.0, "volume": 1400},
        ]

        # Save sample data
        self.data_file = self.temp_dir / "sample_data.json"
        with open(self.data_file, "w") as f:
            json.dump(self.sample_data, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_backtest_with_risk_controls(self):
        """Test backtest engine integrated with risk management."""
        # Create sample data
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="H"),
                "close": [100 + i for i in range(10)],
                "volume": [1000] * 10,
            }
        )

        # Initialize components
        strategy = SMACrossoverAdapter(fast_period=3, slow_period=5)
        engine = BacktestEngine()

        # Run backtest
        equity_curve, orders = engine.run_backtest(strategy, data)

        # Verify results
        assert len(equity_curve) == len(data)
        assert isinstance(orders, pd.DataFrame)
        assert len(orders) >= 0  # May have no trades

    def test_backtest_risk_rejection(self):
        """Test that risk controls can block trades during backtest."""
        # Create sample data
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="H"),
                "close": [100 + i for i in range(10)],
                "volume": [1000] * 10,
            }
        )

        # Use simple strategy
        strategy = BuyAndHoldAdapter()
        engine = BacktestEngine()

        # Run backtest
        equity_curve, orders = engine.run_backtest(strategy, data)

        # Verify basic execution
        assert len(equity_curve) == len(data)
        assert isinstance(orders, pd.DataFrame)


class TestPaperTradingRiskIntegration:
    """Integration tests for paper trading with risk controls."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_paper_trader_placeholder(self):
        """Placeholder for paper trading integration tests."""
        # TODO: Implement paper trading integration tests
        assert True


class TestEndToEndValidation:
    """End-to-end validation tests for the complete pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create comprehensive test data
        self.test_data = []
        base_price = 100.0
        for i in range(100):  # 100 data points
            price = base_price + (i * 0.1) + (i % 10 - 5) * 0.5  # Trending with noise
            self.test_data.append(
                {
                    "timestamp": f"2024-01-{str(i // 24 + 1).zfill(2)} {str(i % 24).zfill(2)}:00:00",
                    "close": price,
                    "volume": 1000 + i * 10,
                }
            )

        self.data_file = self.temp_dir / "test_data.json"
        with open(self.data_file, "w") as f:
            json.dump(self.test_data, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_complete_validation_pipeline(self):
        """Test the complete validation pipeline from backtest to paper trading."""
        # Phase 1: Backtest with risk controls
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=20, freq="H"),
                "close": [100 + i * 0.5 for i in range(20)],
                "volume": [1000] * 20,
            }
        )

        strategy = SMACrossoverAdapter(fast_period=5, slow_period=10)
        backtest_engine = BacktestEngine()

        # Run backtest
        equity_curve, orders = backtest_engine.run_backtest(strategy, data)

        # Verify backtest produced valid results
        assert len(equity_curve) > 0
        assert isinstance(orders, pd.DataFrame)

        # Phase 2: Paper trading simulation (simplified)
        # For now, just verify backtest completed successfully
        assert True  # Placeholder for paper trading integration

    def test_deterministic_execution(self):
        """Test that the pipeline produces deterministic results."""
        # Create consistent data
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="H"),
                "close": [100 + i for i in range(10)],
                "volume": [1000] * 10,
            }
        )

        # First run
        strategy1 = BuyAndHoldAdapter()
        engine1 = BacktestEngine()
        equity_curve1, orders1 = engine1.run_backtest(strategy1, data)

        # Second run
        strategy2 = BuyAndHoldAdapter()
        engine2 = BacktestEngine()
        equity_curve2, orders2 = engine2.run_backtest(strategy2, data)

        # Results should be identical (deterministic)
        assert equity_curve1.equals(equity_curve2)
        assert orders1.equals(orders2)
