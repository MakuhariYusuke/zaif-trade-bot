"""Unit tests for backtest module."""

import pytest

pytest.skip("ztb.trading.risk module not implemented", allow_module_level=True)

from unittest.mock import Mock, patch

import pandas as pd

from ztb.trading.backtest.adapters import (
    BuyAndHoldAdapter,
    RLPolicyAdapter,
    SMACrossoverAdapter,
)
from ztb.trading.backtest.metrics import MetricsCalculator
from ztb.trading.backtest.runner import BacktestEngine


class TestBacktestEngine:
    """Test backtest engine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = BacktestEngine()

    def test_initialization(self):
        """Test engine initializes correctly."""
        assert self.engine is not None
        assert hasattr(self.engine, "run_backtest")

    @patch("ztb.backtest.runner.BacktestEngine.load_data")
    def test_run_backtest_basic(self, mock_load_data: Mock) -> None:
        """Test basic backtest execution."""
        # Mock data
        mock_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="h"),
                "close": 100.0,
                "volume": 1000.0,
            }
        )
        mock_load_data.return_value = mock_data

        # Mock strategy
        strategy = Mock()
        strategy.generate_signal.return_value = {"action": "hold", "confidence": 0.5}

        # Run backtest
        result = self.engine.run_backtest(strategy, mock_data)

        assert result is not None
        assert len(result) == 2  # Should return tuple of (equity_curve, orders)


class TestMetricsCalculator:
    """Test metrics calculation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.calculator = MetricsCalculator()

    def test_calculate_returns(self) -> None:
        """Test return calculations."""
        equity_curve = pd.Series([100, 101, 99, 102, 98])
        returns = self.calculator.calculate_returns(equity_curve)

        assert len(returns) == 5  # Returns has same length as equity curve
        assert returns.iloc[0] == 0.0  # First return is always 0
        assert returns.iloc[1] == pytest.approx(0.01, rel=1e-2)
        assert returns.iloc[2] == pytest.approx(-0.0198, rel=1e-2)

    def test_calculate_sharpe_ratio(self) -> None:
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, -0.005, 0.015, -0.002, 0.008])
        sharpe = self.calculator.calculate_sharpe_ratio(returns)

        assert isinstance(sharpe, float)
        assert sharpe > 0  # Should be positive for this return series

    def test_calculate_max_drawdown(self) -> None:
        """Test maximum drawdown calculation."""
        portfolio_values = pd.Series([100, 105, 102, 98, 103, 95, 100])
        max_dd = self.calculator.calculate_max_drawdown(portfolio_values)

        assert max_dd < 0  # Max drawdown is negative
        assert abs(max_dd) > 0  # But should be non-zero

    def test_calculate_all_metrics(self) -> None:
        """Test comprehensive metrics calculation."""
        portfolio_values = pd.Series([100000, 101000, 99000, 102000, 98000])
        trades = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5),
                "pnl": [1000, -1000, 2000, -2000, 1000],
            }
        )

        metrics = self.calculator.calculate_all_metrics(portfolio_values, trades)

        # metrics is a BacktestMetrics object
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "total_return")
        assert hasattr(metrics, "max_drawdown")
        assert hasattr(metrics, "win_rate")
        assert hasattr(metrics, "total_trades")


class TestStrategyAdapters:
    """Test strategy adapter implementations."""

    def setup_method(self) -> None:
        """Set up test data."""
        self.data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="h"),
                "close": [100 + i * 0.1 for i in range(100)],
                "volume": [1000.0] * 100,
            }
        )

    def test_sma_adapter(self) -> None:
        """Test SMA strategy adapter."""
        adapter = SMACrossoverAdapter(fast_period=5, slow_period=20)

        signals = adapter.generate_signals(self.data)

        assert len(signals) == len(self.data)
        assert "signal" in signals.columns
        assert all(signals["signal"].isin([-1, 0, 1]))

    def test_buy_hold_adapter(self) -> None :
        """Test buy and hold strategy adapter."""
        adapter = BuyAndHoldAdapter()

        signals = adapter.generate_signals(self.data)

        assert len(signals) == len(self.data)
        assert "signal" in signals.columns
        # Buy and hold should have initial buy signal and hold
        assert signals["signal"].iloc[0] == 1  # Buy signal
        assert all(signals["signal"].iloc[1:] == 0)  # Hold signals

    def test_rl_adapter(self) -> None:
        """Test RL policy adapter."""
        adapter = RLPolicyAdapter(model_path="dummy_path")

        signals = adapter.generate_signals(self.data)

        assert len(signals) == len(self.data)
        assert "signal" in signals.columns
        # RL adapter should generate signals (mocked for now)
        assert isinstance(signals, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__])
