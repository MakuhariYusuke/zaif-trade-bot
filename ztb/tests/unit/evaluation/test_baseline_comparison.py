"""Unit tests for baseline comparison functionality."""

import numpy as np
import pandas as pd

from ztb.evaluation.baseline_comparison import (
    BaselineComparisonEngine,
    BaselineResult,
    BuyAndHoldStrategy,
    SMAStrategy,
)


class TestBuyAndHoldStrategy:
    """Test buy and hold baseline strategy."""

    def test_evaluate_buy_and_hold(self):
        """Test buy and hold evaluation."""
        # Create sample price data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.random.randn(100).cumsum() + 100  # Random walk
        price_data = pd.DataFrame({"close": prices}, index=dates)

        strategy = BuyAndHoldStrategy("Buy and Hold")
        result = strategy.evaluate(price_data)

        assert result.strategy_name == "Buy and Hold"
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert result.total_trades == 1
        assert "start_price" in result.metrics
        assert "end_price" in result.metrics


class TestSMAStrategy:
    """Test SMA crossover strategy."""

    def test_evaluate_sma_strategy(self):
        """Test SMA strategy evaluation."""
        # Create trending price data
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        trend = np.linspace(100, 120, 200)
        noise = np.random.randn(200) * 2
        prices = trend + noise
        price_data = pd.DataFrame({"close": prices}, index=dates)

        strategy = SMAStrategy("SMA Crossover")
        result = strategy.evaluate(price_data, fast_period=10, slow_period=30)

        assert result.strategy_name == "SMA Crossover"
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.win_rate, float)
        assert result.total_trades >= 0
        assert result.metrics["fast_period"] == 10
        assert result.metrics["slow_period"] == 30


class TestBaselineComparisonEngine:
    """Test baseline comparison engine."""

    def test_compare_with_baselines(self):
        """Test comparison against baselines."""
        # Create sample price data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.random.randn(100).cumsum() + 100
        price_data = pd.DataFrame({"close": prices}, index=dates)

        engine = BaselineComparisonEngine()

        # Mock model result
        model_result = BaselineResult(
            strategy_name="Trained Model",
            total_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=-0.1,
            win_rate=0.55,
            total_trades=50,
            metrics={},
        )

        comparison = engine.compare(model_result, price_data)

        assert comparison.model_result == model_result
        assert len(comparison.baseline_results) >= 1  # At least buy_hold
        assert len(comparison.superiority_metrics) > 0
        assert len(comparison.statistical_tests) > 0

    def test_generate_report(self):
        """Test report generation."""
        # Create sample price data
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        prices = np.random.randn(50).cumsum() + 100
        price_data = pd.DataFrame({"close": prices}, index=dates)

        engine = BaselineComparisonEngine()

        model_result = BaselineResult(
            strategy_name="Test Model",
            total_return=0.10,
            sharpe_ratio=1.2,
            max_drawdown=-0.08,
            win_rate=0.52,
            total_trades=25,
            metrics={},
        )

        comparison = engine.compare(model_result, price_data)
        report = engine.generate_report(comparison)

        assert "# Baseline Comparison Report" in report
        assert "## Model Performance" in report
        assert "## Baseline Strategies" in report
        assert "## Superiority Metrics" in report
        assert "Test Model" in report

    def test_add_custom_strategy(self):
        """Test adding custom baseline strategy."""
        engine = BaselineComparisonEngine()

        custom_strategy = BuyAndHoldStrategy("Custom Strategy")
        engine.add_strategy(custom_strategy)

        assert "Custom Strategy" in engine.strategies
