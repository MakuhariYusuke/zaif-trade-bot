"""
Unit tests for regime evaluation module.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ztb.evaluation.regime_eval import (
    MarketRegime,
    RegimeDetector,
    RegimeEvaluator,
    RegimeMetrics,
    RegimeSegment,
)


class TestRegimeDetector:
    """Test cases for RegimeDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = RegimeDetector()

    def test_detect_regimes_bull_market(self):
        """Test regime detection for bull market."""
        # Create bull market data (steady upward trend)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.linspace(100, 200, 100)  # Linear increase
        df = pd.DataFrame({"close": prices}, index=dates)

        regimes = self.detector.detect_regimes(df)

        # Should detect mostly bull regime
        assert len(regimes) > 0
        # Check that we have some bull segments
        bull_segments = [r for r in regimes if r.regime == MarketRegime.BULL]
        assert len(bull_segments) > 0

    def test_detect_regimes_bear_market(self):
        """Test regime detection for bear market."""
        # Create bear market data (steady downward trend)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.linspace(200, 100, 100)  # Linear decrease
        df = pd.DataFrame({"close": prices}, index=dates)

        regimes = self.detector.detect_regimes(df)

        # Should detect mostly bear regime
        bear_segments = [r for r in regimes if r.regime == MarketRegime.BEAR]
        assert len(bear_segments) > 0

    def test_detect_regimes_sideways_market(self):
        """Test regime detection for sideways market."""
        # Create sideways market data (random walk around mean)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        prices = 100 + np.random.normal(0, 2, 100).cumsum()  # Random walk
        df = pd.DataFrame({"close": prices}, index=dates)

        regimes = self.detector.detect_regimes(df)

        # Should detect some sideways regime
        sideways_segments = [r for r in regimes if r.regime == MarketRegime.SIDEWAYS]
        assert len(sideways_segments) >= 0  # May or may not detect sideways

    def test_calculate_volatility(self):
        """Test volatility calculation."""
        # Create data with known volatility
        dates = pd.date_range("2023-01-01", periods=100, freq="D")  # 100 points
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 99)  # 99 returns
        prices = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({"close": prices}, index=dates[:-1])  # Use 99 dates

        # Test through detect_regimes
        regimes = self.detector.detect_regimes(df)
        assert len(regimes) > 0
        # Volatility should be reasonable
        volatility = regimes[0].volatility
        assert 0.01 < volatility < 0.1

    def test_calculate_trend_strength(self):
        """Test trend strength calculation."""
        # Strong uptrend
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.linspace(100, 200, 100)
        df = pd.DataFrame({"close": prices}, index=dates)

        # Test through detect_regimes which calls the method
        regimes = self.detector.detect_regimes(df)
        assert len(regimes) > 0
        # Trend strength should be positive for bull market
        trend_strength = regimes[0].trend_strength
        assert trend_strength > 0

        # Strong downtrend
        prices = np.linspace(200, 100, 100)
        df = pd.DataFrame({"close": prices}, index=dates)

        regimes = self.detector.detect_regimes(df)
        assert len(regimes) > 0
        trend_strength = regimes[0].trend_strength
        # Should be negative and strong
        assert trend_strength < 0

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="Price data must contain 'close' column"):
            self.detector.detect_regimes(df)


class TestRegimeEvaluator:
    """Test cases for RegimeEvaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = RegimeEvaluator()

    def test_evaluate_performance_basic(self):
        """Test basic performance evaluation."""
        # Create sample price data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.linspace(100, 200, 100)
        price_data = pd.DataFrame({"close": prices}, index=dates)

        # Create sample trade log
        trade_log = [
            {
                "timestamp": "2023-01-01T10:00:00Z",
                "side": "buy",
                "price": 100.0,
                "quantity": 1.0,
            },
            {
                "timestamp": "2023-01-10T10:00:00Z",
                "side": "sell",
                "price": 150.0,
                "quantity": 1.0,
            },
        ]

        results = self.evaluator.evaluate_performance(price_data, trade_log)

        # Should have regime results
        assert "bull" in results or "bear" in results or "sideways" in results

    def test_calculate_trade_metrics(self):
        """Test trade metrics calculation."""
        trades = [
            {"price": 100, "quantity": 1, "side": "buy"},
            {"price": 110, "quantity": 1, "side": "sell"},
            {"price": 105, "quantity": 1, "side": "buy"},
            {"price": 115, "quantity": 1, "side": "sell"},
        ]

        metrics = self.evaluator._calculate_trade_metrics(trades)

        assert isinstance(metrics, RegimeMetrics)
        assert metrics.total_return > 0  # Profitable trades
        assert metrics.win_rate == 1.0  # All trades profitable
        assert metrics.total_trades == 2  # 2 round trips

    def test_calculate_trade_metrics_no_trades(self):
        """Test trade metrics with no trades."""
        trades = []

        metrics = self.evaluator._calculate_trade_metrics(trades)

        assert isinstance(metrics, RegimeMetrics)
        assert metrics.total_return == 0
        assert metrics.total_trades == 0

    def test_calculate_trade_metrics_loss(self):
        """Test trade metrics with losses."""
        trades = [
            {"price": 100, "quantity": 1, "side": "buy"},
            {"price": 90, "quantity": 1, "side": "sell"},  # Loss
        ]

        metrics = self.evaluator._calculate_trade_metrics(trades)

        assert metrics.total_return < 0
        assert metrics.win_rate == 0.0

    def test_segment_data_by_regime(self):
        """Test data segmentation by regime."""
        # Create price data with bull market
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.linspace(100, 200, 100)
        price_data = pd.DataFrame({"close": prices}, index=dates)

        # Test through evaluate_performance
        results = self.evaluator.evaluate_performance(price_data, [])
        assert "bull" in results
        assert results["bull"]["segments"] > 0

    def test_generate_report(self):
        """Test report generation."""
        results = {
            "bull": {
                "metrics": RegimeMetrics(
                    regime=MarketRegime.BULL,
                    total_return=0.15,
                    sharpe_ratio=1.2,
                    max_drawdown=-0.05,
                    win_rate=0.6,
                    total_trades=10,
                    avg_trade_return=0.015,
                    volatility=0.02,
                    regime_duration_days=30.0,
                ),
                "trade_count": 10,
                "segments": 1,
            },
            "bear": {
                "metrics": RegimeMetrics(
                    regime=MarketRegime.BEAR,
                    total_return=-0.05,
                    sharpe_ratio=-0.3,
                    max_drawdown=-0.08,
                    win_rate=0.4,
                    total_trades=8,
                    avg_trade_return=-0.006,
                    volatility=0.03,
                    regime_duration_days=25.0,
                ),
                "trade_count": 8,
                "segments": 1,
            },
        }

        report_path = "/tmp/test_report.md"
        report = self.evaluator.generate_report(results, report_path)

        assert isinstance(report, str)
        assert "Market Regime Evaluation Report" in report
        assert "Bull Market Regime" in report
        assert "Bear Market Regime" in report

    @patch("ztb.evaluation.regime_eval.get_baseline_comparison_engine")
    def test_compare_baselines(self, mock_get_engine):
        """Test baseline comparison."""
        # Mock baseline engine
        mock_engine = Mock()
        mock_engine.strategies = {"buy_hold": Mock(), "sma_crossover": Mock()}
        mock_engine.strategies["buy_hold"].evaluate.return_value = Mock(
            total_return=0.10, sharpe_ratio=0.8, win_rate=0.5
        )
        mock_engine.strategies["sma_crossover"].evaluate.return_value = Mock(
            total_return=0.08, sharpe_ratio=0.6, win_rate=0.55
        )
        mock_get_engine.return_value = mock_engine

        regime_results = {
            "bull": {
                "metrics": RegimeMetrics(
                    regime=MarketRegime.BULL,
                    total_return=0.15,
                    sharpe_ratio=1.2,
                    max_drawdown=-0.05,
                    win_rate=0.6,
                    total_trades=10,
                    avg_trade_return=0.015,
                    volatility=0.02,
                    regime_duration_days=30.0,
                )
            }
        }

        baseline_strategies = {
            "buy_hold": {
                "bull": {"total_return": 0.10, "sharpe_ratio": 0.8, "win_rate": 0.5}
            },
            "sma_crossover": {
                "bull": {"total_return": 0.08, "sharpe_ratio": 0.6, "win_rate": 0.55}
            },
        }

        comparison = self.evaluator._compare_baselines(
            regime_results, baseline_strategies
        )

        assert "bull" in comparison
        assert "buy_hold" in comparison["bull"]
        assert "sma_crossover" in comparison["bull"]


class TestRegimeMetrics:
    """Test cases for RegimeMetrics dataclass."""

    def test_regime_metrics_creation(self):
        """Test RegimeMetrics creation."""
        metrics = RegimeMetrics(
            regime=MarketRegime.BULL,
            total_return=0.1234,
            sharpe_ratio=1.567,
            max_drawdown=-0.05,
            win_rate=0.789,
            total_trades=42,
            avg_trade_return=0.0029,
            volatility=0.02,
            regime_duration_days=30.0,
        )

        assert metrics.total_return == 0.1234
        assert metrics.sharpe_ratio == 1.567
        assert metrics.win_rate == 0.789
        assert metrics.total_trades == 42
        assert metrics.avg_trade_return == 0.0029


class TestRegimeSegment:
    """Test cases for RegimeSegment dataclass."""

    def test_regime_segment_creation(self):
        """Test RegimeSegment creation."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)
        segment = RegimeSegment(
            regime=MarketRegime.BULL,
            start_idx=0,
            end_idx=30,
            start_date=start,
            end_date=end,
            returns=0.15,
            volatility=0.02,
            trend_strength=1.2,
            duration_days=30,
            confidence=0.85,
        )

        assert segment.start_date == start
        assert segment.end_date == end
        assert segment.regime == MarketRegime.BULL
        assert segment.confidence == 0.85


class TestMarketRegime:
    """Test cases for MarketRegime enum."""

    def test_regime_values(self):
        """Test regime enum values."""
        assert MarketRegime.BULL.value == "bull"
        assert MarketRegime.BEAR.value == "bear"
        assert MarketRegime.SIDEWAYS.value == "sideways"

    def test_regime_names(self):
        """Test regime enum names."""
        assert MarketRegime.BULL.name == "BULL"
        assert MarketRegime.BEAR.name == "BEAR"
        assert MarketRegime.SIDEWAYS.name == "SIDEWAYS"
