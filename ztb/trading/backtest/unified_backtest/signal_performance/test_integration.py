#!/usr/bin/env python3
"""
Integration tests for SignalPerformanceAnalyzer with UnifiedBacktester.

Tests the complete integration of signal performance analysis during backtesting.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from ztb.trading.backtest.unified_backtest import UnifiedBacktester, BacktestConfig
from ztb.trading.backtest.unified_backtest.signal_performance import BacktestSignalPerformanceAnalyzer
from ztb.trading.backtest.adapters import StrategyAdapter


class MockStrategy(StrategyAdapter):
    """Mock strategy for testing."""

    def __init__(self, signals=None):
        self.signals = signals or []
        self.signal_index = 0

    def generate_signal(self, data, position):
        """Generate mock signals."""
        if self.signal_index < len(self.signals):
            signal = self.signals[self.signal_index]
            self.signal_index += 1
            return signal
        return {"action": "hold"}

    @property
    def name(self):
        return "mock_strategy"


class TestUnifiedBacktesterSignalIntegration(unittest.TestCase):
    """Test UnifiedBacktester with signal performance analyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.backtester = UnifiedBacktester()

        # Create mock market data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        self.market_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(49000, 51000, 100),
            'high': np.random.uniform(50000, 52000, 100),
            'low': np.random.uniform(48000, 50000, 100),
            'close': np.random.uniform(49000, 51000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }).set_index('timestamp')

    def test_signal_performance_analyzer_initialization(self):
        """Test that signal performance analyzer is properly initialized."""
        self.assertIsInstance(self.backtester.signal_performance_analyzer, BacktestSignalPerformanceAnalyzer)

    def test_backtest_with_signal_tracking(self):
        """Test backtest execution with signal tracking enabled."""
        # Create mock strategy with predefined signals
        signals = [
            {"action": "buy", "confidence": 0.8},
            {"action": "hold"},
            {"action": "sell", "confidence": 0.7},
            {"action": "hold"},
        ] * 25  # Repeat to cover all data points

        strategy = MockStrategy(signals)
        self.backtester.register_strategy("test_strategy", strategy)

        config = BacktestConfig(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005
        )

        # Run backtest
        result = self.backtester.run_backtest(
            "test_strategy",
            self.market_data,
            config,
            save_results=False
        )

        # Verify result structure
        self.assertEqual(result.strategy_name, "test_strategy")
        self.assertIsInstance(result.performance_metrics, object)  # BacktestMetrics
        self.assertIsInstance(result.trade_history, list)
        self.assertIsInstance(result.portfolio_values, list)

        # Check that signal performance data is included in metadata
        self.assertIn("signal_performance", result.metadata)

    def test_signal_tracking_during_backtest(self):
        """Test that signals are properly tracked during backtest execution."""
        signals = [
            {"action": "buy", "confidence": 0.9},
            {"action": "hold"},
            {"action": "hold"},
            {"action": "sell", "confidence": 0.8},
        ] * 25

        strategy = MockStrategy(signals)
        self.backtester.register_strategy("tracking_test", strategy)

        config = BacktestConfig(initial_capital=10000)

        # Run backtest
        result = self.backtester.run_backtest(
            "tracking_test",
            self.market_data,
            config,
            save_results=False
        )

        # Verify signal performance data
        signal_perf = result.metadata.get("signal_performance", {})
        self.assertIn("signal_tracking", signal_perf)
        self.assertIn("performance_analysis", signal_perf)

        # Check signal counts
        tracking_data = signal_perf["signal_tracking"]
        self.assertGreater(tracking_data["total_signals"], 0)

    def test_trade_outcome_recording(self):
        """Test that trade outcomes are properly recorded."""
        # Create signals that should result in trades
        signals = [
            {"action": "buy", "confidence": 0.8},  # Should execute buy
            {"action": "hold"},
            {"action": "hold"},
            {"action": "sell", "confidence": 0.7},  # Should execute sell
        ] * 25

        strategy = MockStrategy(signals)
        self.backtester.register_strategy("trade_test", strategy)

        config = BacktestConfig(initial_capital=10000)

        result = self.backtester.run_backtest(
            "trade_test",
            self.market_data,
            config,
            save_results=False
        )

        # Check that trades were recorded
        self.assertGreater(len(result.trade_history), 0)

        # Check signal performance includes trade outcomes
        signal_perf = result.metadata.get("signal_performance", {})
        perf_analysis = signal_perf.get("performance_analysis", {})
        self.assertIn("total_trades", perf_analysis)

    def test_signal_performance_report_generation(self):
        """Test that comprehensive performance reports are generated."""
        signals = [
            {"action": "buy", "confidence": 0.85},
            {"action": "hold"},
            {"action": "sell", "confidence": 0.75},
        ] * 33

        strategy = MockStrategy(signals)
        self.backtester.register_strategy("report_test", strategy)

        config = BacktestConfig(initial_capital=10000)

        result = self.backtester.run_backtest(
            "report_test",
            self.market_data,
            config,
            save_results=False
        )

        signal_perf = result.metadata.get("signal_performance", {})

        # Verify report structure
        required_keys = [
            "signal_tracking",
            "performance_analysis",
            "correlation_analysis",
            "signal_quality_metrics"
        ]

        for key in required_keys:
            self.assertIn(key, signal_perf, f"Missing key: {key}")

    def test_multiple_strategy_comparison_with_signals(self):
        """Test comparing multiple strategies with signal performance analysis."""
        # Strategy 1: Conservative
        signals1 = [
            {"action": "buy", "confidence": 0.6},
            {"action": "hold"},
            {"action": "hold"},
            {"action": "sell", "confidence": 0.6},
        ] * 25

        # Strategy 2: Aggressive
        signals2 = [
            {"action": "buy", "confidence": 0.9},
            {"action": "hold"},
            {"action": "sell", "confidence": 0.9},
        ] * 33

        strategy1 = MockStrategy(signals1)
        strategy2 = MockStrategy(signals2)

        self.backtester.register_strategy("conservative", strategy1)
        self.backtester.register_strategy("aggressive", strategy2)

        config = BacktestConfig(initial_capital=10000)

        # Compare strategies
        results = self.backtester.compare_strategies(
            ["conservative", "aggressive"],
            self.market_data,
            config
        )

        # Verify both results have signal performance data
        for strategy_name, result in results.items():
            self.assertIn("signal_performance", result.metadata)
            signal_perf = result.metadata["signal_performance"]
            self.assertIn("signal_tracking", signal_perf)
            self.assertIn("performance_analysis", signal_perf)


if __name__ == '__main__':
    unittest.main()