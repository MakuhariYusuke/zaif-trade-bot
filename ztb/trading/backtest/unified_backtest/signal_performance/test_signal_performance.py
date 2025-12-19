#!/usr/bin/env python3
"""
Unit tests for signal_performance package.

Tests the integration of SignalPerformanceAnalyzer with unified backtest framework.
"""

import unittest
from datetime import datetime
from unittest.mock import patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

from ztb.trading.backtest.unified_backtest.signal_performance import (
    BacktestSignalPerformanceAnalyzer,
    SignalTracker,
    BacktestPerformanceAnalyzer
)


class TestSignalTracker(unittest.TestCase):
    """Test SignalTracker functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = SignalTracker()

    def test_track_signal(self):
        """Test signal tracking."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        signal_data = {"action": "buy", "confidence": 0.8}
        market_data = {"close": 50000, "volume": 100}
        position = 0

        self.tracker.track_signal(timestamp, signal_data, market_data, position)

        self.assertEqual(len(self.tracker.signals), 1)
        signal = self.tracker.signals[0]
        self.assertEqual(signal.timestamp, timestamp)
        self.assertEqual(signal.action, "buy")
        self.assertEqual(signal.confidence, 0.8)
        self.assertEqual(signal.market_price, 50000)
        self.assertEqual(signal.position, 0)

    def test_get_signals_in_range(self):
        """Test retrieving signals in time range."""
        base_time = datetime(2023, 1, 1, 12, 0, 0)

        # Add signals at different times
        self.tracker.track_signal(
            base_time, {"action": "buy", "confidence": 0.8},
            {"close": 50000}, 0
        )
        self.tracker.track_signal(
            base_time.replace(hour=13), {"action": "sell", "confidence": 0.7},
            {"close": 51000}, 1
        )
        self.tracker.track_signal(
            base_time.replace(hour=14), {"action": "buy", "confidence": 0.9},
            {"close": 52000}, 0
        )

        # Test range query
        start_time = base_time.replace(hour=12, minute=30)
        end_time = base_time.replace(hour=13, minute=30)

        signals_in_range = self.tracker.get_signals_in_range(start_time, end_time)
        self.assertEqual(len(signals_in_range), 1)
        self.assertEqual(signals_in_range[0].action, "sell")


class TestBacktestPerformanceAnalyzer(unittest.TestCase):
    """Test BacktestPerformanceAnalyzer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = BacktestPerformanceAnalyzer()

    def test_record_trade_outcome_executed(self):
        """Test recording executed trade outcome."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        trade_data = {
            "action": "buy",
            "price": 50000,
            "shares": 1.0,
            "pnl": 0.0
        }

        self.analyzer.record_trade_outcome(timestamp, trade_data, "executed")

        # Check that trade was recorded
        self.assertIn(timestamp, self.analyzer.trade_outcomes)
        outcome = self.analyzer.trade_outcomes[timestamp]
        self.assertEqual(outcome["outcome"], "executed")
        self.assertEqual(outcome["action"], "buy")

    def test_record_trade_outcome_failed(self):
        """Test recording failed trade outcome."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        trade_data = {
            "action": "buy",
            "price": 50000,
            "reason": "insufficient_capital"
        }

        self.analyzer.record_trade_outcome(timestamp, trade_data, "failed")

        outcome = self.analyzer.trade_outcomes[timestamp]
        self.assertEqual(outcome["outcome"], "failed")
        self.assertEqual(outcome["reason"], "insufficient_capital")

    @patch('ztb.trading.signal_performance.SignalPerformanceAnalyzer.analyze_correlation')
    def test_get_performance_report(self, mock_analyze_correlation):
        """Test generating performance report."""
        mock_analyze_correlation.return_value = {
            "correlation_coefficient": 0.75,
            "p_value": 0.01,
            "significant": True
        }

        # Add some test data
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        self.analyzer.record_trade_outcome(
            timestamp,
            {"action": "buy", "price": 50000, "pnl": 1000},
            "executed"
        )

        report = self.analyzer.get_performance_report()

        self.assertIn("signal_quality_score", report)
        self.assertIn("correlation_analysis", report)
        self.assertIn("trade_outcomes", report)
        self.assertEqual(len(report["trade_outcomes"]), 1)


class TestBacktestSignalPerformanceAnalyzer(unittest.TestCase):
    """Test BacktestSignalPerformanceAnalyzer integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = BacktestSignalPerformanceAnalyzer()

    def test_track_signal_integration(self):
        """Test signal tracking through main analyzer."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        signal_data = {"action": "buy", "confidence": 0.8}
        market_data = {"close": 50000}
        position = 0

        self.analyzer.track_signal(timestamp, signal_data, market_data, position)

        # Check that signal was tracked
        self.assertEqual(len(self.analyzer.signal_tracker.signals), 1)

    def test_record_trade_outcome_integration(self):
        """Test trade outcome recording through main analyzer."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        trade_data = {"action": "buy", "price": 50000, "pnl": 500}

        self.analyzer.record_trade_outcome(timestamp, trade_data, "executed")

        # Check that trade outcome was recorded
        self.assertEqual(len(self.analyzer.performance_analyzer.trade_outcomes), 1)

    def test_get_performance_report_integration(self):
        """Test integrated performance report generation."""
        # Add signal and trade data
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        self.analyzer.track_signal(
            timestamp,
            {"action": "buy", "confidence": 0.8},
            {"close": 50000},
            0
        )
        self.analyzer.record_trade_outcome(
            timestamp,
            {"action": "buy", "price": 50000, "pnl": 1000},
            "executed"
        )

        report = self.analyzer.get_performance_report()

        self.assertIn("signal_tracking", report)
        self.assertIn("performance_analysis", report)
        self.assertIn("total_signals", report["signal_tracking"])
        self.assertIn("total_trades", report["performance_analysis"])


if __name__ == '__main__':
    unittest.main()
