#!/usr/bin/env python3
"""
Unit tests for signal_performance package.

Tests the integration of SignalPerformanceAnalyzer with unified backtest framework.
"""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
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
        timestamp = pd.Timestamp(datetime(2023, 1, 1, 12, 0, 0))
        signal_data = {
            "signal_type": "buy",
            "direction": 1.0,
            "strength": 0.8,
            "confidence": 0.9,
            "source_patterns": ["pattern1", "pattern2"]
        }
        market_data = pd.Series({
            "open": 50000, "high": 51000, "low": 49000,
            "close": 50500, "volume": 100, "returns": 0.01
        })
        position_before = 0
        position_after = 1
        trade_executed = True
        trade_result = {"pnl": 100.0, "action": "buy"}

        self.tracker.track_signal(
            timestamp, signal_data, market_data, position_before,
            position_after, trade_executed, trade_result
        )

        self.assertEqual(len(self.tracker.signals), 1)
        signal = self.tracker.signals[0]
        self.assertEqual(signal.timestamp, timestamp)
        self.assertEqual(signal.signal_type, "buy")
        self.assertEqual(signal.direction, 1.0)
        self.assertEqual(signal.confidence, 0.9)
        self.assertEqual(signal.position_before, 0)
        self.assertEqual(signal.position_after, 1)
        self.assertTrue(signal.trade_executed)

    def test_get_signal_summary(self):
        """Test signal summary generation."""
        # Add some test signals
        timestamp = pd.Timestamp(datetime(2023, 1, 1, 12, 0, 0))
        signal_data = {
            "signal_type": "buy", "direction": 1.0, "strength": 0.8,
            "confidence": 0.9, "source_patterns": ["pattern1"]
        }
        market_data = pd.Series({"close": 50000, "volume": 100})

        self.tracker.track_signal(
            timestamp, signal_data, market_data, 0, 1, True, {"pnl": 100.0}
        )

        summary = self.tracker.get_signal_summary()
        self.assertIn("total_signals", summary)
        self.assertIn("executed_trades", summary)
        self.assertEqual(summary["total_signals"], 1)
        self.assertEqual(summary["executed_trades"], 1)


class TestBacktestPerformanceAnalyzer(unittest.TestCase):
    """Test BacktestPerformanceAnalyzer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = BacktestPerformanceAnalyzer()

    def test_record_trade_outcome(self):
        """Test recording trade outcome."""
        timestamp = pd.Timestamp(datetime(2023, 1, 1, 12, 0, 0))
        trade_result = {
            "action": "buy",
            "price": 50000,
            "shares": 1.0,
            "pnl": 1000.0
        }
        signal_data = {
            "signal_type": "buy",
            "confidence": 0.8,
            "strength": 0.7
        }

        self.analyzer.record_trade_outcome(timestamp, trade_result, signal_data)

        # Check that trade was recorded
        self.assertEqual(len(self.analyzer.trade_outcomes), 1)
        outcome = self.analyzer.trade_outcomes[0]
        self.assertEqual(outcome["action"], "buy")
        self.assertEqual(outcome["pnl"], 1000.0)

    @patch('ztb.trading.strategies.action_signal_guide.analysis.signal_performance_analyzer.SignalPerformanceAnalyzer.analyze_correlation')
    def test_get_performance_report(self, mock_analyze_correlation):
        """Test generating performance report."""
        mock_analyze_correlation.return_value = {
            "correlation_coefficient": 0.75,
            "p_value": 0.01,
            "significant": True
        }

        # Add some test data
        timestamp = pd.Timestamp(datetime(2023, 1, 1, 12, 0, 0))
        self.analyzer.record_trade_outcome(
            timestamp,
            {"action": "buy", "price": 50000, "pnl": 1000},
            {"signal_type": "buy", "confidence": 0.8}
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
        timestamp = pd.Timestamp(datetime(2023, 1, 1, 12, 0, 0))
        signal_data = {
            "signal_type": "buy", "direction": 1.0, "strength": 0.8,
            "confidence": 0.9, "source_patterns": ["pattern1"]
        }
        market_data = pd.Series({"close": 50000, "volume": 100})

        self.analyzer.track_signal(
            timestamp, signal_data, market_data, 0, 1, True, {"pnl": 100.0}
        )

        # Check that signal was tracked
        self.assertEqual(len(self.analyzer.signal_tracker.signals), 1)

    def test_record_trade_outcome_integration(self):
        """Test trade outcome recording through main analyzer."""
        timestamp = pd.Timestamp(datetime(2023, 1, 1, 12, 0, 0))
        trade_result = {"action": "buy", "price": 50000, "pnl": 500}
        signal_data = {"signal_type": "buy", "confidence": 0.8}

        self.analyzer.record_trade_outcome(timestamp, trade_result, signal_data)

        # Check that trade outcome was recorded
        self.assertEqual(len(self.analyzer.performance_analyzer.trade_outcomes), 1)

    def test_get_performance_report_integration(self):
        """Test integrated performance report generation."""
        # Add signal and trade data
        timestamp = pd.Timestamp(datetime(2023, 1, 1, 12, 0, 0))
        signal_data = {"signal_type": "buy", "direction": 1.0, "confidence": 0.8}
        market_data = pd.Series({"close": 50000})

        self.analyzer.track_signal(timestamp, signal_data, market_data, 0, 1, True)
        self.analyzer.record_trade_outcome(
            timestamp,
            {"action": "buy", "price": 50000, "pnl": 1000},
            signal_data
        )

        report = self.analyzer.get_performance_report()

        self.assertIn("signal_tracking", report)
        self.assertIn("performance_analysis", report)
        self.assertIn("total_signals", report["signal_tracking"])
        self.assertIn("total_trades", report["performance_analysis"])


if __name__ == '__main__':
    unittest.main()