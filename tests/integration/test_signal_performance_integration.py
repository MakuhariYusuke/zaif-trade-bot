#!/usr/bin/env python3
"""
Signal Performance Integration Tests

Comprehensive integration tests for SignalPerformanceAnalyzer with UnifiedBacktester.
Tests cover signal tracking, trade outcome recording, and performance analysis.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from ztb.trading.backtest.signal_performance.signal_tracker import SignalTracker
from ztb.trading.backtest.signal_performance.performance_analyzer import BacktestPerformanceAnalyzer
from ztb.trading.backtest.signal_performance.backtest_integration import BacktestSignalPerformanceAnalyzer


class MockDataGenerator:
    """Helper class for generating mock test data."""

    @staticmethod
    def create_market_data(hours: int = 100, start_price: float = 50000) -> pd.DataFrame:
        """Create mock market data for testing."""
        start_time = datetime(2023, 1, 1, 0, 0, 0)
        timestamps = [start_time + timedelta(hours=i) for i in range(hours)]

        # Generate realistic price data
        np.random.seed(42)
        prices = []
        current_price = start_price

        for i in range(hours):
            # Random walk with slight upward trend
            change = np.random.normal(0.001, 0.02)  # Mean 0.1%, std 2%
            current_price *= (1 + change)
            prices.append(current_price)

        # Create OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Create OHLC around the price
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = data[-1]['close'] if data else price
            volume = np.random.uniform(50, 200)

            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    @staticmethod
    def create_signal_data(signal_type: str = "buy", strength: float = 0.8,
                          confidence: float = 0.9) -> Dict[str, Any]:
        """Create mock signal data."""
        return {
            'signal_type': signal_type,
            'direction': 1.0 if signal_type == "buy" else -1.0,
            'strength': strength,
            'confidence': confidence,
            'source_patterns': ['test_pattern', 'fibonacci']
        }

    @staticmethod
    def create_trade_result(action: str = "buy", price: float = 50000,
                           pnl: float = 100.0) -> Dict[str, Any]:
        """Create mock trade result."""
        return {
            'action': action,
            'price': price,
            'pnl': pnl,
            'shares': 1.0,
            'timestamp': pd.Timestamp(datetime(2023, 1, 1, 12, 0, 0))
        }


class TestSignalPerformanceIntegration(unittest.TestCase):
    """Integration tests for signal performance system."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_data = MockDataGenerator()
        self.analyzer = BacktestSignalPerformanceAnalyzer()

    def test_basic_signal_tracking(self):
        """Test basic signal tracking functionality."""
        # Create test data
        timestamp = pd.Timestamp(datetime(2023, 1, 1, 12, 0, 0))
        signal_data = self.mock_data.create_signal_data()
        market_series = pd.Series({
            'open': 50000, 'high': 51000, 'low': 49000,
            'close': 50500, 'volume': 100, 'returns': 0.01
        })

        # Track signal
        self.analyzer.track_signal(
            timestamp, signal_data, market_series,
            position_before=0, position_after=1,
            trade_executed=True,
            trade_result={'pnl': 100.0, 'action': 'buy'}
        )

        # Verify signal was tracked
        report = self.analyzer.get_performance_report()
        self.assertIn("signal_tracking", report)
        self.assertEqual(report["signal_tracking"]["total_signals"], 1)
        self.assertEqual(report["signal_tracking"]["executed_trades"], 1)

    def test_trade_outcome_recording(self):
        """Test trade outcome recording and analysis."""
        timestamp = pd.Timestamp(datetime(2023, 1, 1, 12, 0, 0))
        signal_data = self.mock_data.create_signal_data()
        trade_result = self.mock_data.create_trade_result()

        # Record trade outcome
        self.analyzer.record_trade_outcome(timestamp, trade_result, signal_data)

        # Verify trade was recorded
        report = self.analyzer.get_performance_report()
        self.assertIn("performance_analysis", report)

        perf_analysis = report["performance_analysis"]
        self.assertIn("trade_count", perf_analysis)
        self.assertEqual(perf_analysis["trade_count"], 1)
        self.assertEqual(perf_analysis["total_return"], 100.0)

    def test_complete_integration_workflow(self):
        """Test complete integration workflow from signal to performance report."""
        # Create multiple signals and trades
        timestamps = [
            pd.Timestamp(datetime(2023, 1, 1, 12, 0, 0)),
            pd.Timestamp(datetime(2023, 1, 1, 13, 0, 0)),
            pd.Timestamp(datetime(2023, 1, 1, 14, 0, 0))
        ]

        market_series = pd.Series({
            'open': 50000, 'high': 51000, 'low': 49000,
            'close': 50500, 'volume': 100, 'returns': 0.01
        })

        # Track signals and record trades
        for i, timestamp in enumerate(timestamps):
            signal_data = self.mock_data.create_signal_data(
                signal_type="buy" if i % 2 == 0 else "sell",
                strength=0.7 + i * 0.1,
                confidence=0.8 + i * 0.05
            )

            # Track signal
            self.analyzer.track_signal(
                timestamp, signal_data, market_series,
                position_before=0, position_after=1 if i % 2 == 0 else -1,
                trade_executed=True,
                trade_result={'pnl': 100.0 * (1 if i % 2 == 0 else -1), 'action': signal_data['signal_type']}
            )

            # Record trade outcome
            trade_result = self.mock_data.create_trade_result(
                action=signal_data['signal_type'],
                pnl=100.0 * (1 if i % 2 == 0 else -1)
            )
            self.analyzer.record_trade_outcome(timestamp, trade_result, signal_data)

        # Generate comprehensive report
        report = self.analyzer.get_performance_report()

        # Verify report structure
        self.assertIn("signal_tracking", report)
        self.assertIn("performance_analysis", report)
        self.assertIn("integration_status", report)
        self.assertEqual(report["integration_status"], "active")

        # Verify signal tracking
        signal_tracking = report["signal_tracking"]
        self.assertEqual(signal_tracking["total_signals"], 3)
        self.assertEqual(signal_tracking["executed_trades"], 3)

        # Verify performance analysis
        perf_analysis = report["performance_analysis"]
        self.assertEqual(perf_analysis["trade_count"], 3)
        self.assertIn("overall_win_rate", perf_analysis)
        self.assertIn("total_return", perf_analysis)

    def test_performance_report_structure(self):
        """Test that performance report has all required components."""
        # Add some test data
        timestamp = pd.Timestamp(datetime(2023, 1, 1, 12, 0, 0))
        signal_data = self.mock_data.create_signal_data()
        market_series = pd.Series({
            'open': 50000, 'high': 51000, 'low': 49000,
            'close': 50500, 'volume': 100, 'returns': 0.01
        })
        trade_result = self.mock_data.create_trade_result()

        self.analyzer.track_signal(timestamp, signal_data, market_series, 0, 1, True)
        self.analyzer.record_trade_outcome(timestamp, trade_result, signal_data)

        report = self.analyzer.get_performance_report()

        # Check top-level structure
        required_keys = ["signal_tracking", "performance_analysis", "integration_status", "report_timestamp"]
        for key in required_keys:
            self.assertIn(key, report, f"Missing required key: {key}")

        # Check signal_tracking structure
        signal_keys = ["total_signals", "executed_trades", "unique_signal_types"]
        for key in signal_keys:
            self.assertIn(key, report["signal_tracking"], f"Missing signal tracking key: {key}")

        # Check performance_analysis structure
        perf_keys = ["trade_count", "overall_win_rate", "total_return", "signal_quality_score"]
        for key in perf_keys:
            self.assertIn(key, report["performance_analysis"], f"Missing performance analysis key: {key}")

    def test_error_handling(self):
        """Test error handling in integration scenarios."""
        # Test with invalid data
        try:
            self.analyzer.track_signal(
                None, {}, pd.Series(), 0, 0, False
            )
            # Should not raise exception
        except Exception:
            # If exception occurs, ensure it's handled gracefully
            pass

        # Report should still be generatable
        report = self.analyzer.get_performance_report()
        self.assertIsInstance(report, dict)


class TestMockDataGenerator(unittest.TestCase):
    """Tests for mock data generation utilities."""

    def test_create_market_data(self):
        """Test market data generation."""
        data = MockDataGenerator.create_market_data(hours=10)
        self.assertEqual(len(data), 10)
        self.assertIn('open', data.columns)
        self.assertIn('high', data.columns)
        self.assertIn('low', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)

    def test_create_signal_data(self):
        """Test signal data generation."""
        signal = MockDataGenerator.create_signal_data()
        required_keys = ['signal_type', 'direction', 'strength', 'confidence', 'source_patterns']
        for key in required_keys:
            self.assertIn(key, signal)

    def test_create_trade_result(self):
        """Test trade result generation."""
        trade = MockDataGenerator.create_trade_result()
        required_keys = ['action', 'price', 'pnl', 'shares', 'timestamp']
        for key in required_keys:
            self.assertIn(key, trade)


if __name__ == "__main__":
    # Configure test output
    unittest.main(verbosity=2)