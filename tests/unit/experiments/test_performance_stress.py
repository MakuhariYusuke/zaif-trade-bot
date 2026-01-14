#!/usr/bin/env python3
"""
Performance and stress tests for Action Signal Guide components.

This module contains performance tests for large datasets, memory constraints,
and stress testing scenarios.
"""

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.strategies.action_signal_guide.components.market_regime import (
    MarketConditionAnalyzer,
    MarketRegimeDetector,
    RegimeAdaptiveSignalProcessor,
)
from ztb.trading.strategies.action_signal_guide.components.sac_integration import (
    SACPerformanceMonitor,
)


class TestPerformanceAndStress(unittest.TestCase):
    def setUp(self):
        self.performance_monitor = SACPerformanceMonitor()
        self.regime_detector = MarketRegimeDetector()
        self.regime_processor = RegimeAdaptiveSignalProcessor()
        self.market_analyzer = MarketConditionAnalyzer()

    def test_large_dataset_signal_validation(self):
        """Test signal validation performance with large number of signals."""
        # Generate large number of signals
        num_signals = 1000
        signals = []

        for i in range(num_signals):
            signal = MockActionSignal(
                action="BUY" if i % 2 == 0 else "SELL",
                confidence=0.5 + (i % 500) / 1000,  # Varying confidence
                price=100 + (i % 100),
                pattern_type="fibonacci" if i % 3 == 0 else "harmonic",
            )
            signals.append(signal)

        # Measure validation time
        start_time = time.time()
        validated_count = 0

        for signal in signals:
            result = self.signal_validator.validate_signal(signal)
            if result.is_valid:
                validated_count += 1

        end_time = time.time()
        validation_time = end_time - start_time

        # Performance assertions
        self.assertGreater(validated_count, 0)
        self.assertLess(validation_time, 5.0)  # Should complete within 5 seconds
        self.assertGreaterEqual(validation_time, 0.1)  # Should take at least some time

        print(
            f"Validated {validated_count}/{num_signals} signals in {validation_time:.3f}s"
        )

    def test_large_market_data_sanitization(self):
        """Test data sanitization performance with large market data."""
        # Generate large market dataset
        num_rows = 50000  # 50k rows
        dates = pd.date_range(start="2020-01-01", periods=num_rows, freq="1min")

        # Generate realistic OHLC data with trends and volatility
        np.random.seed(42)
        trend = np.sin(np.linspace(0, 20 * np.pi, num_rows)) * 10 + 100
        noise = np.random.normal(0, 2, num_rows)
        close_prices = trend + noise

        large_data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": close_prices + np.random.normal(0, 0.5, num_rows),
                "high": close_prices + abs(np.random.normal(0, 1, num_rows)),
                "low": close_prices - abs(np.random.normal(0, 1, num_rows)),
                "close": close_prices,
                "volume": np.random.normal(10000, 2000, num_rows),
            }
        )

        # Ensure OHLC integrity
        large_data["high"] = np.maximum(large_data["high"], large_data["close"])
        large_data["low"] = np.minimum(large_data["low"], large_data["close"])
        large_data["low"] = np.maximum(large_data["low"], 0)

        # Add some NaN values to test robustness
        nan_indices = np.random.choice(
            num_rows, size=int(num_rows * 0.05), replace=False
        )
        large_data.loc[nan_indices, "close"] = np.nan

        # Measure sanitization time
        start_time = time.time()
        sanitized_data, report = self.data_sanitizer.sanitize_market_data(large_data)
        end_time = time.time()

        sanitization_time = end_time - start_time

        # Performance assertions
        self.assertIsInstance(sanitized_data, pd.DataFrame)
        self.assertEqual(len(sanitized_data), num_rows)
        self.assertLess(sanitization_time, 30.0)  # Should complete within 30 seconds
        self.assertIn("issues_found", report)

        print(f"Sanitized {num_rows} rows in {sanitization_time:.3f}s")

    def test_regime_detection_large_dataset(self):
        """Test regime detection performance with large dataset."""
        # Generate large dataset
        num_rows = 25000
        dates = pd.date_range(start="2020-01-01", periods=num_rows, freq="5min")

        # Create trending data with regime changes
        np.random.seed(42)

        # First half: bullish trend
        trend1 = np.linspace(100, 150, num_rows // 2)
        # Second half: bearish trend
        trend2 = np.linspace(150, 120, num_rows // 2)
        trend = np.concatenate([trend1, trend2])

        noise = np.random.normal(0, 3, num_rows)
        close_prices = trend + noise

        large_data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": close_prices + np.random.normal(0, 1, num_rows),
                "high": close_prices + abs(np.random.normal(0, 2, num_rows)),
                "low": close_prices - abs(np.random.normal(0, 2, num_rows)),
                "close": close_prices,
                "volume": np.random.normal(10000, 3000, num_rows),
            }
        )

        # Measure regime detection time
        start_time = time.time()
        regime = self.regime_detector.detect_regime(large_data)
        end_time = time.time()

        detection_time = end_time - start_time

        # Performance assertions
        self.assertIsInstance(regime, dict)
        self.assertIn("regime", regime)
        self.assertLess(detection_time, 10.0)  # Should complete within 10 seconds

        print(f"Detected regime in {detection_time:.3f}s for {num_rows} data points")

    def test_performance_tracking_memory_usage(self):
        """Test performance tracking with large history."""
        # Record many performance entries
        num_entries = 5000
        base_time = pd.Timestamp.now() - pd.Timedelta(days=100)

        start_time = time.time()

        for i in range(num_entries):
            entry_time = base_time + pd.Timedelta(hours=i * 2)
            exit_time = entry_time + pd.Timedelta(hours=4)

            # Alternate profitable/unprofitable trades
            exit_price = 100 * (1.02 if i % 3 != 0 else 0.98)

            self.performance_tracker.record_signal_performance(
                f"signal_{i}", 100.0, exit_price, entry_time, exit_time, "fibonacci"
            )

        recording_time = time.time() - start_time

        # Verify history management
        self.assertLessEqual(len(self.performance_tracker.performance_history), 5000)

        # Test metrics calculation
        metrics_start = time.time()
        metrics = self.performance_tracker.get_performance_metrics()
        metrics_time = time.time() - metrics_start

        # Performance assertions
        self.assertLess(recording_time, 60.0)  # Should complete within 1 minute
        self.assertLess(metrics_time, 2.0)  # Metrics should calculate quickly
        self.assertIsInstance(metrics, dict)

        print(f"Recorded {num_entries} performance entries in {recording_time:.3f}s")
        print(f"Calculated metrics in {metrics_time:.3f}s")

    def test_concurrent_signal_processing_stress(self):
        """Test stress handling of concurrent signal processing."""
        # Simulate concurrent processing load
        num_iterations = 100
        signals_per_iteration = 50

        total_signals_processed = 0
        total_time = 0

        for iteration in range(num_iterations):
            # Generate batch of signals
            signals = []
            for i in range(signals_per_iteration):
                signal = MockActionSignal(
                    action="BUY" if (iteration + i) % 2 == 0 else "SELL",
                    confidence=0.6 + (i % 40) / 100,
                    price=100 + (i % 50),
                )
                signals.append(signal)

            # Process batch
            start_time = time.time()
            processed_count = 0

            for signal in signals:
                result = self.signal_validator.validate_signal(signal)
                if result.is_valid:
                    processed_count += 1

            batch_time = time.time() - start_time
            total_time += batch_time
            total_signals_processed += processed_count

        avg_time_per_batch = total_time / num_iterations
        avg_time_per_signal = total_time / total_signals_processed

        # Performance assertions
        self.assertLess(avg_time_per_batch, 0.5)  # Average batch under 0.5s
        self.assertLess(avg_time_per_signal, 0.005)  # Average signal under 5ms
        self.assertGreater(
            total_signals_processed, num_iterations * signals_per_iteration * 0.8
        )  # 80% success rate

        print(f"Processed {total_signals_processed} signals in {total_time:.3f}s")
        print(f"Average time per batch: {avg_time_per_batch:.3f}s")
        print(f"Average time per signal: {avg_time_per_signal:.6f}s")

    def test_memory_efficiency_large_history(self):
        """Test memory efficiency with large performance history."""
        # Test automatic cleanup
        initial_history_size = len(self.performance_tracker.performance_history)

        # Add old entries (beyond 30-day retention)
        old_base_time = pd.Timestamp.now() - pd.Timedelta(days=60)
        num_old_entries = 2000

        for i in range(num_old_entries):
            entry_time = old_base_time + pd.Timedelta(hours=i)
            exit_time = entry_time + pd.Timedelta(hours=2)

            self.performance_tracker.record_signal_performance(
                f"old_signal_{i}", 100.0, 102.0, entry_time, exit_time, "test"
            )

        # Add recent entries
        recent_base_time = pd.Timestamp.now() - pd.Timedelta(hours=1)
        num_recent_entries = 100

        for i in range(num_recent_entries):
            entry_time = recent_base_time + pd.Timedelta(minutes=i * 5)
            exit_time = entry_time + pd.Timedelta(hours=1)

            self.performance_tracker.record_signal_performance(
                f"recent_signal_{i}", 100.0, 102.0, entry_time, exit_time, "test"
            )

        final_history_size = len(self.performance_tracker.performance_history)

        # Verify cleanup worked
        self.assertLess(final_history_size, num_old_entries + num_recent_entries)
        self.assertGreaterEqual(
            final_history_size, num_recent_entries * 0.9
        )  # Keep most recent

        print(f"History size: {initial_history_size} -> {final_history_size}")
        print(
            f"Cleanup removed {num_old_entries + num_recent_entries - final_history_size} old entries"
        )

    def test_extreme_market_conditions_stress(self):
        """Test handling of extreme market conditions."""
        # Test with extreme volatility
        num_points = 1000
        dates = pd.date_range(start="2024-01-01", periods=num_points, freq="1min")

        # Generate extreme volatility data
        np.random.seed(42)
        base_price = 100
        prices = [base_price]

        for i in range(num_points - 1):
            # Extreme volatility: ±50% daily moves
            change = np.random.normal(0, 0.1)  # 10% volatility per minute
            new_price = prices[-1] * (1 + change)
            # Prevent extreme outliers
            new_price = np.clip(new_price, 0.01, 10000)
            prices.append(new_price)

        extreme_data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.05))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.05))) for p in prices],
                "close": prices,
                "volume": [np.random.normal(10000, 5000) for _ in range(num_points)],
            }
        )

        # Test all components with extreme data
        start_time = time.time()

        # Data sanitization
        sanitized_data, report = self.data_sanitizer.sanitize_market_data(extreme_data)

        # Regime detection
        regime = self.regime_detector.detect_regime(sanitized_data)

        # Market analysis
        market_conditions = self.market_analyzer.analyze_market_conditions(
            sanitized_data
        )

        # SAC validation with extreme data
        signals = [MockActionSignal(action="BUY", confidence=0.8)]
        sac_decisions = {"action": "BUY", "confidence": 0.7}
        validated_signals = self.sac_validator.validate_with_sac(
            signals, sac_decisions, sanitized_data
        )

        # Decision integration
        final_decision = self.decision_integrator.integrate_decisions(
            validated_signals, sac_decisions, sanitized_data
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify all components handled extreme data
        self.assertIsInstance(sanitized_data, pd.DataFrame)
        self.assertIsInstance(regime, dict)
        self.assertIsInstance(market_conditions, dict)
        self.assertIsInstance(final_decision, dict)
        self.assertLess(
            processing_time, 10.0
        )  # Should handle extreme data reasonably fast

        print(f"Processed extreme market data in {processing_time:.3f}s")


if __name__ == "__main__":
    unittest.main()
