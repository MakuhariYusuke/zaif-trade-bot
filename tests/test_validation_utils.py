#!/usr/bin/env python3
"""
Unit tests for Validation Utility Components.

This module contains comprehensive unit tests for the validation utility components
including SignalValidator, DataSanitizer, and PerformanceTracker.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.strategies.action_signal_guide.components.validation import (
    SignalValidator,
    DataSanitizer,
    PerformanceTracker,
    ValidationResult,
)




class TestSignalValidator(unittest.TestCase):
    def setUp(self):
        self.validator = SignalValidator()

    def test_validator_initialization(self):
        self.assertIn("required_fields", self.validator.validation_rules)
        self.assertIn("data_types", self.validator.validation_rules)

    def test_validate_signal_valid(self):
        """Test validation of a valid signal."""
        signal = MockActionSignal(
            action="BUY",
            confidence=0.8,
            timestamp=pd.Timestamp.now(),
            pattern_type="fibonacci",
            price=100.0,
            stop_loss=95.0,
            take_profit=110.0
        )

        result = self.validator.validate_signal(signal)

        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertGreater(result.confidence_score, 0.9)

    def test_validate_signal_missing_fields(self):
        """Test validation of signal with missing required fields."""
        # Create signal missing required fields
        signal = MockActionSignal()
        delattr(signal, 'action')  # Remove required field

        result = self.validator.validate_signal(signal)

        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)
        self.assertIn("Missing required field: action", result.issues)

    def test_validate_signal_invalid_data_types(self):
        """Test validation of signal with invalid data types."""
        signal = MockActionSignal()
        signal.confidence = "invalid"  # Should be numeric

        result = self.validator.validate_signal(signal)

        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.issues), 0)

    def test_validate_signal_invalid_values(self):
        """Test validation of signal with invalid values."""
        signal = MockActionSignal(confidence=1.5)  # Confidence > 1.0

        result = self.validator.validate_signal(signal)

        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.issues), 0)

    def test_validate_signal_boundary_values(self):
        """Test validation with boundary values."""
        # Test with zero confidence
        signal = MockActionSignal(confidence=0.0)
        result = self.validator.validate_signal(signal)
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)  # Zero confidence should be valid

        # Test with maximum confidence
        signal = MockActionSignal(confidence=1.0)
        result = self.validator.validate_signal(signal)
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)

        # Test with negative confidence (invalid)
        signal = MockActionSignal(confidence=-0.1)
        result = self.validator.validate_signal(signal)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)

        # Test with confidence > 1.0 (invalid)
        signal = MockActionSignal(confidence=1.5)
        result = self.validator.validate_signal(signal)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)

    def test_validate_signal_extreme_prices(self):
        """Test validation with extreme price values."""
        # Test with zero price
        signal = MockActionSignal(price=0.0)
        result = self.validator.validate_signal(signal)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)  # Zero price should be invalid

        # Test with negative price
        signal = MockActionSignal(price=-100.0)
        result = self.validator.validate_signal(signal)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)

        # Test with very large price
        signal = MockActionSignal(price=1e10)
        result = self.validator.validate_signal(signal)
        self.assertIsInstance(result, ValidationResult)
        # Large price should be valid (no upper bound in validation)
        self.assertTrue(result.is_valid)

        # Test with very small positive price
        signal = MockActionSignal(price=1e-6)
        result = self.validator.validate_signal(signal)
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)

    def test_validate_signal_boundary_stop_loss_take_profit(self):
        """Test validation with boundary stop loss and take profit values."""
        # Test with stop loss equal to price (invalid for BUY)
        signal = MockActionSignal(action="BUY", price=100.0, stop_loss=100.0)
        result = self.validator.validate_signal(signal)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)

        # Test with take profit equal to price (invalid for BUY)
        signal = MockActionSignal(action="BUY", price=100.0, take_profit=100.0)
        result = self.validator.validate_signal(signal)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)

        # Test with stop loss above price for BUY (invalid)
        signal = MockActionSignal(action="BUY", price=100.0, stop_loss=105.0)
        result = self.validator.validate_signal(signal)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)

        # Test with take profit below price for BUY (invalid)
        signal = MockActionSignal(action="BUY", price=100.0, take_profit=95.0)
        result = self.validator.validate_signal(signal)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)

    def test_validate_logical_consistency_buy_signal(self):
        """Test logical consistency for BUY signals."""
        signal = MockActionSignal(
            action="BUY",
            price=100.0,
            stop_loss=105.0,  # Stop loss above price (invalid for BUY)
            take_profit=95.0   # Take profit below price (invalid for BUY)
        )

        result = self.validator._validate_logical_consistency(signal)

        self.assertFalse(result["passed"])
        self.assertGreater(len(result["issues"]), 0)

    def test_validate_logical_consistency_sell_signal(self):
        """Test logical consistency for SELL signals."""
        signal = MockActionSignal(
            action="SELL",
            price=100.0,
            stop_loss=95.0,   # Stop loss below price (invalid for SELL)
            take_profit=105.0  # Take profit above price (invalid for SELL)
        )

        result = self.validator._validate_logical_consistency(signal)

        self.assertFalse(result["passed"])
        self.assertGreater(len(result["issues"]), 0)


class TestDataSanitizer(unittest.TestCase):
    """Test cases for DataSanitizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.sanitizer = DataSanitizer()

    def test_initialization(self):
        """Test sanitizer initialization."""
        self.assertIsInstance(self.sanitizer, DataSanitizer)
        self.assertIn("remove_outliers", self.sanitizer.sanitization_rules)

    def test_sanitize_market_data_normal(self):
        """Test sanitization of normal market data."""
        data = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [95, 96, 97],
            "close": [103, 104, 105],
            "volume": [1000, 1100, 1200]
        })

        sanitized_data, report = self.sanitizer.sanitize_market_data(data)

        self.assertIsInstance(sanitized_data, pd.DataFrame)
        self.assertIsInstance(report, dict)
        self.assertIn("original_rows", report)
        self.assertIn("final_rows", report)
        self.assertEqual(report["original_rows"], report["final_rows"])  # No changes expected

    def test_remove_outliers(self):
        """Test outlier removal."""
        # Create data with outliers
        normal_prices = [100, 101, 102, 103, 104]
        outlier_prices = [100, 101, 1000, 103, 104]  # 1000 is outlier

        data = pd.DataFrame({"close": outlier_prices})

        result = self.sanitizer._remove_outliers(data)

        self.assertIn("data", result)
        self.assertIn("issues", result)
        if result.get("issues"):
            self.assertIn("outliers", result["issues"][0].lower())

    def test_fill_missing_values(self):
        """Test missing value filling."""
        # Create data with missing values
        data = pd.DataFrame({
            "close": [100, np.nan, 102, np.nan, 104],
            "volume": [1000, 1100, np.nan, 1300, 1400]
        })

        result = self.sanitizer._fill_missing_values(data)

        self.assertIn("data", result)
        # Check that NaN values are filled
        self.assertFalse(result["data"]["close"].isnull().any())
        self.assertFalse(result["data"]["volume"].isnull().any())

    def test_normalize_data_types(self):
        """Test data type normalization."""
        # Create data with wrong types
        data = pd.DataFrame({
            "close": ["100", "101", "102"],  # Strings instead of numbers
            "volume": [1000, 1100, 1200]
        })

        sanitized_data, report = self.sanitizer.sanitize_market_data(data)

        self.assertIsInstance(sanitized_data, pd.DataFrame)
        # Check that close column is now numeric after sanitization
        self.assertTrue(pd.api.types.is_numeric_dtype(sanitized_data["close"]))

    def test_validate_ohlc_consistency(self):
        """Test OHLC data consistency validation."""
        # Create inconsistent OHLC data
        data = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [95, 96, 97],  # High < Open (inconsistent)
            "low": [105, 106, 107],  # Low > Open (inconsistent)
            "close": [103, 104, 105]
        })

        result = self.sanitizer._validate_ohlc_consistency(data)

        self.assertIn("issues", result)
        self.assertGreater(len(result["issues"]), 0)
        self.assertIn("quality_penalty", result)

    def test_sanitize_market_data_boundary_cases(self):
        """Test sanitization with boundary data cases."""
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        sanitized_data, report = self.sanitizer.sanitize_market_data(empty_data)
        self.assertIsInstance(sanitized_data, pd.DataFrame)
        self.assertEqual(len(sanitized_data), 0)

        # Test with single row
        single_row_data = pd.DataFrame({
            "close": [100.0],
            "volume": [1000]
        })
        sanitized_data, report = self.sanitizer.sanitize_market_data(single_row_data)
        self.assertIsInstance(sanitized_data, pd.DataFrame)
        self.assertEqual(len(sanitized_data), 1)

        # Test with all NaN values
        nan_data = pd.DataFrame({
            "close": [np.nan, np.nan, np.nan],
            "volume": [np.nan, np.nan, np.nan]
        })
        sanitized_data, report = self.sanitizer.sanitize_market_data(nan_data)
        self.assertIsInstance(sanitized_data, pd.DataFrame)
        self.assertTrue(sanitized_data.isnull().any().any())  # Should still have NaNs if can't fill

        # Test with extreme values
        extreme_data = pd.DataFrame({
            "close": [1e-10, 1e10, 0.0, -100.0],
            "volume": [0, 1e15, -1000, np.nan]
        })
        sanitized_data, report = self.sanitizer.sanitize_market_data(extreme_data)
        self.assertIsInstance(sanitized_data, pd.DataFrame)
        self.assertIn("issues_found", report)


class TestPerformanceTracker(unittest.TestCase):
    """Test cases for PerformanceTracker."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = PerformanceTracker()

    def test_initialization(self):
        """Test tracker initialization."""
        self.assertIsInstance(self.tracker, PerformanceTracker)
        self.assertIsInstance(self.tracker.performance_history, list)

    def test_record_signal_performance_boundary_cases(self):
        """Test performance recording with boundary cases."""
        # Test with zero entry price (should handle gracefully or raise appropriate error)
        with self.assertRaises(ZeroDivisionError):
            self.tracker.record_signal_performance(
                "test_1", 0.0, 0.0, pd.Timestamp.now(), pd.Timestamp.now() + pd.Timedelta(hours=1), "test"
            )

        # Test with negative entry price (should handle gracefully)
        record = self.tracker.record_signal_performance(
            "test_2", -100.0, -90.0, pd.Timestamp.now(), pd.Timestamp.now() + pd.Timedelta(hours=1), "test"
        )
        self.assertIsInstance(record, dict)
        self.assertGreater(record["price_change_pct"], 0)  # -90 > -100, so positive change

        # Test with very large prices
        record = self.tracker.record_signal_performance(
            "test_3", 1e10, 1.1e10, pd.Timestamp.now(), pd.Timestamp.now() + pd.Timedelta(hours=1), "test"
        )
        self.assertIsInstance(record, dict)
        self.assertAlmostEqual(record["price_change_pct"], 0.1, places=5)

        # Test with zero holding time (should handle division by zero)
        now = pd.Timestamp.now()
        record = self.tracker.record_signal_performance(
            "test_4", 100.0, 105.0, now, now, "test"
        )
        self.assertIsInstance(record, dict)
        self.assertEqual(record["holding_time_hours"], 0.0)

        # Test with very long holding time
        record = self.tracker.record_signal_performance(
            "test_5", 100.0, 105.0, now, now + pd.Timedelta(days=365), "test"
        )
        self.assertIsInstance(record, dict)
        self.assertGreater(record["holding_time_hours"], 8000)  # Should be around 8760 hours

    def test_get_performance_metrics_no_data(self):
        """Test performance metrics with no data."""
        metrics = self.tracker.get_performance_metrics()

        self.assertIn("total_signals", metrics)
        self.assertEqual(metrics["total_signals"], 0)
        self.assertEqual(metrics["win_rate"], 0.0)

    def test_get_performance_metrics_with_data(self):
        """Test performance metrics with data."""
        # Record some performance data
        base_time = pd.Timestamp.now()

        for i in range(10):
            entry_time = base_time + pd.Timedelta(hours=i)
            exit_time = entry_time + pd.Timedelta(hours=1)
            # Alternate profitable and unprofitable trades
            exit_price = 100 * (1.02 if i % 2 == 0 else 0.98)

            self.tracker.record_signal_performance(
                f"signal_{i}", 100.0, exit_price, entry_time, exit_time, "fibonacci"
            )

        metrics = self.tracker.get_performance_metrics()

        self.assertEqual(metrics["total_signals"], 10)
        self.assertEqual(metrics["win_rate"], 0.5)  # 50% win rate
        self.assertIn("avg_return_pct", metrics)
        self.assertIn("volatility", metrics)

    def test_get_performance_metrics_filtered(self):
        """Test performance metrics with filtering."""
        base_time = pd.Timestamp.now()

        # Record signals with different patterns
        patterns = ["fibonacci", "harmonic", "bollinger"]
        for i, pattern in enumerate(patterns):
            entry_time = base_time + pd.Timedelta(hours=i)
            exit_time = entry_time + pd.Timedelta(hours=1)
            exit_price = 100 * 1.02  # All profitable

            self.tracker.record_signal_performance(
                f"signal_{i}", 100.0, exit_price, entry_time, exit_time, pattern
            )

        # Get metrics for fibonacci patterns only
        metrics = self.tracker.get_performance_metrics(pattern_type="fibonacci")

        self.assertEqual(metrics["total_signals"], 1)
        self.assertEqual(metrics["pattern_performance"]["fibonacci"], 0.02)

    def test_history_cleanup(self):
        """Test automatic history cleanup."""
        # Record many signals with old timestamps to trigger cleanup
        base_time = pd.Timestamp.now() - pd.Timedelta(days=60)  # Old data beyond 30-day limit

        for i in range(1200):  # More than the expected limit
            entry_time = base_time + pd.Timedelta(hours=i)
            exit_time = entry_time + pd.Timedelta(hours=1)

            # Manually set recorded_at to old time to trigger cleanup
            record = self.tracker.record_signal_performance(
                f"signal_{i}", 100.0, 102.0, entry_time, exit_time, "test"
            )
            # Override the recorded_at to be old
            record["recorded_at"] = base_time + pd.Timedelta(hours=i)
            # Re-add to history with old timestamp
            self.tracker.performance_history[-1]["recorded_at"] = base_time + pd.Timedelta(hours=i)

        # Should have cleaned up old records (older than 30 days)
        # Since we added old data, most should be cleaned up
        # But let's check that cleanup actually runs by calling record_signal_performance again
        self.tracker.record_signal_performance(
            "new_signal", 100.0, 102.0, pd.Timestamp.now(), pd.Timestamp.now() + pd.Timedelta(hours=1), "test"
        )

        # After adding a new record, old records should be cleaned up
        old_records = sum(1 for p in self.tracker.performance_history if p["recorded_at"] < pd.Timestamp.now() - pd.Timedelta(days=30))
        self.assertEqual(old_records, 0)


if __name__ == "__main__":
    unittest.main()