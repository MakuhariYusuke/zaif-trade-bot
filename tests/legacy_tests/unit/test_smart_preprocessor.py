#!/usr/bin/env python3
"""
Unit tests for SmartPreprocessor
"""

import unittest

import numpy as np
import pandas as pd

from ztb.evaluation.preprocess import SmartPreprocessor


class TestSmartPreprocessor(unittest.TestCase):
    """Test cases for SmartPreprocessor"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample OHLC data (without CommonPreprocessor.preprocess)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        np.random.seed(42)
        self.sample_data = pd.DataFrame(
            {
                "open": 100 + np.random.randn(100).cumsum(),
                "high": 105 + np.random.randn(100).cumsum(),
                "low": 95 + np.random.randn(100).cumsum(),
                "close": 100 + np.random.randn(100).cumsum(),
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

    def test_preprocess_ema_only(self):
        """Test preprocessing with only EMA requirements"""
        required = ["ema:12", "ema:26"]
        preprocessor = SmartPreprocessor(set(required))
        result = preprocessor.preprocess(self.sample_data)

        # Should contain required EMA columns
        self.assertIn("ema_12", result.columns)
        self.assertIn("ema_26", result.columns)

        # Should not contain other EMA columns
        self.assertNotIn("ema_5", result.columns)
        self.assertNotIn("ema_50", result.columns)

        # Should contain basic columns
        self.assertIn("close", result.columns)
        self.assertIn("open", result.columns)

    def test_preprocess_rolling_only(self):
        """Test preprocessing with only rolling requirements"""
        required = ["rolling_mean:10", "rolling_std:20"]
        preprocessor = SmartPreprocessor(set(required))
        result = preprocessor.preprocess(self.sample_data)

        # Should contain required rolling columns
        self.assertIn("rolling_mean_10", result.columns)
        self.assertIn("rolling_std_20", result.columns)

        # Should not contain other rolling columns
        self.assertNotIn("rolling_mean_5", result.columns)
        self.assertNotIn("rolling_std_30", result.columns)

    def test_preprocess_mixed_requirements(self):
        """Test preprocessing with mixed requirements"""
        required = ["ema:12", "rolling_mean:20", "close"]
        preprocessor = SmartPreprocessor(set(required))
        result = preprocessor.preprocess(self.sample_data)

        # Should contain all required columns
        self.assertIn("ema_12", result.columns)
        self.assertIn("rolling_mean_20", result.columns)
        self.assertIn("close", result.columns)

        # Should not contain non-required columns
        self.assertNotIn("ema_26", result.columns)
        self.assertNotIn("rolling_std_20", result.columns)

    def test_preprocess_empty_requirements(self):
        """Test preprocessing with empty requirements"""
        required = []
        preprocessor = SmartPreprocessor(set(required))
        result = preprocessor.preprocess(self.sample_data)

        # Should still include basic columns
        self.assertIn("close", result.columns)
        self.assertIn("open", result.columns)
        self.assertIn("high", result.columns)
        self.assertIn("low", result.columns)
        self.assertIn("volume", result.columns)

    def test_preprocess_caching(self):
        """Test that preprocessing caches calculations"""
        required = ["ema:12", "ema:26"]
        preprocessor = SmartPreprocessor(set(required))

        # First call
        result1 = preprocessor.preprocess(self.sample_data)

        # Second call with same requirements should use cache
        result2 = preprocessor.preprocess(self.sample_data)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_preprocess_invalid_requirement(self):
        """Test preprocessing with invalid requirement format"""
        required = ["invalid_format", "ema:12"]
        preprocessor = SmartPreprocessor(set(required))
        result = preprocessor.preprocess(self.sample_data)

        # Should still process valid requirements
        self.assertIn("ema_12", result.columns)

        # Should contain basic columns
        self.assertIn("close", result.columns)


if __name__ == "__main__":
    unittest.main()
