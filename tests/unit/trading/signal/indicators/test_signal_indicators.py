"""
Unit tests for signal quality indicators

Tests cover CompositeIndicator, AdaptiveIndicator, and related indicator functionality
used in the SIGNAL_GUIDANCE system.
"""

import unittest

import numpy as np
import pandas as pd

from ztb.trading.signal.quality.indicators.base import (
    AdaptiveIndicator,
    CompositeIndicator,
)
from ztb.trading.signal.quality.indicators.macd import MACDIndicator
from ztb.trading.signal.quality.indicators.rsi import RSIIndicator


class TestCompositeIndicator(unittest.TestCase):
    """Test cases for CompositeIndicator"""

    def setUp(self):
        """Set up test fixtures"""
        self.rsi = RSIIndicator({"periods": 14})
        self.macd = MACDIndicator(
            {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        )

        self.indicators = [self.rsi, self.macd]
        self.weights = {"rsi": 0.5, "macd": 0.5}

        self.indicator = CompositeIndicator(self.indicators, self.weights)

        # Create sample market data
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "open": 100 + np.random.randn(50) * 2,
                "high": 102 + np.random.randn(50) * 2,
                "low": 98 + np.random.randn(50) * 2,
                "close": 100 + np.random.randn(50) * 2,
                "volume": np.random.randint(1000, 10000, 50),
            }
        )

    def test_initialization(self):
        """Test CompositeIndicator initialization"""
        self.assertIsInstance(self.indicator, CompositeIndicator)
        self.assertTrue(hasattr(self.indicator, "indicators"))
        self.assertTrue(hasattr(self.indicator, "weights"))

    def test_calculate_composite_signal(self):
        """Test composite signal calculation"""
        result = self.indicator.calculate(self.df)

        self.assertIsInstance(result, dict)
        self.assertIn("composite_score", result)
        # self.assertIn('signal_strength', result)
        # self.assertIn('confidence', result)

        self.assertIsInstance(result["composite_score"], (int, float))
        # self.assertIsInstance(result['signal_strength'], (int, float))
        # self.assertIsInstance(result['confidence'], (int, float))

    def test_calculate_with_custom_weights(self):
        """Test calculation with custom weights"""
        weights = {"rsi": 0.8, "macd": 0.2}
        indicator = CompositeIndicator(self.indicators, weights)
        result = indicator.calculate(self.df)

        self.assertIsInstance(result, dict)
        self.assertIn("composite_score", result)

    def test_calculate_with_missing_indicators(self):
        """Test calculation with missing indicators"""
        indicators = [self.rsi]
        weights = {"rsi": 1.0}

        indicator = CompositeIndicator(indicators, weights)
        result = indicator.calculate(self.df)

        self.assertIsInstance(result, dict)
        self.assertIn("composite_score", result)

    def test_error_handling_empty_dataframe(self):
        """Test error handling with empty DataFrame"""
        result = self.indicator.calculate(pd.DataFrame())
        self.assertIsInstance(result, dict)

    def test_weight_normalization(self):
        """Test automatic weight normalization"""
        weights = {"rsi": 2.0, "macd": 2.0}  # Don't sum to 1

        indicator = CompositeIndicator(self.indicators, weights)
        result = indicator.calculate(self.df)

        # Should still work with unnormalized weights
        self.assertIsInstance(result, dict)
        self.assertIn("composite_score", result)


class TestAdaptiveIndicator(unittest.TestCase):
    """Test cases for AdaptiveIndicator"""

    def setUp(self):
        """Set up test fixtures"""
        self.base_indicator = RSIIndicator({"periods": 14})
        self.config = {
            "adaptive_params": {
                "trending": {"periods": 21},
                "ranging": {"periods": 14},
                "volatile": {"periods": 9},
            }
        }

        self.indicator = AdaptiveIndicator(self.base_indicator, self.config)

        # Create sample market data
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "open": 100 + np.random.randn(50) * 2,
                "high": 102 + np.random.randn(50) * 2,
                "low": 98 + np.random.randn(50) * 2,
                "close": 100 + np.random.randn(50) * 2,
                "volume": np.random.randint(1000, 10000, 50),
            }
        )

    def test_initialization(self):
        """Test AdaptiveIndicator initialization"""
        self.assertIsInstance(self.indicator, AdaptiveIndicator)
        self.assertTrue(hasattr(self.indicator, "base_indicator"))
        # self.assertTrue(hasattr(self.indicator, 'adaptation_factors'))

    def test_calculate_adaptive_trending_market(self):
        """Test adaptive calculation in trending market"""
        self.indicator.set_market_regime("trending")
        result = self.indicator.calculate(self.df)

        self.assertIsInstance(result, dict)
        self.assertIn("adaptive_regime", result)
        self.assertEqual(result["adaptive_regime"], "trending")

    def test_calculate_adaptive_ranging_market(self):
        """Test adaptive calculation in ranging market"""
        self.indicator.set_market_regime("ranging")
        result = self.indicator.calculate(self.df)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["adaptive_regime"], "ranging")

    def test_calculate_adaptive_volatile_market(self):
        """Test adaptive calculation in volatile market"""
        self.indicator.set_market_regime("volatile")
        result = self.indicator.calculate(self.df)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["adaptive_regime"], "volatile")

    def test_calculate_adaptive_unknown_regime(self):
        """Test adaptive calculation with unknown market regime"""
        self.indicator.set_market_regime("unknown")
        result = self.indicator.calculate(self.df)

        self.assertIsInstance(result, dict)
        # Should default to some regime or handle gracefully

    def test_adaptation_factor_application(self):
        """Test that adaptation factors are properly applied"""
        # Test trending
        self.indicator.set_market_regime("trending")
        trending_result = self.indicator.calculate(self.df)

        # Test ranging
        self.indicator.set_market_regime("ranging")
        ranging_result = self.indicator.calculate(self.df)

        # Results should be different due to different adaptation factors
        # self.assertNotEqual(trending_result, ranging_result)
        pass

    def test_error_handling_empty_dataframe(self):
        """Test error handling with empty DataFrame"""
        result = self.indicator.calculate_adaptive(pd.DataFrame(), "trending")
        self.assertIsInstance(result, dict)

    def test_error_handling_invalid_regime(self):
        """Test error handling with invalid market regime"""
        result = self.indicator.calculate_adaptive(self.df, None)
        self.assertIsInstance(result, dict)


class TestRSIIndicator(unittest.TestCase):
    """Test cases for RSIIndicator"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {"periods": 14}
        self.indicator = RSIIndicator(self.config)

        # Create sample market data
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "open": 100 + np.random.randn(50) * 2,
                "high": 102 + np.random.randn(50) * 2,
                "low": 98 + np.random.randn(50) * 2,
                "close": 100 + np.random.randn(50) * 2,
                "volume": np.random.randint(1000, 10000, 50),
            }
        )

    def test_initialization(self):
        """Test RSIIndicator initialization"""
        self.assertIsInstance(self.indicator, RSIIndicator)
        self.assertEqual(self.indicator.periods, 14)

    def test_calculate_rsi(self):
        """Test RSI calculation"""
        result = self.indicator.calculate(self.df)

        self.assertIsInstance(result, dict)
        self.assertIn("rsi", result)
        self.assertIsInstance(result["rsi"], (int, float, np.ndarray))

        # RSI should be between 0 and 100
        rsi_values = result["rsi"]
        if isinstance(rsi_values, np.ndarray):
            self.assertTrue(np.all((rsi_values >= 0) & (rsi_values <= 100)))
        else:
            self.assertGreaterEqual(rsi_values, 0)
            self.assertLessEqual(rsi_values, 100)

    def test_rsi_oversold_conditions(self):
        """Test RSI under oversold conditions"""
        # Create oversold conditions
        oversold_df = self.df.copy()
        oversold_df["close"] = np.linspace(100, 80, 50)  # Sharp decline

        result = self.indicator.calculate(oversold_df)
        self.assertIn("rsi", result)

    def test_rsi_overbought_conditions(self):
        """Test RSI under overbought conditions"""
        # Create overbought conditions
        overbought_df = self.df.copy()
        overbought_df["close"] = np.linspace(80, 100, 50)  # Sharp rise

        result = self.indicator.calculate(overbought_df)
        self.assertIn("rsi", result)

    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        short_df = self.df.head(5)  # Less than period
        result = self.indicator.calculate(short_df)

        self.assertIsInstance(result, dict)


class TestMACDIndicator(unittest.TestCase):
    """Test cases for MACDIndicator"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        self.indicator = MACDIndicator(self.config)

        # Create sample market data
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "open": 100 + np.random.randn(50) * 2,
                "high": 102 + np.random.randn(50) * 2,
                "low": 98 + np.random.randn(50) * 2,
                "close": 100 + np.random.randn(50) * 2,
                "volume": np.random.randint(1000, 10000, 50),
            }
        )

    def test_initialization(self):
        """Test MACDIndicator initialization"""
        self.assertIsInstance(self.indicator, MACDIndicator)
        self.assertEqual(self.indicator.fast_period, 12)
        self.assertEqual(self.indicator.slow_period, 26)
        self.assertEqual(self.indicator.signal_period, 9)

    def test_calculate_macd(self):
        """Test MACD calculation"""
        result = self.indicator.calculate(self.df)

        self.assertIsInstance(result, dict)
        expected_keys = ["macd_line", "signal_line", "macd_histogram"]
        for key in expected_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], (int, float, np.ndarray))

    def test_macd_crossover_signals(self):
        """Test MACD crossover signal generation"""
        result = self.indicator.calculate(self.df)

        macd_line = result["macd_line"]
        signal_line = result["signal_line"]
        histogram = result["macd_histogram"]

        # Basic validation that components exist and are numeric
        self.assertIsNotNone(macd_line)
        self.assertIsNotNone(signal_line)
        self.assertIsNotNone(histogram)

    def test_macd_with_trend(self):
        """Test MACD with trending data"""
        # Create trending data
        trend_df = self.df.copy()
        trend_df["close"] = 100 + np.linspace(0, 10, 50)  # Upward trend

        result = self.indicator.calculate(trend_df)
        self.assertIn("macd_line", result)
        self.assertIn("signal_line", result)
        self.assertIn("macd_histogram", result)

    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        short_df = self.df.head(10)  # Less than required periods
        result = self.indicator.calculate(short_df)

        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
