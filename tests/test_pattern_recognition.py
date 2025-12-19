#!/usr/bin/env python3
"""
Unit tests for Pattern Recognition Components.

This module contains comprehensive unit tests for the pattern recognition
components including Fibonacci patterns, technical indicators, and pattern analysis.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.strategies.action_signal_guide.pattern_recognition.fibonacci_patterns import (
    FibonacciAnalyzer,
    FibonacciRetracementRecognizer,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import SignalResult
from ztb.trading.strategies.action_signal_guide.pattern_recognition.rsi import RSIPatternRecognizer
from ztb.trading.strategies.action_signal_guide.pattern_recognition.macd import MACDPatternRecognizer
from ztb.trading.strategies.action_signal_guide.pattern_recognition.bollinger_patterns import BollingerBandsRecognizer


class TestFibonacciAnalyzer(unittest.TestCase):
    """Test cases for FibonacciAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FibonacciAnalyzer()

    def test_calculate_retracement_levels(self):
        """Test Fibonacci retracement level calculation."""
        high = 120.0
        low = 100.0

        levels = FibonacciAnalyzer.calculate_retracement_levels(high, low)

        self.assertIsInstance(levels, dict)
        self.assertIn(0.236, levels)
        self.assertIn(0.382, levels)
        self.assertIn(0.618, levels)

        # Verify calculations
        expected_236 = 100 + (120 - 100) * 0.236
        expected_618 = 100 + (120 - 100) * 0.618

        self.assertAlmostEqual(levels[0.236], expected_236, places=5)
        self.assertAlmostEqual(levels[0.618], expected_618, places=5)

    def test_calculate_extension_levels_bullish(self):
        """Test bullish Fibonacci extension level calculation."""
        high = 120.0
        low = 100.0

        levels = FibonacciAnalyzer.calculate_extension_levels(high, low, direction=1)

        self.assertIsInstance(levels, dict)
        self.assertIn(0.618, levels)
        self.assertIn(1.618, levels)

        # Verify bullish extension from low
        expected_618 = 100 + (120 - 100) * 0.618
        expected_1618 = 100 + (120 - 100) * 1.618

        self.assertAlmostEqual(levels[0.618], expected_618, places=5)
        self.assertAlmostEqual(levels[1.618], expected_1618, places=5)

    def test_calculate_extension_levels_bearish(self):
        """Test bearish Fibonacci extension level calculation."""
        high = 120.0
        low = 100.0

        levels = FibonacciAnalyzer.calculate_extension_levels(high, low, direction=-1)

        self.assertIsInstance(levels, dict)

        # Verify bearish extension from high
        expected_618 = 120 - (120 - 100) * 0.618
        expected_1618 = 120 - (120 - 100) * 1.618

        self.assertAlmostEqual(levels[0.618], expected_618, places=5)
        self.assertAlmostEqual(levels[1.618], expected_1618, places=5)

    def test_calculate_deviation_from_ideal(self):
        """Test deviation calculation from ideal Fibonacci levels."""
        current_price = 110.0
        levels = {0.382: 108.0, 0.5: 110.0, 0.618: 112.0}

        deviation = FibonacciAnalyzer.calculate_deviation_from_ideal(current_price, levels)

        self.assertIsInstance(deviation, dict)
        self.assertIn("level", deviation)
        self.assertIn("deviation_pct", deviation)
        self.assertIn("ideal_price", deviation)

        # Current price 110 should match 0.5 level exactly
        self.assertEqual(deviation["level"], 0.5)
        self.assertAlmostEqual(deviation["deviation_pct"], 0.0, places=5)

    def test_find_support_resistance_levels(self):
        """Test support and resistance level identification."""
        prices = pd.Series([100, 105, 110, 108, 112, 115, 113, 118])

        levels = FibonacciAnalyzer.find_support_resistance_levels(prices)

        self.assertIsInstance(levels, dict)
        self.assertIn("support", levels)
        self.assertIn("resistance", levels)
        self.assertIsInstance(levels["support"], list)
        self.assertIsInstance(levels["resistance"], list)


class TestFibonacciRetracementRecognizer(unittest.TestCase):
    """Test cases for FibonacciRetracementRecognizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.pattern = FibonacciRetracementRecognizer()

    def test_initialization(self):
        """Test pattern initialization."""
        self.assertIsInstance(self.pattern, FibonacciRetracementRecognizer)
        self.assertIn("retracement_threshold", self.pattern.thresholds)

    def test_fibonacci_boundary_cases(self):
        """Test Fibonacci retracement pattern recognition with boundary cases."""
        # Test with insufficient data
        small_data = pd.DataFrame({
            "open": [100, 101],
            "high": [105, 106],
            "low": [95, 96],
            "close": [102, 103],
            "volume": [1000, 1100]
        })
        result = self.pattern.recognize(small_data)
        self.assertIsNone(result)  # Should return None for insufficient data

        # Test with normal data
        data = pd.DataFrame({
            "open": [100] * 50,
            "high": [105] * 50,
            "low": [95] * 50,
            "close": [102] * 50,
            "volume": [1000] * 50
        })
        result = self.pattern.recognize(data)
        # Result may be None or SignalResult depending on conditions
        if result is not None:
            self.assertIsInstance(result, SignalResult)


class TestRSIPatternRecognizer(unittest.TestCase):
    """Test cases for RSIPatternRecognizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.rsi = RSIPatternRecognizer()

    def test_initialization(self):
        """Test RSI pattern recognizer initialization."""
        self.assertIsInstance(self.rsi, RSIPatternRecognizer)

    def test_rsi_boundary_cases(self):
        """Test RSI pattern recognition with boundary cases."""
        # Test with insufficient data
        small_data = pd.DataFrame({
            "open": [100, 101],
            "high": [105, 106],
            "low": [95, 96],
            "close": [102, 103],
            "volume": [1000, 1100]
        })
        result = self.rsi.recognize(small_data)
        self.assertIsNone(result)  # Should return None for insufficient data

        # Test with normal data
        data = pd.DataFrame({
            "open": [100] * 50,
            "high": [105] * 50,
            "low": [95] * 50,
            "close": [102] * 50,
            "volume": [1000] * 50
        })
        result = self.rsi.recognize(data)
        # Result may be None or SignalResult depending on conditions
        if result is not None:
            self.assertIsInstance(result, SignalResult)


class TestMACDPatternRecognizer(unittest.TestCase):
    """Test cases for MACDPatternRecognizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.macd = MACDPatternRecognizer()

    def test_initialization(self):
        """Test MACD pattern recognizer initialization."""
        self.assertIsInstance(self.macd, MACDPatternRecognizer)
        self.assertEqual(self.macd.fast_period, 12)
        self.assertEqual(self.macd.slow_period, 26)
        self.assertEqual(self.macd.signal_period, 9)

    def test_calculate_macd_normal_data(self):
        """Test MACD calculation with normal market data."""
        # Create trending data
        np.random.seed(42)
        prices = []
        price = 100
        for i in range(100):
            change = np.random.normal(0, 1)
            price += change
            prices.append(price)

        data = pd.DataFrame({"close": prices})
        macd_result = self.macd.calculate(data)

        self.assertIsInstance(macd_result, dict)
        self.assertIn("macd", macd_result)
        self.assertIn("signal", macd_result)
        self.assertIn("histogram", macd_result)

        # Check lengths
        self.assertEqual(len(macd_result["macd"]), len(data))
        self.assertEqual(len(macd_result["signal"]), len(data))
        self.assertEqual(len(macd_result["histogram"]), len(data))

    def test_calculate_macd_insufficient_data(self):
        """Test MACD calculation with insufficient data."""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})  # Less than slow period

        macd_result = self.macd.calculate(data)

        self.assertIsInstance(macd_result, dict)
        # Should return empty or NaN results
        self.assertEqual(len(macd_result["macd"]), len(data))

    def test_macd_boundary_cases(self):
        """Test MACD pattern recognition with boundary cases."""
        # Test with insufficient data
        small_data = pd.DataFrame({
            "open": [100, 101],
            "high": [105, 106],
            "low": [95, 96],
            "close": [102, 103],
            "volume": [1000, 1100]
        })
        result = self.macd.recognize(small_data)
        self.assertIsNone(result)  # Should return None for insufficient data

        # Test with normal data
        data = pd.DataFrame({
            "open": [100] * 50,
            "high": [105] * 50,
            "low": [95] * 50,
            "close": [102] * 50,
            "volume": [1000] * 50
        })
        result = self.macd.recognize(data)
        # Result may be None or SignalResult depending on conditions
        if result is not None:
            self.assertIsInstance(result, SignalResult)


class TestBollingerBandsRecognizer(unittest.TestCase):
    """Test cases for BollingerBandsRecognizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.pattern = BollingerBandsRecognizer()

    def test_initialization(self):
        """Test Bollinger Bands recognizer initialization."""
        self.assertIsInstance(self.pattern, BollingerBandsRecognizer)

    def test_bollinger_boundary_cases(self):
        """Test Bollinger Bands pattern recognition with boundary cases."""
        # Test with insufficient data
        small_data = pd.DataFrame({
            "open": [100, 101],
            "high": [105, 106],
            "low": [95, 96],
            "close": [102, 103],
            "volume": [1000, 1100]
        })
        result = self.pattern.recognize(small_data)
        self.assertIsNone(result)  # Should return None for insufficient data

        # Test with normal data
        data = pd.DataFrame({
            "open": [100] * 50,
            "high": [105] * 50,
            "low": [95] * 50,
            "close": [102] * 50,
            "volume": [1000] * 50
        })
        result = self.pattern.recognize(data)
        # Result may be None or SignalResult depending on conditions
        if result is not None:
            self.assertIsInstance(result, SignalResult)


if __name__ == "__main__":
    unittest.main()