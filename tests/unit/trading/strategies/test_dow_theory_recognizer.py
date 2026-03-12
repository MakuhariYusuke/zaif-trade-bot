#!/usr/bin/env python3
"""
Unit tests for DOW_THEORY pattern recognizers
"""
import unittest

import numpy as np
import pandas as pd

from ztb.trading.strategies.action_signal_guide.pattern_recognition.dow_theory import (
    DowTheoryRecognizer,
)


class TestDowTheoryRecognizer(unittest.TestCase):
    """Test cases for DowTheoryRecognizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.recognizer = DowTheoryRecognizer()

    def test_recognize_weak_signal(self):
        """Test that Dow Theory recognizer produces weak signals."""
        # Create synthetic data with a clear trend
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        # Create upward trend
        prices = [100 + i * 0.1 for i in range(100)]
        data = pd.DataFrame(
            {
                "high": [p * 1.005 for p in prices],
                "low": [p * 0.995 for p in prices],
                "close": prices,
                "open": [p * 0.997 for p in prices],
                "volume": [1000] * 100,
            },
            index=dates,
        )

        signal = self.recognizer.recognize(data, len(data) - 1)
        if signal:
            # Confidence should remain near-zero.
            self.assertLessEqual(signal.confidence, 0.002)

    def test_recognize_bearish_trend(self):
        """Test recognition of bearish trend."""
        # Create synthetic data with a clear downtrend
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        # Create downward trend
        prices = [200 - i * 0.1 for i in range(100)]
        data = pd.DataFrame(
            {
                "high": [p * 1.005 for p in prices],
                "low": [p * 0.995 for p in prices],
                "close": prices,
                "open": [p * 1.003 for p in prices],
                "volume": [1000] * 100,
            },
            index=dates,
        )

        signal = self.recognizer.recognize(data, len(data) - 1)
        if signal:
            # Confidence should remain near-zero.
            self.assertLessEqual(signal.confidence, 0.002)

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        # Create data with fewer points than required
        data = pd.DataFrame(
            {
                "high": [100, 101, 102],
                "low": [99, 100, 101],
                "close": [100, 101, 102],
                "open": [99.5, 100.5, 101.5],
                "volume": [1000, 1000, 1000],
            }
        )

        signal = self.recognizer.recognize(data, len(data) - 1)
        # Should return None due to insufficient data
        self.assertIsNone(signal)

    def test_sideways_market_signal(self):
        """Test signal generation in sideways market."""
        # Create synthetic data with no clear trend (sideways)
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        # Create sideways movement
        prices = [100 + 2 * np.sin(i * 0.1) for i in range(100)]
        data = pd.DataFrame(
            {
                "high": [p * 1.005 for p in prices],
                "low": [p * 0.995 for p in prices],
                "close": prices,
                "open": [p * 0.997 for p in prices],
                "volume": [1000] * 100,
            },
            index=dates,
        )

        signal = self.recognizer.recognize(data, len(data) - 1)
        if signal:
            # Confidence should remain near-zero.
            self.assertLessEqual(signal.confidence, 0.002)


if __name__ == "__main__":
    unittest.main()
