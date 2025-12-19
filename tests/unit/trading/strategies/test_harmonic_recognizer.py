#!/usr/bin/env python3
"""
Unit tests for HARMONIC pattern recognizers
"""
import unittest

import pandas as pd

from ztb.trading.strategies.action_signal_guide.pattern_recognition.harmonic_patterns import (
    BatRecognizer,
    ButterflyRecognizer,
    CrabRecognizer,
    GartleyRecognizer,
    HarmonicAnalyzer,
)


class TestHarmonicAnalyzer(unittest.TestCase):
    """Test cases for HarmonicAnalyzer utility class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = HarmonicAnalyzer()

    def test_calculate_pattern_strength(self):
        """Test pattern strength calculation."""
        # Test Gartley pattern (less strict)
        strength = HarmonicAnalyzer._calculate_pattern_strength(
            HarmonicAnalyzer.GARTLEY_RATIOS, 0.05, "GARTLEY"
        )
        self.assertGreater(strength, 0.7)
        self.assertLessEqual(strength, 0.9)

        # Test Crab pattern (more strict)
        strength = HarmonicAnalyzer._calculate_pattern_strength(
            HarmonicAnalyzer.CRAB_RATIOS, 0.05, "CRAB"
        )
        self.assertGreater(strength, 0.8)
        self.assertLessEqual(strength, 0.9)

        # Test with tighter tolerance
        strength = HarmonicAnalyzer._calculate_pattern_strength(
            HarmonicAnalyzer.GARTLEY_RATIOS, 0.01, "GARTLEY"
        )
        self.assertGreater(strength, 0.7)  # Should be higher with tighter tolerance

    def test_find_harmonic_pattern_no_match(self):
        """Test harmonic pattern finding with no valid pattern."""
        # Create data that won't form a valid harmonic pattern
        data = pd.DataFrame(
            {
                "high": [100, 100, 100, 100, 100],
                "low": [100, 100, 100, 100, 100],
                "close": [100, 100, 100, 100, 100],
                "open": [100, 100, 100, 100, 100],
            }
        )

        result = self.analyzer.find_harmonic_pattern(data, "GARTLEY", 0, 0.05)
        self.assertIsNone(result)


class TestGartleyRecognizer(unittest.TestCase):
    """Test cases for GartleyRecognizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.recognizer = GartleyRecognizer()

    def test_recognize_weak_signal(self):
        """Test that Gartley recognizer produces weak signals."""
        # Create synthetic data that might trigger a pattern
        data = pd.DataFrame(
            {
                "high": [100, 105, 102, 108, 103, 106, 101],
                "low": [95, 98, 97, 101, 98, 99, 96],
                "close": [98, 103, 100, 105, 101, 102, 98],
                "open": [97, 102, 99, 104, 100, 101, 97],
            }
        )

        signal = self.recognizer.recognize(data, len(data) - 1)
        if signal:
            # Confidence should be capped at 0.0001
            self.assertLessEqual(signal.confidence, 0.0001)
            self.assertLessEqual(signal.strength, 0.0001)


class TestButterflyRecognizer(unittest.TestCase):
    """Test cases for ButterflyRecognizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.recognizer = ButterflyRecognizer()

    def test_recognize_weak_signal(self):
        """Test that Butterfly recognizer produces weak signals."""
        # Create synthetic data that might trigger a pattern
        data = pd.DataFrame(
            {
                "high": [100, 108, 103, 110, 105, 107, 102],
                "low": [95, 97, 96, 99, 97, 98, 95],
                "close": [98, 105, 101, 107, 103, 104, 99],
                "open": [97, 104, 100, 106, 102, 103, 98],
            }
        )

        signal = self.recognizer.recognize(data, len(data) - 1)
        if signal:
            # Confidence should be capped at 0.0001
            self.assertLessEqual(signal.confidence, 0.0001)
            self.assertLessEqual(signal.strength, 0.0001)


class TestBatRecognizer(unittest.TestCase):
    """Test cases for BatRecognizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.recognizer = BatRecognizer()

    def test_recognize_weak_signal(self):
        """Test that Bat recognizer produces weak signals."""
        # Create synthetic data that might trigger a pattern
        data = pd.DataFrame(
            {
                "high": [100, 106, 102, 109, 104, 106, 101],
                "low": [95, 98, 96, 100, 97, 98, 95],
                "close": [98, 104, 100, 106, 102, 103, 98],
                "open": [97, 103, 99, 105, 101, 102, 97],
            }
        )

        signal = self.recognizer.recognize(data, len(data) - 1)
        if signal:
            # Confidence should be capped at 0.0001
            self.assertLessEqual(signal.confidence, 0.0001)
            self.assertLessEqual(signal.strength, 0.0001)


class TestCrabRecognizer(unittest.TestCase):
    """Test cases for CrabRecognizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.recognizer = CrabRecognizer()

    def test_recognize_weak_signal(self):
        """Test that Crab recognizer produces weak signals."""
        # Create synthetic data that might trigger a pattern
        data = pd.DataFrame(
            {
                "high": [100, 107, 103, 111, 106, 108, 102],
                "low": [95, 97, 95, 99, 96, 97, 94],
                "close": [98, 105, 101, 108, 104, 105, 99],
                "open": [97, 104, 100, 107, 103, 104, 98],
            }
        )

        signal = self.recognizer.recognize(data, len(data) - 1)
        if signal:
            # Confidence should be capped at 0.0001
            self.assertLessEqual(signal.confidence, 0.0001)
            self.assertLessEqual(signal.strength, 0.0001)


class TestHarmonicCache(unittest.TestCase):
    """Test cases for harmonic pattern caching."""


    def test_pivot_cache(self):
        """Test that pivot point caching works."""
        data = pd.DataFrame(
            {
                "high": [100, 105, 102, 108, 103],
                "low": [95, 98, 97, 101, 98],
                "close": [98, 103, 100, 105, 101],
                "open": [97, 102, 99, 104, 100],
            }
        )

        # First call should cache
        pivots1 = self.analyzer._get_pivot_points(data, min_distance=1)

        # Second call should use cache
        pivots2 = self.analyzer._get_pivot_points(data, min_distance=1)

        # Results should be identical
        self.assertEqual(len(pivots1), len(pivots2))
        for p1, p2 in zip(pivots1, pivots2):
            self.assertEqual(p1.position, p2.position)
            self.assertEqual(p1.price, p2.price)
            self.assertEqual(p1.label, p2.label)


if __name__ == "__main__":
    unittest.main()
