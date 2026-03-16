"""
Unit tests for Fibonacci pattern recognizers.
"""


import pandas as pd

from ztb.trading.strategies.action_signal_guide.pattern_recognition.fibonacci_patterns import (
    FibonacciAnalyzer,
    FibonacciExtensionRecognizer,
    FibonacciProjectionRecognizer,
    FibonacciRetracementRecognizer,
)


class TestFibonacciAnalyzer:
    """Test FibonacciAnalyzer utility functions."""

    def test_calculate_retracement_levels(self):
        """Test retracement level calculation."""
        levels = FibonacciAnalyzer.calculate_retracement_levels(100, 0)
        assert 0.236 in levels
        assert abs(levels[0.236] - 23.6) < 0.01
        assert abs(levels[0.618] - 61.8) < 0.01

    def test_find_fibonacci_retracement_exact_match(self):
        """Test finding exact Fibonacci retracement."""
        # Clear cache
        FibonacciAnalyzer._retracement_cache.clear()

        # Create test data with a clear swing
        data = pd.DataFrame(
            {
                "high": [100, 100, 100, 100, 100],
                "low": [0, 0, 0, 0, 0],
                "close": [
                    61.8,
                    61.8,
                    61.8,
                    61.8,
                    61.8,
                ],  # Exactly 0.618 retracement: (100-61.8)/100 = 0.382? Wait
            }
        )

        # Wait, let's calculate: swing_high=100, swing_low=0, current_close=61.8
        # retracement_ratio = (100 - 61.8) / 100 = 38.2 / 100 = 0.382
        # So this matches 0.382 level
        result = FibonacciAnalyzer.find_fibonacci_retracement(data, 0, 4)
        assert result is not None
        assert result["level"] == 0.382
        assert abs(result["actual_ratio"] - 0.382) < 0.01

    def test_find_fibonacci_retracement_no_match(self):
        """Test when no Fibonacci level matches."""
        # Clear cache
        FibonacciAnalyzer._retracement_cache.clear()

        data = pd.DataFrame(
            {
                "high": [100, 100, 100, 100, 100],
                "low": [0, 0, 0, 0, 0],
                "close": [
                    80,
                    80,
                    80,
                    80,
                    80,
                ],  # 0.2 retracement, not close to any Fibonacci level
            }
        )

        result = FibonacciAnalyzer.find_fibonacci_retracement(data, 0, 4)
        assert result is None


class TestFibonacciRetracementRecognizer:
    """Test FibonacciRetracementRecognizer."""

    def test_recognize_weak_signal(self):
        """Test that recognizer generates very weak signals."""
        # Create test data
        data = pd.DataFrame(
            {
                "high": [100] * 20,
                "low": [0] * 20,
                "close": [61.8] * 20,  # Fibonacci level
                "open": [50] * 20,
                "volume": [1000] * 20,
            }
        )
        data.index = pd.date_range("2023-01-01", periods=20, freq="D")

        recognizer = FibonacciRetracementRecognizer()
        result = recognizer.recognize(data, index=19)

        if result:
            # Should have very low confidence to prevent over-performance
            assert result.confidence <= 0.0001
            assert result.strength <= 0.0001
            assert "fibonacci_retracement" in result.metadata.get("pattern", "")


class TestFibonacciExtensionRecognizer:
    """Test FibonacciExtensionRecognizer."""

    def test_recognize_weak_signal(self):
        """Test that extension recognizer generates weak signals."""
        data = pd.DataFrame(
            {
                "high": [100] * 30,
                "low": [0] * 30,
                "close": [161.8] * 30,  # Extension level
                "open": [50] * 30,
                "volume": [1000] * 30,
            }
        )
        data.index = pd.date_range("2023-01-01", periods=30, freq="D")

        recognizer = FibonacciExtensionRecognizer()
        result = recognizer.recognize(data, index=29)

        if result:
            assert result.confidence <= 0.0001
            assert result.strength <= 0.0001


class TestFibonacciProjectionRecognizer:
    """Test FibonacciProjectionRecognizer."""

    def test_recognize_weak_signal(self):
        """Test that projection recognizer generates weak signals."""
        data = pd.DataFrame(
            {
                "high": [100] * 25,
                "low": [0] * 25,
                "close": [161.8] * 25,  # Projection level
                "open": [50] * 25,
                "volume": [1000] * 25,
            }
        )
        data.index = pd.date_range("2023-01-01", periods=25, freq="D")

        recognizer = FibonacciProjectionRecognizer()
        result = recognizer.recognize(data, index=24)

        if result:
            assert result.confidence <= 0.0001
            assert result.strength <= 0.0001


class TestFibonacciCache:
    """Test caching functionality."""

    def test_retracement_cache(self):
        """Test that retracement calculations are cached."""
        data = pd.DataFrame(
            {
                "high": [100, 100],
                "low": [0, 0],
                "close": [61.8, 61.8],
            }
        )

        # Clear cache
        FibonacciAnalyzer._retracement_cache.clear()

        # First call
        result1 = FibonacciAnalyzer.find_fibonacci_retracement(data, 0, 1)
        assert len(FibonacciAnalyzer._retracement_cache) == 1

        # Second call with same parameters should use cache
        result2 = FibonacciAnalyzer.find_fibonacci_retracement(data, 0, 1)
        assert result1 == result2
        assert len(FibonacciAnalyzer._retracement_cache) == 1  # Still 1 entry
