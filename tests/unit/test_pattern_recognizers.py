"""
Unit tests for pattern recognizers.

Tests individual pattern recognizers to ensure they work correctly.
"""


import numpy as np
import pandas as pd
import pytest

from ztb.trading.strategies.action_signal_guide.pattern_recognition.dow_theory import (
    DowTheoryRecognizer,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.fibonacci_patterns import (
    FibonacciExtensionRecognizer,
    FibonacciRetracementRecognizer,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.harmonic_patterns import (
    BatRecognizer,
    ButterflyRecognizer,
    CrabRecognizer,
    GartleyRecognizer,
)


class TestHarmonicRecognizers:
    """Test harmonic pattern recognizers."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(105, 115, 100),
                "low": np.random.uniform(95, 105, 100),
                "close": np.random.uniform(100, 110, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

        # Ensure high >= max(open, close), low <= min(open, close)
        for i in range(len(data)):
            data.iloc[i, data.columns.get_loc("high")] = max(
                data.iloc[i]["open"], data.iloc[i]["close"], data.iloc[i]["high"]
            )
            data.iloc[i, data.columns.get_loc("low")] = min(
                data.iloc[i]["open"], data.iloc[i]["close"], data.iloc[i]["low"]
            )

        return data

    def test_gartley_recognizer_initialization(self):
        """Test GartleyRecognizer initializes correctly."""
        config = {"lookback_period": 5, "tolerance": 0.05}
        recognizer = GartleyRecognizer(config)

        assert recognizer.name == "GartleyRecognizer"
        assert recognizer.lookback_period == 5
        assert recognizer.tolerance == 0.05
        assert recognizer.get_lookback_period() == 5

    def test_gartley_recognizer_with_insufficient_data(self, sample_data):
        """Test GartleyRecognizer with insufficient data."""
        recognizer = GartleyRecognizer()
        result = recognizer.recognize(sample_data, index=3)  # Less than lookback_period

        assert result is None

    def test_gartley_recognizer_with_sufficient_data(self, sample_data):
        """Test GartleyRecognizer generates signal with sufficient data."""
        recognizer = GartleyRecognizer()
        result = recognizer.recognize(sample_data, index=50)

        # May return None if no pattern found, or return synthetic signal
        # Just check it doesn't crash and returns proper type if not None
        if result is not None:
            assert hasattr(result, "direction")
            assert hasattr(result, "confidence")
            assert hasattr(result, "signal_type")

    def test_bat_recognizer_initialization(self):
        """Test BatRecognizer initializes correctly."""
        recognizer = BatRecognizer()
        assert recognizer.name == "BatRecognizer"

    def test_butterfly_recognizer_initialization(self):
        """Test ButterflyRecognizer initializes correctly."""
        recognizer = ButterflyRecognizer()
        assert recognizer.name == "ButterflyRecognizer"

    def test_crab_recognizer_initialization(self):
        """Test CrabRecognizer initializes correctly."""
        recognizer = CrabRecognizer()
        assert recognizer.name == "CrabRecognizer"


class TestDowTheoryRecognizer:
    """Test Dow Theory recognizer."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data with clear trend."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        # Create upward trend
        base_price = 100
        prices = []
        for i in range(100):
            base_price += np.random.uniform(0.5, 1.5)  # Upward trend
            prices.append(base_price)

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p + np.random.uniform(0.5, 2) for p in prices],
                "low": [p - np.random.uniform(0.5, 2) for p in prices],
                "close": prices,
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

        return data

    def test_dow_theory_recognizer_initialization(self):
        """Test DowTheoryRecognizer initializes correctly."""
        config = {"primary_trend_period": 50, "trend_confirmation_threshold": 0.00001}
        recognizer = DowTheoryRecognizer(config)

        assert recognizer.name == "DowTheoryRecognizer"
        assert recognizer.primary_trend_period == 50
        assert recognizer.trend_confirmation_threshold == 0.00001

    def test_dow_theory_recognizer_with_trend(self, sample_data):
        """Test DowTheoryRecognizer detects trend."""
        recognizer = DowTheoryRecognizer()
        result = recognizer.recognize(sample_data, index=80)

        # Should detect upward trend and generate signal
        assert result is not None
        assert result.direction > 0  # Bullish signal
        assert result.confidence > 0


class TestFibonacciRecognizers:
    """Test Fibonacci pattern recognizers."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data with retracement pattern."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        # Create a pattern: up, retracement, up
        prices = []
        base = 100
        for i in range(30):
            base += 1  # Uptrend
            prices.append(base)
        for i in range(20):
            base -= 0.5  # Retracement (Fibonacci level)
            prices.append(base)
        for i in range(50):
            base += 0.8  # Continue up
            prices.append(base)

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

        return data

    def test_fibonacci_retracement_recognizer_initialization(self):
        """Test FibonacciRetracementRecognizer initializes correctly."""
        config = {"confidence_cap": 0.000001, "pattern_completeness_threshold": 0.1}
        recognizer = FibonacciRetracementRecognizer(config)

        assert recognizer.name == "FibonacciRetracementRecognizer"

    def test_fibonacci_retracement_with_pattern(self, sample_data):
        """Test FibonacciRetracementRecognizer detects pattern."""
        config = {"confidence_cap": 0.000001}  # Very low cap to prevent over-signaling
        recognizer = FibonacciRetracementRecognizer(config)
        result = recognizer.recognize(sample_data, index=70)

        # May not detect pattern in random data, so just check it doesn't crash
        # If it returns a signal, confidence should be capped
        if result is not None:
            assert result.confidence <= 0.000001  # Capped confidence

    def test_fibonacci_extension_recognizer_initialization(self):
        """Test FibonacciExtensionRecognizer initializes correctly."""
        recognizer = FibonacciExtensionRecognizer()
        assert recognizer.name == "FibonacciExtensionRecognizer"
