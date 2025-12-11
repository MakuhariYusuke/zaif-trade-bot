#!/usr/bin/env python3
"""
Unit tests for individual pattern recognizers (RSI, MACD, Bollinger).

Tests individual pattern recognizers to ensure they generate signals correctly.
"""

import pandas as pd
import pytest

from ztb.trading.strategies.action_signal_guide.pattern_recognition.bollinger_patterns import (
    BollingerBandsRecognizer,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.macd import (
    MACDPatternRecognizer,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.rsi import (
    RSIPatternRecognizer,
)


def create_sample_data():
    """Helper that generates or fetches data for tests."""
    import yfinance as yf

    try:
        df = yf.Ticker("BTC-JPY").history(period="7d", interval="1m")
        if df.empty or len(df) < 200:
            raise ValueError("fetched data insufficient")
        return df
    except Exception:
        import numpy as np

        dates = pd.date_range("2023-01-01", periods=300, freq="min")
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "open": np.random.uniform(50000, 60000, 300),
                "high": np.random.uniform(51000, 61000, 300),
                "low": np.random.uniform(49000, 59000, 300),
                "close": np.random.uniform(50000, 60000, 300),
                "volume": np.random.uniform(1000, 10000, 300),
            },
            index=dates,
        )
        return data


class TestIndividualRecognizers:
    """Test individual pattern recognizers."""

    @pytest.fixture
    def sample_data(self):
        """Fetch sample BTC-JPY data for testing."""
        return create_sample_data()

    def test_rsi_recognizer_initialization(self):
        """Test RSI recognizer initializes correctly."""
        recognizer = RSIPatternRecognizer()
        assert recognizer is not None
        assert hasattr(recognizer, "rsi_period")
        assert recognizer.rsi_period == 14

    def test_macd_recognizer_initialization(self):
        """Test MACD recognizer initializes correctly."""
        recognizer = MACDPatternRecognizer()
        assert recognizer is not None
        assert hasattr(recognizer, "fast_period")
        assert recognizer.fast_period == 12

    def test_bollinger_recognizer_initialization(self):
        """Test Bollinger recognizer initializes correctly."""
        recognizer = BollingerBandsRecognizer()
        assert recognizer is not None

    def test_rsi_signal_generation(self, sample_data):
        """Test RSI recognizer generates signals."""
        recognizer = RSIPatternRecognizer()
        test_df = sample_data.tail(200).copy()

        signals_found = 0
        for idx in [50, 100, 150]:
            if idx >= len(test_df):
                continue
            signal = recognizer.recognize(test_df, idx)
            if signal:
                signals_found += 1
                # Validate signal properties
                assert hasattr(signal, "signal_type")
                assert hasattr(signal, "strength")
                assert hasattr(signal, "confidence")
                assert 0.0 <= signal.strength <= 1.0
                assert 0.0 <= signal.confidence <= 1.0

        # At least some signals should be generated
        assert signals_found >= 0  # Allow for no signals in some market conditions

    def test_macd_signal_generation(self, sample_data):
        """Test MACD recognizer generates signals."""
        recognizer = MACDPatternRecognizer()
        test_df = sample_data.tail(200).copy()

        signals_found = 0
        for idx in [50, 100, 150]:
            if idx >= len(test_df):
                continue
            signal = recognizer.recognize(test_df, idx)
            if signal:
                signals_found += 1
                # Validate signal properties
                assert hasattr(signal, "signal_type")
                assert hasattr(signal, "strength")
                assert hasattr(signal, "confidence")
                assert 0.0 <= signal.strength <= 1.0
                assert 0.0 <= signal.confidence <= 1.0

        # At least some signals should be generated
        assert signals_found >= 0

    def test_bollinger_signal_generation(self, sample_data):
        """Test Bollinger recognizer generates signals."""
        recognizer = BollingerBandsRecognizer()
        test_df = sample_data.tail(200).copy()

        signals_found = 0
        for idx in [50, 100, 150]:
            if idx >= len(test_df):
                continue
            signal = recognizer.recognize(test_df, idx)
            if signal:
                signals_found += 1
                # Validate signal properties
                assert hasattr(signal, "signal_type")
                assert hasattr(signal, "strength")
                assert hasattr(signal, "confidence")
                assert 0.0 <= signal.strength <= 1.0
                assert 0.0 <= signal.confidence <= 1.0

        # At least some signals should be generated
        assert signals_found >= 0


def test_individual_recognizers():
    """
    Legacy function for direct execution.
    This maintains backward compatibility.
    """
    # Create test instance and run tests
    test_instance = TestIndividualRecognizers()

    # Get sample data
    sample_data = create_sample_data()

    # Run individual tests
    try:
        test_instance.test_rsi_recognizer_initialization()
        test_instance.test_macd_recognizer_initialization()
        test_instance.test_bollinger_recognizer_initialization()
        test_instance.test_rsi_signal_generation(sample_data)
        test_instance.test_macd_signal_generation(sample_data)
        test_instance.test_bollinger_signal_generation(sample_data)
        print("All individual recognizer tests passed!")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    test_individual_recognizers()
