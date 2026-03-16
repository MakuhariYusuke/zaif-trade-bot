#!/usr/bin/env python3
"""
Unit tests for new pattern recognizers using existing feature classes.
Tests RSI, MACD, and ATR pattern recognizers with pytest framework.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# Import new pattern recognizers
from ztb.trading.strategies.action_signal_guide.pattern_recognition.rsi import RSIPatternRecognizer
from ztb.trading.strategies.action_signal_guide.pattern_recognition.macd import MACDPatternRecognizer
from ztb.trading.strategies.action_signal_guide.pattern_recognition.atr import ATRPatternRecognizer


@pytest.fixture
def sample_data():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)

    # Generate base price series with stronger trend and more volatility
    base_price = 100
    prices = [base_price]

    length = 150
    for i in range(length - 1):
        # Create stronger trend with more volatility
        trend = 0.002 if i < length // 3 else -0.003 if i < 2 * length // 3 else 0.001
        volatility = 0.03  # Higher volatility
        noise = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(max(new_price, 0.1))  # Prevent negative prices

    # Generate OHLCV from price series with more realistic spreads
    data = []
    for i, close in enumerate(prices):
        volatility_factor = np.random.uniform(0.005, 0.02)  # Variable volatility
        high = close * (1 + volatility_factor * np.random.uniform(0.5, 2.0))
        low = close * (1 - volatility_factor * np.random.uniform(0.5, 2.0))
        open_price = data[-1]['close'] if data else close * (1 + np.random.normal(0, 0.005))
        volume = np.random.uniform(5000, 50000)  # Higher volume

        data.append({
            'timestamp': datetime.now() + timedelta(minutes=i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    return pd.DataFrame(data)


class TestRSIPatternRecognizer:
    """Test cases for RSIPatternRecognizer."""

    def test_initialization(self):
        """Test recognizer initialization with default and custom config."""
        # Default config
        recognizer = RSIPatternRecognizer({})
        assert recognizer.config is not None

        # Custom config
        config = {
            'rsi_period': 21,
            'overbought_level': 75,
            'oversold_level': 25
        }
        recognizer = RSIPatternRecognizer(config)
        assert recognizer.config['rsi_period'] == 21

    def test_recognize_with_sample_data(self, sample_data):
        """Test RSI pattern recognition with sample data."""
        recognizer = RSIPatternRecognizer({
            'rsi_period': 14,
            'overbought_level': 70,
            'oversold_level': 30,
            'divergence_lookback': 5
        })

        result = recognizer.recognize(sample_data)

        # Should return a result (may be None if no signal detected)
        assert result is None or hasattr(result, 'signal_type')

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        small_data = pd.DataFrame({
            'timestamp': [datetime.now()] * 5,
            'open': [100] * 5,
            'high': [101] * 5,
            'low': [99] * 5,
            'close': [100] * 5,
            'volume': [1000] * 5
        })

        recognizer = RSIPatternRecognizer({'rsi_period': 14})
        result = recognizer.recognize(small_data)

        # Should handle insufficient data gracefully
        assert result is None


class TestMACDPatternRecognizer:
    """Test cases for MACDPatternRecognizer."""

    def test_initialization(self):
        """Test recognizer initialization."""
        # Default config
        recognizer = MACDPatternRecognizer({})
        assert recognizer.config is not None

        # Custom config
        config = {
            'fast_period': 8,
            'slow_period': 21,
            'signal_period': 5
        }
        recognizer = MACDPatternRecognizer(config)
        assert recognizer.config['fast_period'] == 8

    def test_recognize_with_sample_data(self, sample_data):
        """Test MACD pattern recognition with sample data."""
        recognizer = MACDPatternRecognizer({
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'histogram_threshold': 0.0
        })

        result = recognizer.recognize(sample_data)

        # Should return a result (may be None if no signal detected)
        assert result is None or hasattr(result, 'signal_type')

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        small_data = pd.DataFrame({
            'timestamp': [datetime.now()] * 10,
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })

        recognizer = MACDPatternRecognizer({'slow_period': 26})
        result = recognizer.recognize(small_data)

        # Should handle insufficient data gracefully
        assert result is None


class TestATRPatternRecognizer:
    """Test cases for ATRPatternRecognizer."""

    def test_initialization(self):
        """Test recognizer initialization."""
        # Default config
        recognizer = ATRPatternRecognizer({})
        assert recognizer.config is not None

        # Custom config
        config = {
            'atr_period': 21,
            'volatility_threshold': 1.5,
            'trend_strength_period': 10
        }
        recognizer = ATRPatternRecognizer(config)
        assert recognizer.config['atr_period'] == 21

    def test_recognize_with_sample_data(self, sample_data):
        """Test ATR pattern recognition with sample data."""
        recognizer = ATRPatternRecognizer({
            'atr_period': 14,
            'volatility_threshold': 1.2,
            'trend_strength_period': 5
        })

        result = recognizer.recognize(sample_data)

        # Should return a result (may be None if no signal detected)
        assert result is None or hasattr(result, 'signal_type')

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        small_data = pd.DataFrame({
            'timestamp': [datetime.now()] * 5,
            'open': [100] * 5,
            'high': [101] * 5,
            'low': [99] * 5,
            'close': [100] * 5,
            'volume': [1000] * 5
        })

        recognizer = ATRPatternRecognizer({'atr_period': 14})
        result = recognizer.recognize(small_data)

        # Should handle insufficient data gracefully
        assert result is None


@pytest.mark.parametrize("recognizer_class,config", [
    (RSIPatternRecognizer, {'rsi_period': 14}),
    (MACDPatternRecognizer, {'fast_period': 12, 'slow_period': 26}),
    (ATRPatternRecognizer, {'atr_period': 14}),
])
def test_recognizer_interfaces(recognizer_class, config, sample_data):
    """Test that all recognizers have consistent interfaces."""
    recognizer = recognizer_class(config)

    # Should have recognize method
    assert hasattr(recognizer, 'recognize')

    # Should accept DataFrame and return optional result
    result = recognizer.recognize(sample_data)

    # Result should be None or have expected attributes
    if result is not None:
        assert hasattr(result, 'signal_type')
        assert hasattr(result, 'direction')
        assert hasattr(result, 'strength')
        assert hasattr(result, 'confidence')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])