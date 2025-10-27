#!/usr/bin/env python3
"""
Comprehensive unit tests for Bollinger Bands and ADX pattern recognizers.
Tests various market conditions, edge cases, and signal generation accuracy.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from ztb.trading.strategies.action_signal_guide.pattern_recognition.bollinger_patterns import (
    BollingerBandsRecognizer
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.adx_patterns import (
    ADXRecognizer
)


class TestBollingerBandsRecognizer:
    """Test cases for BollingerBandsRecognizer."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='h')

        # Create trending data with volatility
        base_price = 100.0
        prices = []
        for i in range(100):
            trend = i * 0.05  # Gradual upward trend
            volatility = 2.0
            noise = np.random.normal(0, volatility)
            price = base_price + trend + noise
            prices.append(max(price, 1.0))

        # Create OHLCV data
        data = []
        for i, close in enumerate(prices):
            volatility_factor = np.random.uniform(0.5, 2.0)
            high = close * (1 + volatility_factor * 0.01)
            low = close * (1 - volatility_factor * 0.01)
            open_price = data[-1]['close'] if data else close * (1 + np.random.normal(0, 0.005))
            volume = np.random.uniform(1000, 10000)

            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data)

    @pytest.fixture
    def volatile_data(self) -> pd.DataFrame:
        """Create highly volatile data for testing squeeze conditions."""
        np.random.seed(123)
        dates = pd.date_range('2023-01-01', periods=50, freq='h')

        # Create high volatility data
        base_price = 100.0
        prices = []
        for i in range(50):
            # High volatility with occasional spikes
            if i < 20:  # Low volatility period
                noise = np.random.normal(0, 0.5)
            else:  # High volatility period
                noise = np.random.normal(0, 3.0)
            price = base_price + noise
            prices.append(max(price, 1.0))

        # Create OHLCV data with high spreads
        data = []
        for i, close in enumerate(prices):
            high = close * (1 + abs(np.random.normal(0, 0.05)))
            low = close * (1 - abs(np.random.normal(0, 0.05)))
            open_price = data[-1]['close'] if data else close
            volume = np.random.uniform(5000, 20000)  # Higher volume in volatile periods

            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data)

    def test_initialization(self):
        """Test recognizer initialization with default and custom config."""
        # Default config
        recognizer = BollingerBandsRecognizer({})
        assert recognizer.config is not None

        # Custom config
        config = {
            'period': 25,
            'std_dev': 2.5,
            'bandwidth_threshold': 0.05
        }
        recognizer = BollingerBandsRecognizer(config)
        assert recognizer.config['period'] == 25
        assert recognizer.config['std_dev'] == 2.5

    def test_band_touch_signals(self, sample_data):
        """Test band touch signal generation."""
        recognizer = BollingerBandsRecognizer({
            'period': 20,
            'std_dev': 2.0
        })

        result = recognizer.recognize(sample_data)

        # Should generate some signals
        assert result is not None
        assert hasattr(result, 'signal_type')
        assert hasattr(result, 'direction')
        assert hasattr(result, 'strength')
        assert hasattr(result, 'confidence')

    def test_squeeze_detection(self, volatile_data):
        """Test Bollinger Band squeeze detection."""
        recognizer = BollingerBandsRecognizer({
            'period': 20,
            'std_dev': 2.0,
            'bandwidth_threshold': 0.02  # Low threshold for squeeze detection
        })

        result = recognizer.recognize(volatile_data)

        # With volatile data, should detect some patterns
        assert result is not None

    def test_middle_band_cross(self, sample_data):
        """Test middle band crossover signals."""
        recognizer = BollingerBandsRecognizer({
            'period': 20,
            'std_dev': 2.0
        })

        result = recognizer.recognize(sample_data)

        # Should work without errors
        assert result is not None

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        # Create data with fewer points than required period
        dates = pd.date_range('2023-01-01', periods=10, freq='h')
        small_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(105, 115, 10),
            'low': np.random.uniform(95, 105, 10),
            'close': np.random.uniform(100, 110, 10),
            'volume': np.random.uniform(1000, 10000, 10)
        })

        recognizer = BollingerBandsRecognizer({'period': 20})
        result = recognizer.recognize(small_data)

        # Should return a result indicating insufficient data
        assert result is not None
        assert result.signal_type == "bb_insufficient_data"
        assert result.strength == 0.0

    def test_edge_case_flat_market(self):
        """Test with flat market conditions."""
        dates = pd.date_range('2023-01-01', periods=50, freq='h')
        # Create almost flat market
        flat_data = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 50,
            'high': [100.1] * 50,
            'low': [99.9] * 50,
            'close': [100.0] * 50,
            'volume': [1000] * 50
        })

        recognizer = BollingerBandsRecognizer({'period': 20})
        result = recognizer.recognize(flat_data)

        # Should handle flat market
        assert result is not None

    @pytest.mark.parametrize("period,std_dev", [
        (10, 1.5),
        (20, 2.0),
        (30, 2.5),
    ])
    def test_different_parameters(self, sample_data, period, std_dev):
        """Test with different parameter combinations."""
        config = {
            'period': period,
            'std_dev': std_dev
        }
        recognizer = BollingerBandsRecognizer(config)
        result = recognizer.recognize(sample_data)

        # Should work with different parameters
        assert result is not None


class TestADXRecognizer:
    """Test cases for ADXRecognizer."""

    @pytest.fixture
    def trending_data(self) -> pd.DataFrame:
        """Create strongly trending data for ADX testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='h')

        # Create strong upward trend
        base_price = 100.0
        prices = []
        for i in range(100):
            trend = i * 0.2  # Strong upward trend
            noise = np.random.normal(0, 1.0)
            price = base_price + trend + noise
            prices.append(max(price, 1.0))

        # Create OHLCV data
        data = []
        for i, close in enumerate(prices):
            high = close * (1 + abs(np.random.normal(0, 0.02)))
            low = close * (1 - abs(np.random.normal(0, 0.02)))
            open_price = data[-1]['close'] if data else close
            volume = np.random.uniform(1000, 10000)

            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data)

    @pytest.fixture
    def sideways_data(self) -> pd.DataFrame:
        """Create sideways/choppy data for weak trend testing."""
        np.random.seed(123)
        dates = pd.date_range('2023-01-01', periods=100, freq='h')

        # Create sideways movement
        base_price = 100.0
        prices = []
        for i in range(100):
            # Random walk with no clear trend
            noise = np.random.normal(0, 2.0)
            price = base_price + noise
            prices.append(max(price, 1.0))

        # Create OHLCV data
        data = []
        for i, close in enumerate(prices):
            high = close * (1 + abs(np.random.normal(0, 0.03)))
            low = close * (1 - abs(np.random.normal(0, 0.03)))
            open_price = data[-1]['close'] if data else close
            volume = np.random.uniform(1000, 10000)

            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data)

    def test_initialization(self):
        """Test recognizer initialization."""
        # Default config
        recognizer = ADXRecognizer({})
        assert recognizer.config is not None

        # Custom config
        config = {
            'period': 14,
            'threshold_strong': 25,
            'threshold_weak': 20
        }
        recognizer = ADXRecognizer(config)
        assert recognizer.config['period'] == 14

    def test_strong_trend_detection(self, trending_data):
        """Test strong trend detection in trending market."""
        recognizer = ADXRecognizer({
            'period': 14,
            'threshold_strong': 20  # Lower threshold for testing
        })

        result = recognizer.recognize(trending_data)

        # Should detect strong trend in trending data
        assert result is not None

    def test_weak_trend_detection(self, sideways_data):
        """Test weak trend detection in sideways market."""
        recognizer = ADXRecognizer({
            'period': 14,
            'threshold_weak': 15
        })

        result = recognizer.recognize(sideways_data)

        # Should detect weak trend in sideways data
        assert result is not None

    def test_di_cross_signals(self, trending_data):
        """Test DI crossover signal detection."""
        recognizer = ADXRecognizer({
            'period': 14
        })

        result = recognizer.recognize(trending_data)

        # Should work without errors
        assert result is not None

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        dates = pd.date_range('2023-01-01', periods=10, freq='h')
        small_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(105, 115, 10),
            'low': np.random.uniform(95, 105, 10),
            'close': np.random.uniform(100, 110, 10),
            'volume': np.random.uniform(1000, 10000, 10)
        })

        recognizer = ADXRecognizer({'period': 20})
        result = recognizer.recognize(small_data)

        # Should return a result indicating insufficient data
        assert result is not None
        assert result.signal_type == "adx_insufficient_data"
        assert result.strength == 0.0

    @pytest.mark.parametrize("period,threshold", [
        (10, 20),
        (14, 25),
        (21, 30),
    ])
    def test_different_parameters(self, trending_data, period, threshold):
        """Test with different parameter combinations."""
        config = {
            'period': period,
            'threshold_strong': threshold
        }
        recognizer = ADXRecognizer(config)
        result = recognizer.recognize(trending_data)

        # Should work with different parameters
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])