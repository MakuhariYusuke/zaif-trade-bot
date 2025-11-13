"""
Unit tests for modular technical indicators system

Tests the modular technical indicator system with caching
and various indicator implementations.
"""

import pytest
import pandas as pd
import numpy as np
import inspect
from unittest.mock import Mock, patch

from ztb.trading.signal.quality.indicators.base import (
    BaseTechnicalIndicator, BaseOscillatorIndicator, BaseTrendIndicator,
    BaseVolatilityIndicator, BaseVolumeIndicator, CompositeIndicator, AdaptiveIndicator
)
from ztb.trading.signal.quality.indicators.rsi import RSIIndicator
from ztb.trading.signal.quality.indicators.macd import MACDIndicator


class TestBaseTechnicalIndicator:
    """Test BaseTechnicalIndicator abstract base class"""

    def test_abstract_methods(self):
        """Test that BaseTechnicalIndicator is abstract and has required abstract methods"""
        # Test that it's an abstract class
        assert inspect.isabstract(BaseTechnicalIndicator)

        # Test that required abstract methods are defined
        abstract_methods = BaseTechnicalIndicator.__abstractmethods__
        assert '_calculate_indicator' in abstract_methods
        assert '_get_default_values' in abstract_methods

    def test_cache_functionality(self):
        """Test caching functionality"""
        from ztb.trading.signal.quality.indicators.rsi import RSIIndicator
        indicator = RSIIndicator()

        # Create test data
        data1 = pd.DataFrame({'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]})
        data2 = pd.DataFrame({'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]})  # Same data
        data3 = pd.DataFrame({'close': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]})  # Different data

        # First call should calculate
        result1 = indicator.calculate(data1)
        assert isinstance(result1, dict)
        assert 'rsi' in result1

        # Second call with same data should use cache
        result2 = indicator.calculate(data2)
        assert result2 == result1

        # Different data should give different result (different length)
        data4 = pd.DataFrame({'close': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118]})  # Different length
        result4 = indicator.calculate(data4)
        assert result4 != result1

    def test_indicator_base_functionality(self):
        """Test cache key generation"""
        from ztb.trading.signal.quality.indicators.rsi import RSIIndicator
        indicator = RSIIndicator()

        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'timestamp': pd.date_range('2024-01-01', periods=15)
        })
        data.set_index('timestamp', inplace=True)

        key1 = indicator.get_cache_key(data)
        key2 = indicator.get_cache_key(data)

        assert key1 == key2  # Same data should generate same key
        assert isinstance(key1, str)

    def test_empty_data_cache_key(self):
        """Test cache key for empty data"""
        from ztb.trading.signal.quality.indicators.rsi import RSIIndicator
        indicator = RSIIndicator()

        data = pd.DataFrame()
        key = indicator.get_cache_key(data)

        assert key == "empty"


class TestBaseOscillatorIndicator:
    """Test BaseOscillatorIndicator"""

    def test_inheritance(self):
        """Test that BaseOscillatorIndicator is abstract and inherits from BaseTechnicalIndicator"""
        # Test that it's an abstract class
        assert inspect.isabstract(BaseOscillatorIndicator)

        # Test inheritance
        assert issubclass(BaseOscillatorIndicator, BaseTechnicalIndicator)

    def test_abstract_calculate_method(self):
        """Test that _calculate_indicator method is abstract"""
        # Test that required abstract methods are defined
        abstract_methods = BaseOscillatorIndicator.__abstractmethods__
        assert '_calculate_indicator' in abstract_methods
        assert '_get_default_values' in abstract_methods

class TestBaseTrendIndicator:
    """Test BaseTrendIndicator"""

    def test_inheritance(self):
        """Test that BaseTrendIndicator is abstract and inherits from BaseTechnicalIndicator"""
        # Test that it's an abstract class
        assert inspect.isabstract(BaseTrendIndicator)

        # Test inheritance
        assert issubclass(BaseTrendIndicator, BaseTechnicalIndicator)

    def test_abstract_calculate_method(self):
        """Test that _calculate_indicator method is abstract"""
        # Test that required abstract methods are defined
        abstract_methods = BaseTrendIndicator.__abstractmethods__
        assert '_calculate_indicator' in abstract_methods
        assert '_get_default_values' in abstract_methods

class TestBaseVolatilityIndicator:
    """Test BaseVolatilityIndicator"""

    def test_inheritance(self):
        """Test that BaseVolatilityIndicator is abstract and inherits from BaseTechnicalIndicator"""
        # Test that it's an abstract class
        assert inspect.isabstract(BaseVolatilityIndicator)

        # Test inheritance
        assert issubclass(BaseVolatilityIndicator, BaseTechnicalIndicator)

    def test_abstract_calculate_method(self):
        """Test that _calculate_indicator method is abstract"""
        # Test that required abstract methods are defined
        abstract_methods = BaseVolatilityIndicator.__abstractmethods__
        assert '_calculate_indicator' in abstract_methods
        assert '_get_default_values' in abstract_methods

class TestBaseVolumeIndicator:
    """Test BaseVolumeIndicator"""

    def test_inheritance(self):
        """Test that BaseVolumeIndicator is abstract and inherits from BaseTechnicalIndicator"""
        # Test that it's an abstract class
        assert inspect.isabstract(BaseVolumeIndicator)

        # Test inheritance
        assert issubclass(BaseVolumeIndicator, BaseTechnicalIndicator)

    def test_abstract_calculate_method(self):
        """Test that _calculate_indicator method is abstract"""
        # Test that required abstract methods are defined
        abstract_methods = BaseVolumeIndicator.__abstractmethods__
        assert '_calculate_indicator' in abstract_methods
        assert '_get_default_values' in abstract_methods


class TestRSIIndicator:
    """Test RSIIndicator implementation"""

    @pytest.fixture
    def rsi_indicator(self):
        """Create RSI indicator instance"""
        return RSIIndicator({'periods': 14})

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing"""
        # Create 20 periods of price data
        prices = []
        base_price = 100.0
        for i in range(20):
            # Add some oscillation
            change = np.sin(i / 3) * 2
            price = base_price + change
            prices.append(price)
            base_price = price

        return pd.DataFrame({'close': prices})

    def test_initialization(self, rsi_indicator):
        """Test RSI indicator initialization"""
        assert rsi_indicator.config['periods'] == 14
        assert hasattr(rsi_indicator, 'calculate')

    def test_calculate_rsi_basic(self, rsi_indicator, sample_price_data):
        """Test basic RSI calculation"""
        result = rsi_indicator.calculate(sample_price_data)

        assert isinstance(result, dict)
        assert 'rsi' in result
        assert isinstance(result['rsi'], (int, float))
        assert 0 <= result['rsi'] <= 100

    def test_rsi_overbought_oversold(self, rsi_indicator):
        """Test RSI overbought/oversold conditions"""
        # Create overbought condition (consistently rising prices)
        overbought_prices = list(range(100, 120))  # 20 increasing prices
        data = pd.DataFrame({'close': overbought_prices})

        result = rsi_indicator.calculate(data)
        assert result['rsi'] > 70  # Should be overbought

        # Create oversold condition (consistently falling prices)
        oversold_prices = list(range(120, 100, -1))  # 20 decreasing prices
        data = pd.DataFrame({'close': oversold_prices})

        result = rsi_indicator.calculate(data)
        assert result['rsi'] < 30  # Should be oversold

    def test_rsi_neutral(self, rsi_indicator):
        """Test RSI neutral conditions"""
        # Create neutral oscillating data
        neutral_prices = [100, 102, 98, 101, 99, 100, 102, 98, 101, 99,
                         100, 102, 98, 101, 99, 100, 102, 98, 101, 99]
        data = pd.DataFrame({'close': neutral_prices})

        result = rsi_indicator.calculate(data)
        assert 40 <= result['rsi'] <= 60  # Should be neutral

    def test_rsi_insufficient_data(self, rsi_indicator):
        """Test RSI with insufficient data"""
        data = pd.DataFrame({'close': [100, 101]})  # Less than period (14)

        result = rsi_indicator.calculate(data)

        # Should still return a result (may be default or calculated with available data)
        assert isinstance(result, dict)
        assert 'rsi' in result

    def test_rsi_caching(self, rsi_indicator, sample_price_data):
        """Test RSI caching functionality"""
        # First calculation
        result1 = rsi_indicator.calculate(sample_price_data)

        # Second calculation with same data should use cache
        result2 = rsi_indicator.calculate(sample_price_data)

        assert result1 == result2

    def test_rsi_signal_interpretation(self, rsi_indicator):
        """Test RSI signal interpretation ranges"""
        test_cases = [
            # (prices, expected_range)
            ([100] * 20, (45, 55)),  # No change, neutral
            (list(range(100, 120)), (70, 100)),  # Rising, overbought
            (list(range(120, 100, -1)), (0, 30)),  # Falling, oversold
        ]

        for prices, expected_range in test_cases:
            data = pd.DataFrame({'close': prices})
            result = rsi_indicator.calculate(data)
            rsi_value = result['rsi']

            assert expected_range[0] <= rsi_value <= expected_range[1], \
                f"RSI {rsi_value} not in expected range {expected_range} for prices {prices[:3]}..."


class TestMACDIndicator:
    """Test MACDIndicator implementation"""

    @pytest.fixture
    def macd_indicator(self):
        """Create MACD indicator instance"""
        return MACDIndicator({
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        })

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for MACD testing"""
        # Need enough data for MACD calculation (slow_period + signal_period)
        np.random.seed(42)
        prices = []
        base_price = 100.0

        for i in range(50):  # More data for MACD
            # Add trend with noise
            trend = 0.001 * i
            noise = np.random.normal(0, 0.01)
            price = base_price * (1 + trend + noise)
            prices.append(price)
            base_price = price

        return pd.DataFrame({'close': prices})

    def test_initialization(self, macd_indicator):
        """Test MACD indicator initialization"""
        assert macd_indicator.config['fast_period'] == 12
        assert macd_indicator.config['slow_period'] == 26
        assert macd_indicator.config['signal_period'] == 9

    def test_calculate_macd_basic(self, macd_indicator, sample_price_data):
        """Test basic MACD calculation"""
        result = macd_indicator.calculate(sample_price_data)

        assert isinstance(result, dict)
        assert 'macd_line' in result
        assert 'signal_line' in result
        assert 'histogram' in result

        # All should be numeric
        for key in ['macd_line', 'signal_line', 'histogram']:
            assert isinstance(result[key], (int, float, np.number))

    def test_macd_bullish_crossover(self, macd_indicator):
        """Test MACD bullish crossover detection"""
        # Create bullish crossover scenario
        # Start with downtrend, then uptrend
        prices = []
        base_price = 100.0

        # Downtrend
        for i in range(30):
            price = base_price * (1 - 0.001 * i)
            prices.append(price)

        # Uptrend (bullish crossover)
        for i in range(30):
            price = base_price * (1 + 0.001 * i)
            prices.append(price)

        data = pd.DataFrame({'close': prices})
        result = macd_indicator.calculate(data)

        # MACD line should cross above signal line in bullish scenario
        # (This is a simplified test - actual crossover detection would need more analysis)
        assert 'macd_line' in result
        assert 'signal_line' in result

    def test_macd_bearish_crossover(self, macd_indicator):
        """Test MACD bearish crossover detection"""
        # Create bearish crossover scenario
        prices = []
        base_price = 100.0

        # Uptrend
        for i in range(30):
            price = base_price * (1 + 0.001 * i)
            prices.append(price)

        # Downtrend (bearish crossover)
        for i in range(30):
            price = base_price * (1 - 0.001 * i)
            prices.append(price)

        data = pd.DataFrame({'close': prices})
        result = macd_indicator.calculate(data)

        assert 'macd_line' in result
        assert 'signal_line' in result

    def test_macd_histogram(self, macd_indicator, sample_price_data):
        """Test MACD histogram calculation"""
        result = macd_indicator.calculate(sample_price_data)

        # Histogram should be MACD line minus signal line
        expected_histogram = result['macd_line'] - result['signal_line']
        assert abs(result['histogram'] - expected_histogram) < 1e-10

    def test_macd_insufficient_data(self, macd_indicator):
        """Test MACD with insufficient data"""
        data = pd.DataFrame({'close': [100, 101, 102]})  # Very little data

        result = macd_indicator.calculate(data)

        # Should still return a result
        assert isinstance(result, dict)
        assert 'macd_line' in result

    def test_macd_caching(self, macd_indicator, sample_price_data):
        """Test MACD caching functionality"""
        # First calculation
        result1 = macd_indicator.calculate(sample_price_data)

        # Second calculation with same data should use cache
        result2 = macd_indicator.calculate(sample_price_data)

        assert result1 == result2

    def test_macd_signal_strength(self, macd_indicator):
        """Test MACD signal strength interpretation"""
        # Strong uptrend
        strong_up_prices = [100 + i * 0.5 for i in range(60)]
        data = pd.DataFrame({'close': strong_up_prices})
        result = macd_indicator.calculate(data)

        # In strong uptrend, MACD should be positive
        assert result['macd_line'] > 0

        # Strong downtrend
        strong_down_prices = [150 - i * 0.5 for i in range(60)]
        data = pd.DataFrame({'close': strong_down_prices})
        result = macd_indicator.calculate(data)

        # In strong downtrend, MACD should be negative
        assert result['macd_line'] < 0


class TestCompositeIndicator:
    """Test CompositeIndicator"""

    def test_initialization(self):
        """Test CompositeIndicator initialization"""
        # Create mock indicators
        mock_rsi = Mock(spec=BaseTechnicalIndicator)
        mock_rsi.name = 'rsi'
        mock_rsi.calculate.return_value = {'rsi': 65.0}

        mock_macd = Mock(spec=BaseTechnicalIndicator)
        mock_macd.name = 'macd'
        mock_macd.calculate.return_value = {'macd_line': 1.5, 'signal_line': 1.2, 'histogram': 0.3}

        indicators = [mock_rsi, mock_macd]
        weights = {'rsi': 0.6, 'macd': 0.4}

        composite = CompositeIndicator(indicators, weights)

        assert composite.indicators == indicators
        assert composite.weights == weights

    def test_calculate_composite(self):
        """Test composite calculation"""
        # Create mock indicators
        mock_rsi = Mock(spec=BaseTechnicalIndicator)
        mock_rsi.name = 'rsi'
        mock_rsi.calculate.return_value = {'rsi': 70.0}

        mock_macd = Mock(spec=BaseTechnicalIndicator)
        mock_macd.name = 'macd'
        mock_macd.calculate.return_value = {'macd_line': 2.0, 'signal_line': 1.5, 'histogram': 0.5}

        indicators = [mock_rsi, mock_macd]
        weights = {'rsi': 0.6, 'macd': 0.4}

        composite = CompositeIndicator(indicators, weights)
        data = pd.DataFrame({'close': [100, 101, 102]})

        result = composite.calculate(data)

        assert isinstance(result, dict)
        assert 'composite_score' in result
        assert isinstance(result['composite_score'], (int, float))

    def test_equal_weights_default(self):
        """Test default equal weights"""
        mock_indicator1 = Mock(spec=BaseTechnicalIndicator)
        mock_indicator1.name = 'indicator1'
        mock_indicator1.calculate.return_value = {'value': 1.0}

        mock_indicator2 = Mock(spec=BaseTechnicalIndicator)
        mock_indicator2.name = 'indicator2'
        mock_indicator2.calculate.return_value = {'value': 2.0}

        indicators = [mock_indicator1, mock_indicator2]
        composite = CompositeIndicator(indicators)  # No weights provided

        # Should have equal weights
        expected_weights = {'indicator1': 0.5, 'indicator2': 0.5}
        assert composite.weights == expected_weights


class TestAdaptiveIndicator:
    """Test AdaptiveIndicator"""

    def test_initialization(self):
        """Test AdaptiveIndicator initialization"""
        base_indicator = Mock(spec=BaseTechnicalIndicator)
        base_indicator.name = 'base_indicator'

        adaptive = AdaptiveIndicator(base_indicator)

        assert adaptive.base_indicator == base_indicator
        assert hasattr(adaptive, 'adapt_parameters')

    def test_adaptive_calculation(self):
        """Test adaptive calculation with market regime"""
        base_indicator = Mock(spec=BaseTechnicalIndicator)
        base_indicator.name = 'rsi'
        base_indicator.calculate.return_value = {'rsi': 65.0}
        base_indicator.config = {'periods': 14}  # Add config attribute
        base_indicator._get_default_values.return_value = {'rsi': 50.0}  # Mock default values

        adaptive = AdaptiveIndicator(base_indicator)

        # Mock adapt_parameters method
        adaptive.adapt_parameters = Mock(return_value={'periods': 21})  # Adapted config

        data = pd.DataFrame({'close': [100, 101, 102]})
        market_regime = 'trending'

        result = adaptive.calculate_adaptive(data, market_regime)

        assert isinstance(result, dict)
        assert 'rsi' in result
        adaptive.adapt_parameters.assert_called_once_with(market_regime)


class TestIndicatorIntegration:
    """Test integration of multiple indicators"""

    def test_rsi_and_macd_together(self):
        """Test RSI and MACD working together"""
        rsi_indicator = RSIIndicator({'periods': 14})
        macd_indicator = MACDIndicator({
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        })

        # Create test data
        prices = [100 + np.sin(i / 5) * 5 for i in range(60)]
        data = pd.DataFrame({'close': prices})

        rsi_result = rsi_indicator.calculate(data)
        macd_result = macd_indicator.calculate(data)

        # Both should return valid results
        assert 'rsi' in rsi_result
        assert 'macd_line' in macd_result
        assert 'signal_line' in macd_result
        assert 'histogram' in macd_result

        # Values should be reasonable
        assert 0 <= rsi_result['rsi'] <= 100
        assert isinstance(macd_result['macd_line'], (int, float))
        assert isinstance(macd_result['signal_line'], (int, float))
        assert isinstance(macd_result['histogram'], (int, float))

    def test_indicator_error_handling(self):
        """Test error handling in indicators"""
        rsi_indicator = RSIIndicator({'periods': 14})

        # Test with completely invalid data
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})

        # Should handle gracefully (may return default values)
        result = rsi_indicator.calculate(invalid_data)
        assert isinstance(result, dict)

    def test_indicator_config_variations(self):
        """Test indicators with different configurations"""
        # RSI with different periods
        rsi_5 = RSIIndicator({'periods': 5})
        rsi_21 = RSIIndicator({'periods': 21})

        data = pd.DataFrame({'close': list(range(100, 130))})

        result_5 = rsi_5.calculate(data)
        result_21 = rsi_21.calculate(data)

        # Different periods should give different results
        assert result_5 != result_21

        # MACD with different periods
        macd_short = MACDIndicator({'fast_period': 8, 'slow_period': 17, 'signal_period': 6})
        macd_long = MACDIndicator({'fast_period': 12, 'slow_period': 26, 'signal_period': 9})

        result_short = macd_short.calculate(data)
        result_long = macd_long.calculate(data)

        # Different periods should give different results
        assert result_short != result_long

    def test_indicator_caching_integration(self):
        """Test caching works across multiple indicators"""
        rsi_indicator = RSIIndicator({'periods': 14})
        macd_indicator = MACDIndicator({
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        })

        data = pd.DataFrame({'close': list(range(100, 160))})

        # First calculations
        rsi_result1 = rsi_indicator.calculate(data)
        macd_result1 = macd_indicator.calculate(data)

        # Second calculations with same data (should use cache)
        rsi_result2 = rsi_indicator.calculate(data)
        macd_result2 = macd_indicator.calculate(data)

        # Results should be identical
        assert rsi_result1 == rsi_result2
        assert macd_result1 == macd_result2