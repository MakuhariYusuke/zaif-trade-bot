"""
Unit tests for Technical Indicators Module

テストケース:
- RSI calculation accuracy
- MACD calculation and signals
- Bollinger Bands calculation
- ATR calculation
- Comprehensive technical signals
"""

import numpy as np
import pandas as pd
import pytest

from ztb.trading.signal.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """Test cases for TechnicalIndicators class"""

    @pytest.fixture
    def indicators(self):
        """Create TechnicalIndicators instance"""
        return TechnicalIndicators()

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing"""
        np.random.seed(42)
        n_points = 100

        # Generate realistic price data
        base_price = 50000
        prices = [base_price]
        for i in range(n_points - 1):
            change = np.random.normal(0, 0.02)  # 2% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # Floor price

        # Create OHLCV data
        data = []
        for i, close in enumerate(prices):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = data[-1]['close'] if data else close * (1 + np.random.normal(0, 0.005))
            volume = np.random.lognormal(10, 1)  # Realistic volume

            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data)

    def test_calculate_rsi(self, indicators, sample_ohlcv_data):
        """Test RSI calculation"""
        prices = sample_ohlcv_data['close'].values
        rsi = indicators.calculate_rsi(prices)

        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100

        # Test with oversold data (should be low RSI)
        oversold_prices = np.array([100] * 20 + [90] * 5)  # Sharp decline
        rsi_oversold = indicators.calculate_rsi(oversold_prices)
        assert rsi_oversold < 30  # Should be oversold

        # Test with overbought data (should be high RSI)
        overbought_prices = np.array([100] * 20 + [110] * 5)  # Sharp increase
        rsi_overbought = indicators.calculate_rsi(overbought_prices)
        assert rsi_overbought > 70  # Should be overbought

    def test_calculate_macd(self, indicators, sample_ohlcv_data):
        """Test MACD calculation"""
        prices = sample_ohlcv_data['close'].values
        macd_line, signal_line, histogram = indicators.calculate_macd(prices)

        # All values should be floats
        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert isinstance(histogram, float)

        # MACD histogram should be difference between lines
        assert abs(histogram - (macd_line - signal_line)) < 1e-10

    def test_calculate_bollinger_bands(self, indicators, sample_ohlcv_data):
        """Test Bollinger Bands calculation"""
        prices = sample_ohlcv_data['close'].values
        upper, middle, lower = indicators.calculate_bollinger_bands(prices)

        # All values should be positive
        assert upper > 0
        assert middle > 0
        assert lower > 0

        # Upper should be above middle, middle above lower
        assert upper > middle > lower

        # Current price should be within reasonable range of middle
        current_price = prices[-1]
        assert lower <= current_price <= upper

    def test_calculate_atr(self, indicators, sample_ohlcv_data):
        """Test ATR calculation"""
        high = sample_ohlcv_data['high'].values
        low = sample_ohlcv_data['low'].values
        close = sample_ohlcv_data['close'].values

        atr = indicators.calculate_atr(high, low, close)

        # ATR should be positive
        assert atr > 0

        # ATR should be reasonable (not too large compared to price)
        current_price = close[-1]
        assert atr < current_price * 0.5  # ATR shouldn't be more than 50% of price

    def test_get_technical_signals(self, indicators, sample_ohlcv_data):
        """Test comprehensive technical signals"""
        signals = indicators.get_technical_signals(sample_ohlcv_data)

        # Should contain expected keys
        expected_keys = ['rsi', 'macd_line', 'macd_signal', 'macd_histogram',
                        'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'atr']
        for key in expected_keys:
            assert key in signals
            assert isinstance(signals[key], (int, float))

        # RSI should be valid
        assert 0 <= signals['rsi'] <= 100

        # Bollinger Bands should be properly ordered
        assert signals['bb_upper'] > signals['bb_middle'] > signals['bb_lower']

        # Bollinger position should be between 0 and 1
        assert 0 <= signals['bb_position'] <= 1

        # ATR should be positive
        assert signals['atr'] > 0

    def test_edge_cases(self, indicators):
        """Test edge cases and error handling"""
        # Empty data
        empty_df = pd.DataFrame()
        signals = indicators.get_technical_signals(empty_df)
        assert isinstance(signals, dict)

        # Minimal data
        minimal_df = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [100],
            'volume': [1000]
        })
        signals = indicators.get_technical_signals(minimal_df)
        assert isinstance(signals, dict)

        # NaN values
        nan_df = pd.DataFrame({
            'open': [np.nan],
            'high': [np.nan],
            'low': [np.nan],
            'close': [np.nan],
            'volume': [np.nan]
        })
        signals = indicators.get_technical_signals(nan_df)
        assert isinstance(signals, dict)

    def test_rsi_extremes(self, indicators):
        """Test RSI with extreme price movements"""
        # Strong uptrend
        uptrend_prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                                  110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
        rsi_up = indicators.calculate_rsi(uptrend_prices)
        assert rsi_up > 50  # Should show bullish momentum

        # Strong downtrend
        downtrend_prices = np.array([119, 118, 117, 116, 115, 114, 113, 112, 111, 110,
                                    109, 108, 107, 106, 105, 104, 103, 102, 101, 100])
        rsi_down = indicators.calculate_rsi(downtrend_prices)
        assert rsi_down < 50  # Should show bearish momentum