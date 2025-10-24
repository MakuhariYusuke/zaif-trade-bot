"""
Base classes for pattern recognition in Action Signal Guide.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd

from ztb.trading.constants import ACTION_HOLD, ACTION_BUY, ACTION_SELL


class SignalResult:
    """Result of a pattern recognition signal."""

    def __init__(self,
                 signal_type: str,
                 strength: float,
                 direction: int,  # 1 for buy, -1 for sell, 0 for neutral
                 description: str,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        self.signal_type = signal_type
        self.strength = strength
        self.direction = direction
        self.description = description
        self.metadata = metadata or {}


class PatternRecognizer(ABC):
    """
    Base class for all pattern recognizers.

    Provides common functionality and interface for pattern recognition.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """
        Recognize pattern in the given data at the specified index.

        Args:
            data: OHLCV data as pandas DataFrame
            index: Index to check for pattern (default: last row)

        Returns:
            SignalResult if pattern found, None otherwise
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid
        """
        required_columns = ['open', 'high', 'low', 'close']
        return all(col in data.columns for col in required_columns)

    def get_lookback_period(self) -> int:
        """
        Get the number of periods this pattern needs to look back.

        Returns:
            Number of periods required for pattern recognition
        """
        return self.config.get('lookback_period', 20)

    def calculate_body_size(self, data: pd.DataFrame, index: int) -> float:
        """Calculate candle body size."""
        return abs(data.iloc[index]['close'] - data.iloc[index]['open'])

    def calculate_upper_shadow(self, data: pd.DataFrame, index: int) -> float:
        """Calculate upper shadow size."""
        high = data.iloc[index]['high']
        return high - max(data.iloc[index]['open'], data.iloc[index]['close'])

    def calculate_lower_shadow(self, data: pd.DataFrame, index: int) -> float:
        """Calculate lower shadow size."""
        low = data.iloc[index]['low']
        return abs(min(data.iloc[index]['open'], data.iloc[index]['close']) - low)

    def is_bullish_candle(self, data: pd.DataFrame, index: Optional[int] = None) -> bool:
        """Check if candle is bullish."""
        if isinstance(data, pd.Series):
            # If data is a Series (single candle), check directly
            return data['close'] > data['open']
        elif index is not None:
            # If data is DataFrame and index is provided
            return data.iloc[index]['close'] > data.iloc[index]['open']
        else:
            raise ValueError("Either provide a Series or DataFrame with index")

    def is_bearish_candle(self, data: pd.DataFrame, index: Optional[int] = None) -> bool:
        """Check if candle is bearish."""
        if isinstance(data, pd.Series):
            # If data is a Series (single candle), check directly
            return data['close'] < data['open']
        elif index is not None:
            # If data is DataFrame and index is provided
            return data.iloc[index]['close'] < data.iloc[index]['open']
        else:
            raise ValueError("Either provide a Series or DataFrame with index")

    def get_body_ratio(self, data: pd.DataFrame, index: int) -> float:
        """Get body size as ratio of total range."""
        high = data.iloc[index]['high']
        low = data.iloc[index]['low']
        total_range = high - low
        if total_range == 0:
            return 0.0
        return self.calculate_body_size(data, index) / total_range


class CandlestickPatternRecognizer(PatternRecognizer):
    """
    Base class for candlestick pattern recognizers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.body_ratio_threshold = self.config.get('body_ratio_threshold', 0.6)
        self.shadow_ratio_threshold = self.config.get('shadow_ratio_threshold', 0.3)

    def is_hammer_like(self, data: pd.DataFrame, index: int) -> bool:
        """Check if candle resembles hammer pattern."""
        if not self.validate_data(data) or index < 0 or index >= len(data):
            return False

        body_ratio = self.get_body_ratio(data, index)
        lower_shadow = self.calculate_lower_shadow(data, index)
        upper_shadow = self.calculate_upper_shadow(data, index)
        total_range = data.iloc[index]['high'] - data.iloc[index]['low']

        if total_range == 0:
            return False

        # Hammer characteristics: small body, long lower shadow, small upper shadow
        return (body_ratio < self.body_ratio_threshold and
                lower_shadow > upper_shadow * 2 and
                lower_shadow > body_ratio * total_range)

    def is_shooting_star_like(self, data: pd.DataFrame, index: int) -> bool:
        """Check if candle resembles shooting star pattern."""
        if not self.validate_data(data) or index < 0:
            return False

        body_ratio = self.get_body_ratio(data, index)
        lower_shadow = self.calculate_lower_shadow(data, index)
        upper_shadow = self.calculate_upper_shadow(data, index)
        total_range = data.iloc[index]['high'] - data.iloc[index]['low']

        if total_range == 0:
            return False

        # Shooting star characteristics: small body, long upper shadow, small lower shadow
        return (body_ratio < self.body_ratio_threshold and
                upper_shadow > lower_shadow * 2 and
                upper_shadow > body_ratio * total_range)

    def _is_uptrend(self, data: pd.DataFrame, index: int, lookback: int) -> bool:
        """Check if there's an uptrend over the lookback period."""
        if index < lookback:
            return False
        recent_prices = data.iloc[index-lookback+1:index+1]['close']
        return recent_prices.iloc[-1] > recent_prices.iloc[0]
    
    def _is_downtrend(self, data: pd.DataFrame, index: int, lookback: int) -> bool:
        """Check if there's a downtrend over the lookback period."""
        if index < lookback:
            return False
        recent_prices = data.iloc[index-lookback+1:index+1]['close']
        return recent_prices.iloc[-1] < recent_prices.iloc[0]

    def _is_small_candle(self, candle: pd.Series) -> bool:
        """Check if candle has small body relative to recent volatility."""
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        return body_size / total_range < 0.3 if total_range > 0 else False
    
    def _is_large_candle(self, candle: pd.Series) -> bool:
        """Check if candle has large body relative to recent volatility."""
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        return body_size / total_range > 0.6 if total_range > 0 else False

    def _get_average_body_size(self, data: pd.DataFrame, index: int, lookback: int) -> float:
        """Calculate average body size over lookback period."""
        if index < lookback:
            return 0
        bodies = [self.calculate_body_size(data, i) for i in range(index-lookback+1, index+1)]
        return np.mean(bodies) if bodies else 0


class MultiCandlePatternRecognizer(PatternRecognizer):
    """
    Base class for multi-candle pattern recognizers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.min_trend_length = self.config.get('min_trend_length', 5)

    def detect_trend(self, data: pd.DataFrame, start_index: int, length: int) -> int:
        """
        Detect trend direction over a period.

        Returns:
            1 for uptrend, -1 for downtrend, 0 for sideways
        """
        if start_index - length + 1 < 0 or start_index >= len(data):
            return 0

        prices = data.iloc[start_index - length + 1:start_index + 1]['close'].values
        if len(prices) < length:
            return 0

        # Lightweight trend detection using price difference
        diff = prices[-1] - prices[0]

        if diff > 0.001:  # Uptrend threshold
            return ACTION_BUY
        elif diff < -0.001:  # Downtrend threshold
            return ACTION_SELL
        else:
            return ACTION_HOLD