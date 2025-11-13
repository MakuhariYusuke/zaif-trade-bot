"""
MACD (Moving Average Convergence Divergence) Indicator

Calculates MACD for trend analysis and momentum signals.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from ztb.trading.signal.quality.indicators.base import BaseTrendIndicator


class MACDIndicator(BaseTrendIndicator):
    """
    MACD (Moving Average Convergence Divergence) Indicator

    Shows the relationship between two moving averages of a security's price
    to identify momentum and trend changes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.fast_period = self.config.get('fast_period', 12)
        self.slow_period = self.config.get('slow_period', 26)
        self.signal_period = self.config.get('signal_period', 9)
        # Override required columns - MACD only needs close prices
        self.required_columns = ['close']

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'method': 'ema'  # Only EMA supported for MACD
        }

    def _calculate_indicator(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate MACD values"""
        close = data['close']

        # Calculate EMAs
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        # Get current values
        current_macd = macd_line.iloc[-1] if not macd_line.empty else 0.0
        current_signal = signal_line.iloc[-1] if not signal_line.empty else 0.0
        current_histogram = histogram.iloc[-1] if not histogram.empty else 0.0

        # Calculate MACD momentum (rate of change)
        macd_momentum = macd_line.diff().iloc[-1] if len(macd_line) > 1 else 0.0

        # Determine trend direction
        trend_direction = self.get_trend_direction(current_macd)

        # Signal strength based on histogram
        signal_strength = abs(current_histogram) * 100  # Scale for 0-100

        # Crossover signals
        macd_prev = macd_line.iloc[-2] if len(macd_line) > 1 else current_macd
        signal_prev = signal_line.iloc[-2] if len(signal_line) > 1 else current_signal

        bullish_crossover = (macd_prev <= signal_prev) and (current_macd > current_signal)
        bearish_crossover = (macd_prev >= signal_prev) and (current_macd < current_signal)

        return {
            'macd_line': current_macd,
            'signal_line': current_signal,  # Add signal_line key for compatibility
            'macd_signal': current_signal,  # Keep original key
            'histogram': current_histogram,  # Add histogram key for compatibility
            'macd_histogram': current_histogram,
            'macd_momentum': macd_momentum,
            'trend_direction': trend_direction,
            'signal_strength': min(100.0, signal_strength),
            'bullish_crossover': bullish_crossover,
            'bearish_crossover': bearish_crossover,
            'fast_ema': fast_ema.iloc[-1] if not fast_ema.empty else close.iloc[-1],
            'slow_ema': slow_ema.iloc[-1] if not slow_ema.empty else close.iloc[-1]
        }

    def _get_default_values(self) -> Dict[str, float]:
        """Get default values when calculation fails"""
        return {
            'macd_line': 0.0,
            'signal_line': 0.0,  # Add signal_line key for compatibility
            'macd_signal': 0.0,
            'histogram': 0.0,  # Add histogram key for compatibility
            'macd_histogram': 0.0,
            'macd_momentum': 0.0,
            'trend_direction': 'sideways',
            'signal_strength': 0.0,
            'bullish_crossover': False,
            'bearish_crossover': False,
            'fast_ema': 0.0,
            'slow_ema': 0.0
        }

    def get_required_periods(self) -> int:
        """Get minimum periods required"""
        return max(self.fast_period, self.slow_period, self.signal_period) + 10