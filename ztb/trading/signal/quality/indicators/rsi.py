"""
RSI (Relative Strength Index) Indicator

Calculates RSI oscillator for momentum analysis.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from ztb.trading.signal.quality.indicators.base import BaseOscillatorIndicator


class RSIIndicator(BaseOscillatorIndicator):
    """
    RSI (Relative Strength Index) Indicator

    Measures the speed and change of price movements to evaluate
    overbought or oversold conditions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.periods = self.config.get('periods', 14)

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'periods': 14,
            'smoothing': 'ema'  # 'ema' or 'sma'
        }

    def _calculate_indicator(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate RSI values"""
        close = data['close']

        # Calculate price changes
        delta = close.diff()

        # Separate gains and losses
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))

        # Calculate average gain and loss
        if self.config.get('smoothing', 'ema') == 'ema':
            avg_gain = gain.ewm(span=self.periods, adjust=False).mean()
            avg_loss = loss.ewm(span=self.periods, adjust=False).mean()
        else:
            avg_gain = gain.rolling(window=self.periods).mean()
            avg_loss = loss.rolling(window=self.periods).mean()

        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        # Get current RSI value
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50.0

        # Validate and clamp
        current_rsi = self._validate_oscillator_value(current_rsi)

        # Calculate RSI signal strength
        signal = self.get_oscillator_signal(current_rsi)

        # Calculate RSI slope (momentum)
        rsi_slope = rsi.diff().iloc[-1] if len(rsi) > 1 else 0.0

        return {
            'rsi': current_rsi,
            'rsi_signal': signal,
            'rsi_slope': rsi_slope,
            'avg_gain': avg_gain.iloc[-1] if not avg_gain.empty else 0.0,
            'avg_loss': avg_loss.iloc[-1] if not avg_loss.empty else 0.0
        }

    def _get_default_values(self) -> Dict[str, float]:
        """Get default values when calculation fails"""
        return {
            'rsi': 50.0,
            'rsi_signal': 'neutral',
            'rsi_slope': 0.0,
            'avg_gain': 0.0,
            'avg_loss': 0.0
        }

    def get_required_periods(self) -> int:
        """Get minimum periods required"""
        return self.periods + 1  # Need one extra for diff