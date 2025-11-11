"""
Technical Indicators Module for Signal Quality Scoring

軽量なテクニカル指標計算モジュール
既存の特徴量システムとTA-Libラッパーを活用した実装

Features:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Lightweight wrapper for signal quality scoring
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ztb.utils.talib_wrapper import TaLibWrapper
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """
    Lightweight technical indicators calculator for signal quality scoring

    Provides essential technical indicators using existing infrastructure
    """

    def __init__(self):
        self.talib = TaLibWrapper()

    def calculate_rsi(self, prices: Union[np.ndarray, pd.Series], period: int = 14) -> float:
        """
        Calculate RSI (Relative Strength Index)

        Args:
            prices: Price data array
            period: RSI period (default: 14)

        Returns:
            Current RSI value (0-100)
        """
        try:
            rsi_values = self.talib.rsi(prices, period)
            # Return the last valid RSI value
            valid_rsi = rsi_values[~np.isnan(rsi_values)]
            return float(valid_rsi[-1]) if len(valid_rsi) > 0 else 50.0
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return 50.0  # Neutral value

    def calculate_macd(self, prices: Union[np.ndarray, pd.Series],
                       fast_period: int = 12, slow_period: int = 26,
                       signal_period: int = 9) -> Tuple[float, float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            prices: Price data array
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        try:
            macd_line, signal_line, histogram = self.talib.macd(
                prices, fast_period, slow_period, signal_period
            )

            # Return the last valid values
            def get_last_valid(arr):
                valid = arr[~np.isnan(arr)]
                return float(valid[-1]) if len(valid) > 0 else 0.0

            return (
                get_last_valid(macd_line),
                get_last_valid(signal_line),
                get_last_valid(histogram)
            )
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
            return (0.0, 0.0, 0.0)

    def calculate_bollinger_bands(self, prices: Union[np.ndarray, pd.Series],
                                  period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands

        Args:
            prices: Price data array
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        try:
            upper, middle, lower = self.talib.bbands(prices, period, std_dev)

            # Return the last valid values
            def get_last_valid(arr):
                valid = arr[~np.isnan(arr)]
                return float(valid[-1]) if len(valid) > 0 else 0.0

            return (
                get_last_valid(upper),
                get_last_valid(middle),
                get_last_valid(lower)
            )
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {e}")
            # Fallback calculation
            if len(prices) >= period:
                recent_prices = prices[-period:]
                middle = np.mean(recent_prices)
                std = np.std(recent_prices)
                upper = middle + (std_dev * std)
                lower = middle - (std_dev * std)
                return (upper, middle, lower)
            return (0.0, 0.0, 0.0)

    def calculate_atr(self, high: Union[np.ndarray, pd.Series],
                      low: Union[np.ndarray, pd.Series],
                      close: Union[np.ndarray, pd.Series],
                      period: int = 14) -> float:
        """
        Calculate ATR (Average True Range)

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            Current ATR value
        """
        try:
            atr_values = self.talib.atr(high, low, close, period)
            # Return the last valid ATR value
            valid_atr = atr_values[~np.isnan(atr_values)]
            return float(valid_atr[-1]) if len(valid_atr) > 0 else 0.0
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}")
            return 0.0

    def get_technical_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get comprehensive technical signals from OHLCV data

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Dictionary of technical indicators
        """
        signals = {}

        try:
            # RSI
            if 'close' in df.columns:
                signals['rsi'] = self.calculate_rsi(df['close'].values)

            # MACD
            if 'close' in df.columns:
                macd_line, signal_line, histogram = self.calculate_macd(df['close'].values)
                signals['macd_line'] = macd_line
                signals['macd_signal'] = signal_line
                signals['macd_histogram'] = histogram

            # Bollinger Bands
            if 'close' in df.columns:
                upper, middle, lower = self.calculate_bollinger_bands(df['close'].values)
                signals['bb_upper'] = upper
                signals['bb_middle'] = middle
                signals['bb_lower'] = lower

                # Bollinger Band position (%B)
                if 'close' in df.columns and len(df) > 0:
                    current_price = df['close'].iloc[-1]
                    if upper > lower:  # Avoid division by zero
                        signals['bb_position'] = (current_price - lower) / (upper - lower)
                    else:
                        signals['bb_position'] = 0.5

            # ATR
            if all(col in df.columns for col in ['high', 'low', 'close']):
                signals['atr'] = self.calculate_atr(
                    df['high'].values, df['low'].values, df['close'].values
                )

        except Exception as e:
            logger.error(f"Error calculating technical signals: {e}")

        return signals