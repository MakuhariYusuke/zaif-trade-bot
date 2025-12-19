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

from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from ztb.features.generators.technical.momentum.rsi import compute_rsi
from ztb.features.generators.technical.momentum.stochastic import (
    compute_stochastic,
    compute_stochastic_k,
)
from ztb.utils.logging_utils import get_logger
from ztb.utils.talib_wrapper import TaLibWrapper

logger = get_logger(__name__)


class TechnicalIndicators:
    """
    Lightweight technical indicators calculator for signal quality scoring

    Provides essential technical indicators using existing infrastructure
    """

    def __init__(self):
        self.talib = TaLibWrapper()

    def calculate_rsi(
        self, prices: Union[np.ndarray, pd.Series], period: int = 14
    ) -> float:
        """
        Calculate RSI (Relative Strength Index)

        Args:
            prices: Price data array
            period: RSI period (default: 14)

        Returns:
            Current RSI value (0-100)
        """
        try:
            # Convert to DataFrame
            if isinstance(prices, pd.Series):
                df = pd.DataFrame({"close": prices})
            else:
                df = pd.DataFrame({"close": prices})

            rsi_series = compute_rsi(df, period=period)

            # Get last valid value
            last_val = rsi_series.iloc[-1]
            return float(last_val) if not pd.isna(last_val) else 50.0
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return 50.0  # Neutral value

    def calculate_macd(
        self,
        prices: Union[np.ndarray, pd.Series],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[float, float, float]:
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
                get_last_valid(histogram),
            )
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
            return (0.0, 0.0, 0.0)

    def calculate_bollinger_bands(
        self,
        prices: Union[np.ndarray, pd.Series],
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Tuple[float, float, float]:
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

            return (
                get_last_valid(upper),
                get_last_valid(middle),
                get_last_valid(lower),
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

    def calculate_atr(
        self,
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        period: int = 14,
    ) -> float:
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

    def calculate_stochastic(
        self,
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[float, float]:
        """
        Calculate Stochastic Oscillator

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)

        Returns:
            Tuple of (%K, %D)
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(
                {
                    "high": high
                    if isinstance(high, (pd.Series, np.ndarray))
                    else np.array(high),
                    "low": low
                    if isinstance(low, (pd.Series, np.ndarray))
                    else np.array(low),
                    "close": close
                    if isinstance(close, (pd.Series, np.ndarray))
                    else np.array(close),
                }
            )

            d_series = compute_stochastic(df, period=k_period, smooth_k=d_period)
            k_series = compute_stochastic_k(df, period=k_period, smooth_k=d_period)

            k_val = (
                float(k_series.iloc[-1])
                if not k_series.empty and not pd.isna(k_series.iloc[-1])
                else 50.0
            )
            d_val = (
                float(d_series.iloc[-1])
                if not d_series.empty and not pd.isna(d_series.iloc[-1])
                else 50.0
            )

            return k_val, d_val
        except Exception as e:
            logger.warning(f"Stochastic calculation failed: {e}")
            return 50.0, 50.0
            return (50.0, 50.0)

    def calculate_momentum(
        self, prices: Union[np.ndarray, pd.Series], period: int = 10
    ) -> float:
        """
        Calculate Momentum indicator

        Args:
            prices: Price data array
            period: Momentum period (default: 10)

        Returns:
            Current momentum value
        """
        try:
            if len(prices) < period + 1:
                return 0.0

            # Momentum = Current price - Price n periods ago
            current_price = prices[-1]
            past_price = prices[-(period + 1)]
            momentum = current_price - past_price

            # Normalize to percentage change
            if past_price > 0:
                momentum_pct = (momentum / past_price) * 100
                return momentum_pct
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Momentum calculation failed: {e}")
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
            if "close" in df.columns:
                signals["rsi"] = self.calculate_rsi(df["close"].to_numpy())

            # MACD
            if "close" in df.columns:
                macd_line, signal_line, histogram = self.calculate_macd(
                    df["close"].to_numpy()
                )
                signals["macd_line"] = macd_line
                signals["macd_signal"] = signal_line
                signals["macd_histogram"] = histogram

            # Bollinger Bands
            if "close" in df.columns:
                upper, middle, lower = self.calculate_bollinger_bands(
                    df["close"].to_numpy()
                )
                signals["bb_upper"] = upper
                signals["bb_middle"] = middle
                signals["bb_lower"] = lower

                # Bollinger Band position (%B)
                if "close" in df.columns and len(df) > 0:
                    current_price = df["close"].iloc[-1]
                    if upper > lower:  # Avoid division by zero
                        signals["bb_position"] = (current_price - lower) / (
                            upper - lower
                        )
                    else:
                        signals["bb_position"] = 0.5

            # ATR
            if all(col in df.columns for col in ["high", "low", "close"]):
                signals["atr"] = self.calculate_atr(
                    df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
                )

            # Stochastic Oscillator
            if all(col in df.columns for col in ["high", "low", "close"]):
                stoch_k, stoch_d = self.calculate_stochastic(
                    df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
                )
                signals["stoch_k"] = stoch_k
                signals["stoch_d"] = stoch_d

            # Momentum
            if "close" in df.columns:
                signals["momentum"] = self.calculate_momentum(df["close"].to_numpy())

        except Exception as e:
            logger.error(f"Error calculating technical signals: {e}")

        return signals
