"""
Keltner Channels feature implementation.
Keltner Channels are volatility-based bands that use ATR for band calculation.

Output columns:
  - keltner_upper: Upper Keltner Channel
  - keltner_middle: Middle Keltner Channel (EMA)
  - keltner_lower: Lower Keltner Channel
  - keltner_position: Price position within channel (0-1)
  - keltner_width: Channel width
"""

from typing import Any

import pandas as pd

from ..base import BaseFeature
from ..registry import FeatureRegistry


@FeatureRegistry.register("Keltner_Upper")
def compute_keltner_upper(df: pd.DataFrame) -> pd.Series:
    """Keltner Channel Upper Band"""
    feature = KeltnerChannels()
    result_df = feature.compute(df)
    return result_df["keltner_upper"]


@FeatureRegistry.register("Keltner_Lower")
def compute_keltner_lower(df: pd.DataFrame) -> pd.Series:
    """Keltner Channel Lower Band"""
    feature = KeltnerChannels()
    result_df = feature.compute(df)
    return result_df["keltner_lower"]


@FeatureRegistry.register("Keltner_Position")
def compute_keltner_position(df: pd.DataFrame) -> pd.Series:
    """Keltner Channel Price Position (0-1)"""
    feature = KeltnerChannels()
    result_df = feature.compute(df)
    return result_df["keltner_position"]


@FeatureRegistry.register("Keltner_Width")
def compute_keltner_width(df: pd.DataFrame) -> pd.Series:
    """Keltner Channel Width"""
    feature = KeltnerChannels()
    result_df = feature.compute(df)
    return result_df["keltner_width"]


class KeltnerChannels(BaseFeature):
    """
    Keltner Channels using ATR for band calculation.
    """

    def __init__(self, period: int = 20, multiplier: float = 2.0, **kwargs: Any):
        super().__init__("KeltnerChannels", deps=["high", "low", "close"])
        self.period = period
        self.multiplier = multiplier

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Keltner Channel values.
        """
        period = params.get("period", self.period)
        multiplier = params.get("multiplier", self.multiplier)

        # Use pre-computed EMA if available, otherwise compute it
        ema_col = f"ema_{period}"
        if ema_col in df.columns:
            middle = df[ema_col]
        else:
            middle = df["close"].ewm(span=period, adjust=False).mean()

        # Calculate ATR (assuming it's available, otherwise calculate it)
        if "atr" in df.columns:
            atr = df["atr"]
        else:
            # Simple ATR calculation
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift(1)).abs()
            low_close = (df["low"] - df["close"].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()

        # Calculate Keltner Channels
        upper = middle + (atr * multiplier)
        lower = middle - (atr * multiplier)

        # Calculate position within channel
        position = (df["close"] - lower) / (upper - lower)

        # Calculate channel width
        width = (upper - lower) / middle

        return pd.DataFrame(
            {
                "keltner_upper": upper,
                "keltner_middle": middle,
                "keltner_lower": lower,
                "keltner_position": position,
                "keltner_width": width,
            },
            index=df.index,
        )
