"""
ATR (Average True Range) implementation.
平均真の範囲 - ボラティリティ指標
"""

import numpy as np
import pandas as pd

from ztb.features.registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper


@FeatureRegistry.register("ATR")
def compute_atr(df: pd.DataFrame, period: int = 14) -> "pd.Series":
    """Compute Average True Range (ATR) with Ta-Lib support"""
    high_prices = np.asarray(df["high"].values, dtype=float)
    low_prices = np.asarray(df["low"].values, dtype=float)
    close_prices = np.asarray(df["close"].values, dtype=float)

    atr_values = TaLibWrapper.atr(high_prices, low_prices, close_prices, period)
    return pd.Series(atr_values, index=df.index)


@FeatureRegistry.register("ATR_simplified")
def compute_atr_simplified(df: pd.DataFrame, period: int = 10) -> "pd.Series":
    """Compute Simplified ATR (period=10) with Ta-Lib support"""
    return compute_atr(df, period)
