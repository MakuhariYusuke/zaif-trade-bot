"""
ATR (Average True Range) implementation with multi-timeframe support.
平均真の範囲 - ボラティリティ指標 - 複数時間軸対応
"""

import numpy as np
import pandas as pd
from typing import Optional

from ztb.features.registry import FeatureRegistry
from ztb.features.timeframe import Timeframe
from ztb.utils.talib_wrapper import TaLibWrapper


@FeatureRegistry.register("ATR")
def compute_atr(df: pd.DataFrame, period: Optional[int] = None, timeframe: Optional[Timeframe] = None) -> "pd.Series":
    """Compute Average True Range (ATR) with Ta-Lib support"""
    # Determine period based on timeframe
    if timeframe is not None:
        from ztb.features.timeframe import get_timeframe_params
        tf_params = get_timeframe_params(timeframe)
        period = period or (tf_params["short_period"] // 2)  # ATR typically uses moderate periods
    else:
        period = period or 14

    high_prices = df["high"]
    low_prices = df["low"]
    close_prices = df["close"]

    talib = TaLibWrapper()
    atr_values = talib.atr(high_prices, low_prices, close_prices, period)
    return pd.Series(atr_values, index=df.index)


@FeatureRegistry.register("ATR_simplified")
def compute_atr_simplified(df: pd.DataFrame, period: Optional[int] = None, timeframe: Optional[Timeframe] = None) -> "pd.Series":
    """Compute Simplified ATR with timeframe support"""
    # For simplified ATR, use shorter period
    if timeframe is not None:
        from ztb.features.timeframe import get_timeframe_params
        tf_params = get_timeframe_params(timeframe)
        period = period or (tf_params["short_period"] // 4)
    else:
        period = period or 10

    return compute_atr(df, period, timeframe)


# === Multi-Timeframe ATR Features ===

@FeatureRegistry.register("ATR_M1")
def compute_atr_m1(df: pd.DataFrame) -> pd.Series:
    """ATR for 1-minute timeframe"""
    return compute_atr(df, timeframe=Timeframe.M1)


@FeatureRegistry.register("ATR_M5")
def compute_atr_m5(df: pd.DataFrame) -> pd.Series:
    """ATR for 5-minute timeframe"""
    return compute_atr(df, timeframe=Timeframe.M5)


@FeatureRegistry.register("ATR_M15")
def compute_atr_m15(df: pd.DataFrame) -> pd.Series:
    """ATR for 15-minute timeframe"""
    return compute_atr(df, timeframe=Timeframe.M15)


@FeatureRegistry.register("ATR_H1")
def compute_atr_h1(df: pd.DataFrame) -> pd.Series:
    """ATR for 1-hour timeframe"""
    return compute_atr(df, timeframe=Timeframe.H1)


@FeatureRegistry.register("ATR_H4")
def compute_atr_h4(df: pd.DataFrame) -> pd.Series:
    """ATR for 4-hour timeframe"""
    return compute_atr(df, timeframe=Timeframe.H4)


@FeatureRegistry.register("ATR_D1")
def compute_atr_d1(df: pd.DataFrame) -> pd.Series:
    """ATR for daily timeframe"""
    return compute_atr(df, timeframe=Timeframe.D1)


@FeatureRegistry.register("ATR_simplified_M1")
def compute_atr_simplified_m1(df: pd.DataFrame) -> pd.Series:
    """Simplified ATR for 1-minute timeframe"""
    return compute_atr_simplified(df, timeframe=Timeframe.M1)


@FeatureRegistry.register("ATR_simplified_M5")
def compute_atr_simplified_m5(df: pd.DataFrame) -> pd.Series:
    """Simplified ATR for 5-minute timeframe"""
    return compute_atr_simplified(df, timeframe=Timeframe.M5)


@FeatureRegistry.register("ATR_simplified_M15")
def compute_atr_simplified_m15(df: pd.DataFrame) -> pd.Series:
    """Simplified ATR for 15-minute timeframe"""
    return compute_atr_simplified(df, timeframe=Timeframe.M15)


@FeatureRegistry.register("ATR_simplified_H1")
def compute_atr_simplified_h1(df: pd.DataFrame) -> pd.Series:
    """Simplified ATR for 1-hour timeframe"""
    return compute_atr_simplified(df, timeframe=Timeframe.H1)


@FeatureRegistry.register("ATR_simplified_H4")
def compute_atr_simplified_h4(df: pd.DataFrame) -> pd.Series:
    """Simplified ATR for 4-hour timeframe"""
    return compute_atr_simplified(df, timeframe=Timeframe.H4)


@FeatureRegistry.register("ATR_simplified_D1")
def compute_atr_simplified_d1(df: pd.DataFrame) -> pd.Series:
    """Simplified ATR for daily timeframe"""
    return compute_atr_simplified(df, timeframe=Timeframe.D1)
