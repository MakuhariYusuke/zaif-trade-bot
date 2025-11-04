"""
ADX (Average Directional Index) implementation with multi-timeframe support.
トレンド強度を測定する方向性移動指標 - 複数時間軸対応
"""

from typing import Optional

import pandas as pd

from ztb.features.core.registry import FeatureRegistry
from ztb.features.timeframe import Timeframe
from ztb.utils.talib_wrapper import TaLibWrapper


@FeatureRegistry.register("ADX")
def compute_adx(
    df: pd.DataFrame,
    period: Optional[int] = None,
    timeframe: Optional[Timeframe] = None,
) -> pd.Series:
    """Compute ADX (Average Directional Index) using Ta-Lib wrapper"""
    # Determine period based on timeframe
    if timeframe is not None:
        from ztb.features.timeframe import get_timeframe_params

        tf_params = get_timeframe_params(timeframe)
        period = (
            period or tf_params["medium_period"]
        )  # ADX typically uses longer periods
    else:
        period = period or 14

    high = df["high"]
    low = df["low"]
    close = df["close"]

    talib = TaLibWrapper()
    result = talib.adx(high, low, close, period)
    return pd.Series(result, index=df.index)


@FeatureRegistry.register("PlusDI")
def compute_plus_di(
    df: pd.DataFrame,
    period: Optional[int] = None,
    timeframe: Optional[Timeframe] = None,
) -> pd.Series:
    """Compute +DI (Positive Directional Indicator) using Ta-Lib wrapper"""
    # Determine period based on timeframe
    if timeframe is not None:
        from ztb.features.timeframe import get_timeframe_params

        tf_params = get_timeframe_params(timeframe)
        period = period or tf_params["medium_period"]
    else:
        period = period or 14

    high = df["high"]
    low = df["low"]
    close = df["close"]

    talib = TaLibWrapper()
    result = talib.plus_di(high, low, close, period)
    return pd.Series(result, index=df.index)


@FeatureRegistry.register("MinusDI")
def compute_minus_di(
    df: pd.DataFrame,
    period: Optional[int] = None,
    timeframe: Optional[Timeframe] = None,
) -> pd.Series:
    """Compute -DI (Negative Directional Indicator) using Ta-Lib wrapper"""
    # Determine period based on timeframe
    if timeframe is not None:
        from ztb.features.timeframe import get_timeframe_params

        tf_params = get_timeframe_params(timeframe)
        period = period or tf_params["medium_period"]
    else:
        period = period or 14

    high = df["high"]
    low = df["low"]
    close = df["close"]

    talib = TaLibWrapper()
    result = talib.minus_di(high, low, close, period)
    return pd.Series(result, index=df.index)


# === Multi-Timeframe ADX Features ===


@FeatureRegistry.register("ADX_M1")
def compute_adx_m1(df: pd.DataFrame) -> pd.Series:
    """ADX for 1-minute timeframe"""
    return compute_adx(df, timeframe=Timeframe.M1)


@FeatureRegistry.register("ADX_M5")
def compute_adx_m5(df: pd.DataFrame) -> pd.Series:
    """ADX for 5-minute timeframe"""
    return compute_adx(df, timeframe=Timeframe.M5)


@FeatureRegistry.register("ADX_M15")
def compute_adx_m15(df: pd.DataFrame) -> pd.Series:
    """ADX for 15-minute timeframe"""
    return compute_adx(df, timeframe=Timeframe.M15)


@FeatureRegistry.register("ADX_H1")
def compute_adx_h1(df: pd.DataFrame) -> pd.Series:
    """ADX for 1-hour timeframe"""
    return compute_adx(df, timeframe=Timeframe.H1)


@FeatureRegistry.register("ADX_H4")
def compute_adx_h4(df: pd.DataFrame) -> pd.Series:
    """ADX for 4-hour timeframe"""
    return compute_adx(df, timeframe=Timeframe.H4)


@FeatureRegistry.register("ADX_D1")
def compute_adx_d1(df: pd.DataFrame) -> pd.Series:
    """ADX for daily timeframe"""
    return compute_adx(df, timeframe=Timeframe.D1)


@FeatureRegistry.register("PlusDI_M1")
def compute_plus_di_m1(df: pd.DataFrame) -> pd.Series:
    """+DI for 1-minute timeframe"""
    return compute_plus_di(df, timeframe=Timeframe.M1)


@FeatureRegistry.register("PlusDI_M5")
def compute_plus_di_m5(df: pd.DataFrame) -> pd.Series:
    """+DI for 5-minute timeframe"""
    return compute_plus_di(df, timeframe=Timeframe.M5)


@FeatureRegistry.register("PlusDI_M15")
def compute_plus_di_m15(df: pd.DataFrame) -> pd.Series:
    """+DI for 15-minute timeframe"""
    return compute_plus_di(df, timeframe=Timeframe.M15)


@FeatureRegistry.register("PlusDI_H1")
def compute_plus_di_h1(df: pd.DataFrame) -> pd.Series:
    """+DI for 1-hour timeframe"""
    return compute_plus_di(df, timeframe=Timeframe.H1)


@FeatureRegistry.register("PlusDI_H4")
def compute_plus_di_h4(df: pd.DataFrame) -> pd.Series:
    """+DI for 4-hour timeframe"""
    return compute_plus_di(df, timeframe=Timeframe.H4)


@FeatureRegistry.register("PlusDI_D1")
def compute_plus_di_d1(df: pd.DataFrame) -> pd.Series:
    """+DI for daily timeframe"""
    return compute_plus_di(df, timeframe=Timeframe.D1)


@FeatureRegistry.register("MinusDI_M1")
def compute_minus_di_m1(df: pd.DataFrame) -> pd.Series:
    """-DI for 1-minute timeframe"""
    return compute_minus_di(df, timeframe=Timeframe.M1)


@FeatureRegistry.register("MinusDI_M5")
def compute_minus_di_m5(df: pd.DataFrame) -> pd.Series:
    """-DI for 5-minute timeframe"""
    return compute_minus_di(df, timeframe=Timeframe.M5)


@FeatureRegistry.register("MinusDI_M15")
def compute_minus_di_m15(df: pd.DataFrame) -> pd.Series:
    """-DI for 15-minute timeframe"""
    return compute_minus_di(df, timeframe=Timeframe.M15)


@FeatureRegistry.register("MinusDI_H1")
def compute_minus_di_h1(df: pd.DataFrame) -> pd.Series:
    """-DI for 1-hour timeframe"""
    return compute_minus_di(df, timeframe=Timeframe.H1)


@FeatureRegistry.register("MinusDI_H4")
def compute_minus_di_h4(df: pd.DataFrame) -> pd.Series:
    """-DI for 4-hour timeframe"""
    return compute_minus_di(df, timeframe=Timeframe.H4)


@FeatureRegistry.register("MinusDI_D1")
def compute_minus_di_d1(df: pd.DataFrame) -> pd.Series:
    """-DI for daily timeframe"""
    return compute_minus_di(df, timeframe=Timeframe.D1)
