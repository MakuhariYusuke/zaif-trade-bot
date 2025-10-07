"""
ADX (Average Directional Index) implementation.
トレンド強度を測定する方向性移動指標
"""

import pandas as pd

from ztb.features.registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper


@FeatureRegistry.register("ADX")
def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX (Average Directional Index) using Ta-Lib wrapper"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    result = TaLibWrapper.adx(
        high.to_numpy(), low.to_numpy(), close.to_numpy(), period
    )
    return pd.Series(result, index=df.index)


@FeatureRegistry.register("PlusDI")
def compute_plus_di(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute +DI (Positive Directional Indicator) using Ta-Lib wrapper"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    result = TaLibWrapper.plus_di(
        high.to_numpy(), low.to_numpy(), close.to_numpy(), period
    )
    return pd.Series(result, index=df.index)


@FeatureRegistry.register("MinusDI")
def compute_minus_di(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute -DI (Negative Directional Indicator) using Ta-Lib wrapper"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    result = TaLibWrapper.minus_di(
        high.to_numpy(), low.to_numpy(), close.to_numpy(), period
    )
    return pd.Series(result, index=df.index)
