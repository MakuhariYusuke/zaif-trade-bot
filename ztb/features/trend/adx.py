"""
ADX (Average Directional Index) implementation.
トレンド強度を測定する方向性移動指標
"""

import numpy as np
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
        high.values.astype(np.float64), low.values.astype(np.float64), close.values.astype(np.float64), period
    )
    return pd.Series(result, index=df.index)


@FeatureRegistry.register("PlusDI")
def compute_plus_di(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute +DI (Positive Directional Indicator) using Ta-Lib wrapper"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    result = TaLibWrapper.plus_di(
        high.values.astype(np.float64), low.values.astype(np.float64), close.values.astype(np.float64), period
    )
    return pd.Series(result, index=df.index)


@FeatureRegistry.register("MinusDI")
def compute_minus_di(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute -DI (Negative Directional Indicator) using Ta-Lib wrapper"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    result = TaLibWrapper.minus_di(
        high.values.astype(np.float64), low.values.astype(np.float64), close.values.astype(np.float64), period
    )
    return pd.Series(result, index=df.index)
