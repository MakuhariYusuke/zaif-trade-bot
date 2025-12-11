"""
Stochastic Oscillator implementation.
Stochastic Oscillatorの実装
"""

import numpy as np
import pandas as pd

from ztb.features.core.registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper


@FeatureRegistry.register("Stochastic")
def compute_stochastic(
    df: pd.DataFrame, period: int = 14, smooth_k: int = 3
) -> pd.Series:
    """Compute Stochastic Oscillator with Ta-Lib support"""
    high_prices = np.asarray(df["high"].values, dtype=float)
    low_prices = np.asarray(df["low"].values, dtype=float)
    close_prices = np.asarray(df["close"].values, dtype=float)

    slowk, slowd = TaLibWrapper.stoch(
        high_prices,
        low_prices,
        close_prices,
        fastk_period=period,
        slowk_period=smooth_k,
        slowd_period=smooth_k,
    )

    # Return %D (slowd) as it's the smoothed signal line
    return pd.Series(slowd, index=df.index).fillna(50)


@FeatureRegistry.register("Stochastic_K")
def compute_stochastic_k(
    df: pd.DataFrame, period: int = 14, smooth_k: int = 3
) -> pd.Series:
    """Compute Stochastic Oscillator %K with Ta-Lib support"""
    high_prices = np.asarray(df["high"].values, dtype=float)
    low_prices = np.asarray(df["low"].values, dtype=float)
    close_prices = np.asarray(df["close"].values, dtype=float)

    slowk, slowd = TaLibWrapper.stoch(
        high_prices,
        low_prices,
        close_prices,
        fastk_period=period,
        slowk_period=smooth_k,
        slowd_period=smooth_k,
    )

    return pd.Series(slowk, index=df.index).fillna(50)
