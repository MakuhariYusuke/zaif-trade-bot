"""
SMA (Simple Moving Average) implementation.
単純移動平均線 - トレンド指標
"""

import numpy as np
import pandas as pd

from ztb.features.core.registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper

@FeatureRegistry.register("SMA")
def compute_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute Simple Moving Average (SMA) with Ta-Lib support"""
    close_prices = np.asarray(df["close"].values, dtype=float)

    # Use TaLibWrapper for consistency
    talib = TaLibWrapper()
    sma_values = talib.sma(close_prices, period)

    return pd.Series(sma_values, index=df.index)
