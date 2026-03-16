"""
MACD (Moving Average Convergence Divergence) implementation.
MACDの実装
"""

import numpy as np
import pandas as pd

from ztb.features.core.registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper

_talib = TaLibWrapper()

@FeatureRegistry.register("MACD")
def compute_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    """Compute MACD (Moving Average Convergence Divergence) - Optimized version with Ta-Lib support"""
    close_prices = np.asarray(df["close"].values)

    # Use Ta-Lib if available, otherwise use custom implementation
    macd, signal, hist = _talib.macd(
        close_prices, fast_period, slow_period, signal_period
    )

    # Return MACD histogram (MACD - Signal) as it's the most useful component
    macd_hist = pd.Series(hist, index=df.index).fillna(0)

    return macd_hist
