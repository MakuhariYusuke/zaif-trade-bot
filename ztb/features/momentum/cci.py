"""
CCI (Commodity Channel Index) implementation.
CCIの実装
"""

import numpy as np
import pandas as pd

from ztb.features.registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper


@FeatureRegistry.register("CCI")
def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute CCI (Commodity Channel Index) using Ta-Lib wrapper"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    result = TaLibWrapper.cci(
        high.values.astype(np.float64),
        low.values.astype(np.float64),
        close.values.astype(np.float64),
        period,
    )
    return pd.Series(result, index=df.index)
