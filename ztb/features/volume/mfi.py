"""
MFI (Money Flow Index) implementation.
MFIの実装
"""

import pandas as pd

from ztb.features.registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper


@FeatureRegistry.register("MFI")
def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute MFI (Money Flow Index) using Ta-Lib wrapper"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    result = TaLibWrapper.mfi(
        high.to_numpy(), low.to_numpy(), close.to_numpy(), volume.to_numpy(), period
    )
    return pd.Series(result, index=df.index)
