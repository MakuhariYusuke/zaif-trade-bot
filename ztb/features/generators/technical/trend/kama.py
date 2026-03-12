"""
KAMA (Kaufman's Adaptive Moving Average) feature implementation.
Adaptive moving average that adjusts to market volatility.

Output columns:
  - kama
"""

from typing import Any

import numpy as np
import pandas as pd

from ztb.features.base import ComputableFeature, MovingAverageFeature
from ztb.features.registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper

@FeatureRegistry.register("KAMA")
def compute_kama(df: pd.DataFrame) -> pd.Series:
    """Kaufman's Adaptive Moving Average using Ta-Lib wrapper"""
    feature = KAMA()
    result_df = feature.compute(df)
    return result_df["kama"]

class KAMA(MovingAverageFeature, ComputableFeature):
    """
    Kaufman's Adaptive Moving Average.
    Adjusts smoothing based on market efficiency ratio.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("KAMA", deps=["close"])

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with KAMA values.
        """
        # Use Ta-Lib wrapper for KAMA calculation
        result = TaLibWrapper.kama(df["close"].values.astype(np.float64))

        return pd.DataFrame({"kama": result})
