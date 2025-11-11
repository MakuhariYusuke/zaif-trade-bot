"""
Normalized ATR feature implementation.
Normalized ATR scales the ATR by the closing price to provide a percentage-based volatility measure.

Output columns:
  - normalized_atr: Normalized ATR value (as percentage)
"""

from typing import Any

import numpy as np
import pandas as pd

from ztb.utils.talib_wrapper import TaLibWrapper

from ..base import BaseFeature
from ..registry import FeatureRegistry


@FeatureRegistry.register("Normalized_ATR")
def compute_normalized_atr(df: pd.DataFrame) -> pd.Series:
    """Normalized ATR using Ta-Lib wrapper"""
    feature = NormalizedATR()
    result_df = feature.compute(df)
    return result_df["normalized_atr"]


class NormalizedATR(BaseFeature):
    """
    Normalized ATR indicator.
    ATR normalized by closing price to provide percentage-based volatility measure.
    """

    def __init__(self, period: int = 14, **kwargs: Any):
        super().__init__("NormalizedATR", deps=["high", "low", "close"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Normalized ATR values.
        """
        period = params.get("period", self.period)

        # Calculate ATR using Ta-Lib wrapper
        atr_values = TaLibWrapper.atr(
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            period,
        )

        # Normalize by closing price (as percentage)
        normalized_atr = (atr_values / df["close"].values) * 100

        result_df = pd.DataFrame({"normalized_atr": normalized_atr}, index=df.index)
        return result_df
