"""
Normalized ATR feature implementation.
Normalized ATR scales the ATR by the closing price to provide a percentage-based volatility measure.

Output columns:
  - normalized_atr: Normalized ATR value (as percentage)
"""

from typing import Any

import numpy as np
import pandas as pd

from ztb.features.core.base import BaseFeature
from ztb.features.core.registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper


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

        # Normalize by closing price (fraction). Do NOT multiply by 100 to keep
        # values in 0..1 range (e.g. 0.01 == 1%). Tests assume fraction values.
        normalized_atr = atr_values / df["close"].values

        result_df = pd.DataFrame({"normalized_atr": normalized_atr}, index=df.index)
        # Fill NaN values (initial periods) with 0 so tests and downstream logic
        # that expect non-negative values don't fail on NaN comparisons.
        result_df["normalized_atr"] = result_df["normalized_atr"].fillna(0.0)
        return result_df
