"""
Parabolic SAR feature implementation.
Parabolic SAR (Stop and Reverse) is a trend-following indicator.

Output columns:
  - psar: Parabolic SAR value
  - psar_trend: Trend direction (1 for uptrend, -1 for downtrend)
  - psar_acceleration: Current acceleration factor
"""

from typing import Any

import numpy as np
import pandas as pd

from ..base import BaseFeature
from ..registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper


@FeatureRegistry.register("PSAR")
def compute_psar(df: pd.DataFrame) -> pd.Series:
    """Parabolic SAR (Stop and Reverse)"""
    feature = ParabolicSAR()
    result_df = feature.compute(df)
    return result_df["psar"]


@FeatureRegistry.register("PSAR_Trend")
def compute_psar_trend(df: pd.DataFrame) -> pd.Series:
    """Parabolic SAR Trend Direction (1=uptrend, -1=downtrend)"""
    feature = ParabolicSAR()
    result_df = feature.compute(df)
    return result_df["psar_trend"]


class ParabolicSAR(BaseFeature):
    """
    Parabolic SAR (Stop and Reverse) indicator.
    """

    def __init__(
        self, acceleration: float = 0.02, max_acceleration: float = 0.2, **kwargs: Any
    ):
        super().__init__("ParabolicSAR", deps=["high", "low", "close"])
        self.acceleration = acceleration
        self.max_acceleration = max_acceleration

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Parabolic SAR values.
        """
        acceleration = params.get("acceleration", self.acceleration)
        max_acceleration = params.get("max_acceleration", self.max_acceleration)

        # Use Ta-Lib wrapper for Parabolic SAR calculation
        psar = TaLibWrapper.sar(df["high"].values.astype(np.float64), df["low"].values.astype(np.float64), acceleration, max_acceleration)

        # Calculate trend: 1 if PSAR < close (uptrend), -1 if PSAR > close (downtrend)
        trend = np.where(psar < df["close"], 1, -1)

        # Acceleration factor is fixed for simplicity (Ta-Lib doesn't provide it)
        acceleration_factor = np.full(len(psar), acceleration)

        return pd.DataFrame(
            {
                "psar": psar,
                "psar_trend": trend,
                "psar_acceleration": acceleration_factor,
            },
            index=df.index,
        )
