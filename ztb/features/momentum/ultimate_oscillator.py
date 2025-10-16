"""
Ultimate Oscillator feature implementation.
The Ultimate Oscillator is a momentum oscillator designed to capture momentum across three different timeframes.

Output columns:
  - ultimate_oscillator: Ultimate Oscillator value (0 to 100)
"""

from typing import Any

import numpy as np
import pandas as pd

from ..base import BaseFeature
from ..registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper


@FeatureRegistry.register("Ultimate_Oscillator")
def compute_ultimate_oscillator(df: pd.DataFrame) -> pd.Series:
    """Ultimate Oscillator using Ta-Lib wrapper"""
    feature = UltimateOscillator()
    result_df = feature.compute(df)
    return result_df["ultimate_oscillator"]


class UltimateOscillator(BaseFeature):
    """
    Ultimate Oscillator indicator.
    Combines short-term, intermediate-term, and long-term momentum into one oscillator.
    """

    def __init__(self, short_period: int = 7, medium_period: int = 14, long_period: int = 28, **kwargs: Any):
        super().__init__("UltimateOscillator", deps=["high", "low", "close"])
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Ultimate Oscillator values.
        """
        short_period = params.get("short_period", self.short_period)
        medium_period = params.get("medium_period", self.medium_period)
        long_period = params.get("long_period", self.long_period)

        # Use Ta-Lib wrapper for Ultimate Oscillator calculation
        result = TaLibWrapper.ultimate_oscillator(
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            short_period,
            medium_period,
            long_period
        )

        result_df = pd.DataFrame({"ultimate_oscillator": result}, index=df.index)
        return result_df