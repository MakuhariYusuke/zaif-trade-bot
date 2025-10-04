"""
Williams %R feature implementation.
Williams %R is a momentum indicator that measures overbought and oversold levels.

Output columns:
  - williams_r: Williams %R value (-100 to 0)
"""

from typing import Any

import pandas as pd

from ..base import BaseFeature
from ..registry import FeatureRegistry


@FeatureRegistry.register("Williams_R")
def compute_williams_r(df: pd.DataFrame) -> pd.Series:
    """Williams %R (Williams Percent Range)"""
    feature = WilliamsR()
    result_df = feature.compute(df)
    return result_df["williams_r"]


class WilliamsR(BaseFeature):
    """
    Williams %R indicator.
    Measures the level of the close relative to the highest high for the look-back period.
    """

    def __init__(self, period: int = 14, **kwargs: Any):
        super().__init__("WilliamsR", deps=["high", "low", "close"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Williams %R values.
        """
        period = params.get("period", self.period)

        # Calculate highest high and lowest low over the period
        highest_high = df["high"].rolling(window=period).max()
        lowest_low = df["low"].rolling(window=period).min()

        # Calculate Williams %R
        # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        williams_r = ((highest_high - df["close"]) / (highest_high - lowest_low)) * -100

        return pd.DataFrame({"williams_r": williams_r}, index=df.index)
