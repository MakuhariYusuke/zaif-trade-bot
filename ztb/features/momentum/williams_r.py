"""
Williams %R feature implementation.
Williams %R is a momentum indicator that measures overbought and oversold levels.

Output columns:
  - williams_r: Williams %R value (-100 to 0)
"""

from typing import Any

import numpy as np
import pandas as pd

from ztb.utils.talib_wrapper import TaLibWrapper

from ..base import BaseFeature
from ..registry import FeatureRegistry


@FeatureRegistry.register("Williams_R")
def compute_williams_r(df: pd.DataFrame) -> pd.Series:
    """Williams %R (Williams Percent Range) using Ta-Lib wrapper"""
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

        # Use Ta-Lib wrapper for Williams %R calculation
        result = TaLibWrapper.williams_r(
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            period,
        )

        return pd.DataFrame({"williams_r": result}, index=df.index)
