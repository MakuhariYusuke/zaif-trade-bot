"""
Chaikin AD (Accumulation/Distribution) feature implementation.
Chaikin AD is a volume-based indicator that measures the cumulative flow of money into and out of a security.

Output columns:
  - chaikin_ad: Chaikin AD values
"""

from typing import Any

import numpy as np
import pandas as pd

from ..base import BaseFeature
from ..registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper


@FeatureRegistry.register("Chaikin_AD")
def compute_chaikin_ad(df: pd.DataFrame) -> pd.Series:
    """Chaikin AD using Ta-Lib wrapper"""
    feature = ChaikinAD()
    result_df = feature.compute(df)
    return result_df["chaikin_ad"]


class ChaikinAD(BaseFeature):
    """
    Chaikin AD (Accumulation/Distribution) indicator.
    Measures the cumulative flow of money into and out of a security.
    """

    def __init__(self, **kwargs: Any):
        super().__init__("ChaikinAD", deps=["high", "low", "close", "volume"])

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close', 'volume'].
        Returns a DataFrame with Chaikin AD values.
        """
        # Calculate Chaikin AD using Ta-Lib wrapper
        chaikin_ad_values = TaLibWrapper.ad(
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            df["volume"].values.astype(np.float64)
        )

        result_df = pd.DataFrame({"chaikin_ad": chaikin_ad_values}, index=df.index)
        return result_df