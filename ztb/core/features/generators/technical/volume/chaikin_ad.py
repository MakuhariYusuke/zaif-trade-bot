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
        # Calculate Money Flow Multiplier
        # MFM = [(Close - Low) - (High - Close)] / (High - Low)
        money_flow_multiplier = (
            (df["close"] - df["low"]) - (df["high"] - df["close"])
        ) / (df["high"] - df["low"])

        # Handle division by zero
        money_flow_multiplier = money_flow_multiplier.replace(
            [np.inf, -np.inf], 0
        ).fillna(0)

        # Calculate Money Flow Volume
        # MFV = MFM * Volume
        money_flow_volume = money_flow_multiplier * df["volume"]

        # Calculate Chaikin AD (cumulative sum of MFV)
        chaikin_ad = money_flow_volume.cumsum()

        return pd.DataFrame({"chaikin_ad": chaikin_ad}, index=df.index)
        return result_df
