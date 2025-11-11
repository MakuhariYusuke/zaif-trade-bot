"""
Chaikin Volatility feature implementation.
Chaikin Volatility measures the rate of change of the security's trading range (high - low).

Output columns:
  - chaikin_volatility: Chaikin Volatility value
"""

from typing import Any

import numpy as np
import pandas as pd

from ztb.utils.talib_wrapper import TaLibWrapper

from ..base import BaseFeature
from ..registry import FeatureRegistry


@FeatureRegistry.register("Chaikin_Volatility")
def compute_chaikin_volatility(df: pd.DataFrame) -> pd.Series:
    """Chaikin Volatility using Ta-Lib wrapper"""
    feature = ChaikinVolatility()
    result_df = feature.compute(df)
    return result_df["chaikin_volatility"]


class ChaikinVolatility(BaseFeature):
    """
    Chaikin Volatility indicator.
    Measures the rate of change of the security's trading range over a specified period.
    """

    def __init__(self, period: int = 10, **kwargs: Any):
        super().__init__("ChaikinVolatility", deps=["high", "low"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low'].
        Returns a DataFrame with Chaikin Volatility values.
        """
        period = params.get("period", self.period)

        # Calculate high-low range
        hl_range = df["high"] - df["low"]

        # Use EMA of the range and calculate rate of change
        ema_range = hl_range.ewm(span=period, adjust=False).mean()
        ema_roc = TaLibWrapper.roc(ema_range.values.astype(np.float64), period)

        result_df = pd.DataFrame({"chaikin_volatility": ema_roc}, index=df.index)
        return result_df
