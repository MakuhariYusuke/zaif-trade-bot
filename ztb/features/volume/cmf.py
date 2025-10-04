"""
Chaikin Money Flow (CMF) feature implementation.
Chaikin Money Flow measures the amount of money flow volume over a period.

Output columns:
  - cmf: Chaikin Money Flow value
"""

from typing import Any

import pandas as pd

from ..base import BaseFeature
from ..registry import FeatureRegistry


@FeatureRegistry.register("CMF")
def compute_cmf(df: pd.DataFrame) -> pd.Series:
    """Chaikin Money Flow (CMF)"""
    feature = ChaikinMoneyFlow()
    result_df = feature.compute(df)
    return result_df["cmf"]


class ChaikinMoneyFlow(BaseFeature):
    """
    Chaikin Money Flow (CMF) indicator.
    Combines price and volume to measure buying and selling pressure.
    """

    def __init__(self, period: int = 21, **kwargs: Any):
        super().__init__("ChaikinMoneyFlow", deps=["high", "low", "close", "volume"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close', 'volume'].
        Returns a DataFrame with CMF values.
        """
        period = params.get("period", self.period)

        # Calculate Money Flow Multiplier
        # MFM = ((Close - Low) - (High - Close)) / (High - Low)
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"]
        )

        # Calculate Money Flow Volume
        # MFV = MFM * Volume
        mfv = mfm * df["volume"]

        # Calculate Chaikin Money Flow
        # CMF = MFV 21-period sum / Volume 21-period sum
        mfv_sum = mfv.rolling(window=period).sum()
        volume_sum = df["volume"].rolling(window=period).sum()
        cmf = mfv_sum / volume_sum

        return pd.DataFrame({"cmf": cmf}, index=df.index)
