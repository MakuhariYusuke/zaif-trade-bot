"""
MFI (Money Flow Index) feature implementation.
MFI is a momentum indicator that uses both price and volume to measure buying and selling pressure.

Output columns:
  - mfi: MFI value (0 to 100)
"""

from typing import Any

import pandas as pd

from ..base import BaseFeature
from ..registry import FeatureRegistry

@FeatureRegistry.register("MFI")
def compute_mfi(df: pd.DataFrame) -> pd.Series:
    """MFI (Money Flow Index)"""
    feature = MFI()
    result_df = feature.compute(df)
    return result_df["mfi"]

class MFI(BaseFeature):
    """
    Money Flow Index (MFI).
    Uses both price and volume to measure buying and selling pressure.
    """

    def __init__(self, period: int = 14, **kwargs: Any):
        super().__init__("MFI", deps=["high", "low", "close", "volume"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close', 'volume'].
        Returns a DataFrame with MFI values.
        """
        period = params.get("period", self.period)

        # Calculate Typical Price
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        # Calculate Raw Money Flow
        money_flow = typical_price * df["volume"]

        # Calculate Money Flow Direction
        price_change = typical_price.diff()
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)

        positive_mask = price_change > 0
        negative_mask = price_change < 0

        positive_flow[positive_mask] = money_flow[positive_mask]
        negative_flow[negative_mask] = money_flow[negative_mask]

        # Calculate Money Flow Ratio
        positive_mf_sum = positive_flow.rolling(window=period).sum()
        negative_mf_sum = negative_flow.rolling(window=period).sum()

        money_flow_ratio = positive_mf_sum / negative_mf_sum

        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))

        return pd.DataFrame({"mfi": mfi}, index=df.index)
