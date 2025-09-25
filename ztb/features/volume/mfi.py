"""
MFI (Money Flow Index) feature implementation.
Volume-based momentum oscillator.

Parameters:
  - period: MFI calculation period (default=14)
Output columns:
  - mfi_{period}
"""

import pandas as pd
from ..base import BaseFeature


class MFI(BaseFeature):
    """
    Money Flow Index - volume-based momentum oscillator.
    """

    def __init__(self, period: int = 14, **kwargs):
        super().__init__("MFI", deps=["high", "low", "close", "volume"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close', 'volume'].
        Returns a DataFrame with MFI values.
        """
        # Typical Price
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Raw Money Flow
        money_flow = typical_price * df['volume']

        # Positive and Negative Money Flow
        price_diff = typical_price.diff()
        positive_flow = money_flow.where(price_diff > 0, 0.0)
        negative_flow = money_flow.where(price_diff < 0, 0.0)

        # Money Flow Ratio
        pos_mf_sum = positive_flow.rolling(self.period).sum()
        neg_mf_sum = negative_flow.rolling(self.period).sum()

        money_flow_ratio = pos_mf_sum / (neg_mf_sum + 1e-8)  # Avoid division by zero

        # MFI calculation
        mfi = 100 - (100 / (1 + money_flow_ratio))

        return pd.DataFrame({f'mfi_{self.period}': mfi.fillna(50)})