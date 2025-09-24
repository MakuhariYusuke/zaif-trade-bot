import numpy as np
import pandas as pd
from ..base import BaseFeature

class ReturnMA(BaseFeature):
    """
    Moving Averages of Returns feature implementation.
    Computes short-term and medium-term moving averages of returns.
    Parameters:
      - short_period: Short-term MA period (default=5)
        - medium_period: Medium-term MA period (default=20)
    Output columns:
        - return_ma_short
        - return_ma_medium
    """
    def __init__(self, short_period: int = 5, medium_period: int = 20, **kwargs):
        super().__init__("ReturnMA", deps=["close"])
        self.short_period = short_period
        self.medium_period = medium_period
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with moving averages of returns.
        """
        ret_ma_df = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change()
        ret_ma_df['return_ma_short'] = returns.rolling(window=self.short_period).mean()
        ret_ma_df['return_ma_medium'] = returns.rolling(window=self.medium_period).mean()
        return ret_ma_df