import numpy as np
import pandas as pd
from ..base import BaseFeature

class ReturnStdDev(BaseFeature):
    """
    Standard Deviation of Returns feature implementation.
    Measures the volatility of returns over a specified period.

    Parameters:
      - period: StdDev calculation period (default=20)
    Output columns:
      - return_stddev_{period}
    """
    def __init__(self, period: int = 20, **kwargs):
        super().__init__("ReturnStdDev", deps=["close"])
        self.period = period
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with standard deviation of returns.
        """
        ret_std_df = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change()
        ret_std = returns.rolling(window=self.period).std()
        ret_std_df[f'return_stddev_{self.period}'] = ret_std
        return ret_std_df