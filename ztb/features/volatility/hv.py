import numpy as np
import pandas as pd
from ..base import BaseFeature

class HVFeature(BaseFeature):
    """
    Historical Volatility (HV) feature implementation.
    Measures the volatility of price returns over a specified period.

    Parameters:
      - period: HV calculation period (default=14)
    Output columns:
      - hv_{period}
    """
    def __init__(self, period: int = 14, **kwargs):
        super().__init__("HVFeature", deps=["close"])
        self.period = period
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with Historical Volatility values.
        """
        hv_df = pd.DataFrame(index=df.index)
        log_returns = np.log(df['close'] / df['close'].shift(1))
        hv = log_returns.rolling(window=self.period).std() * np.sqrt(252)  # Annualized volatility
        hv_df[f'hv_{self.period}'] = hv
        return hv_df