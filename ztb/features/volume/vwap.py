import numpy as np
import pandas as pd
from ..base import BaseFeature

class VWAP(BaseFeature):
    """
    VWAP (Volume Weighted Average Price) feature implementation.
    Provides the average price weighted by volume.

    Output columns:
      - vwap
    """
    def __init__(self, **kwargs):
        super().__init__("VWAP", deps=["close", "volume"])
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close', 'volume'].
        Returns a DataFrame with VWAP values.
        """
        vwap_df = pd.DataFrame(index=df.index)
        cum_vol_price = (df['close'] * df['volume']).cumsum()
        cum_volume = df['volume'].cumsum() + 1e-8  # Avoid division by zero
        vwap = cum_vol_price / cum_volume
        vwap_df['vwap'] = vwap
        return vwap_df