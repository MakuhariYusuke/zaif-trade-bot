import numpy as np
import pandas as pd
from ..base import BaseFeature

class PriceVolumeCorr(BaseFeature):
    """
    Price-Volume Correlation feature implementation.
    Measures the rolling correlation between price and volume.
    Parameters:
      - period: Correlation calculation period (default=20)
      Output columns:
        - price_volume_corr_{period}
    """
    def __init__(self, period: int = 20, **kwargs):
        super().__init__("PriceVolumeCorr", deps=["close", "volume"])
        self.period = period
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close', 'volume'].
        Returns a DataFrame with price-volume correlation.
        """
        corr_df = pd.DataFrame(index=df.index)
        price = df['close']
        volume = df['volume']
        corr = price.rolling(window=self.period).corr(volume)
        corr_df[f'price_volume_corr_{self.period}'] = corr
        return corr_df