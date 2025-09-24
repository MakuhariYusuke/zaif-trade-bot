import numpy as np
import pandas as pd
from ..base import BaseFeature

class MACD(BaseFeature):
    """MACD (Moving Average Convergence Divergence)"""

    def __init__(self):
        super().__init__("MACD", deps=["ema_12", "ema_26", "ema_9"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy['macd'] = df_copy['ema_12'] - df_copy['ema_26']
        df_copy['macd_signal'] = df_copy['ema_9']
        df_copy['macd_hist'] = df_copy['macd'] - df_copy['macd_signal']
        return df_copy[['macd', 'macd_signal', 'macd_hist']]