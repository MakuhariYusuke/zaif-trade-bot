import numpy as np
import pandas as pd
from ..base import BaseFeature

class CCI(BaseFeature):
    """Commodity Channel Index"""

    def __init__(self):
        super().__init__("CCI", deps=["rolling_mean_20"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df_copy = df.copy()
        # Mean deviation
        mean_dev = (df_copy['close'] - df_copy['rolling_mean_20']).abs().rolling(20).mean()
        df_copy['cci'] = (df_copy['close'] - df_copy['rolling_mean_20']) / (0.015 * mean_dev)
        df_copy['cci'] = df_copy['cci'].fillna(0)
        return df_copy[['cci']]