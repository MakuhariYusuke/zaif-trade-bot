import numpy as np
import pandas as pd
from ..base import BaseFeature

class Stochastic(BaseFeature):
    """Stochastic Oscillator"""

    def __init__(self):
        super().__init__("Stochastic", deps=["rolling_max_14", "rolling_min_14"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df_copy = df.copy()
        denominator = df_copy['rolling_max_14'] - df_copy['rolling_min_14']
        df_copy['stoch_k'] = np.where(
            denominator != 0,
            100 * (df_copy['close'] - df_copy['rolling_min_14']) / denominator,
            50  # neutral value when denominator is zero
        )
        df_copy['stoch_k'] = df_copy['stoch_k'].fillna(50)  # neutral
        df_copy['stoch_d'] = df_copy['stoch_k'].rolling(3).mean()
        return df_copy[['stoch_k', 'stoch_d']]