import numpy as np
import pandas as pd
from ..base import BaseFeature

class Bollinger(BaseFeature):
    """Bollinger Bands"""

    def __init__(self):
        super().__init__("Bollinger", deps=["rolling_mean_20", "rolling_std_20"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy['bb_upper'] = df_copy['rolling_mean_20'] + 2 * df_copy['rolling_std_20']
        df_copy['bb_lower'] = df_copy['rolling_mean_20'] - 2 * df_copy['rolling_std_20']
        df_copy['bb_middle'] = df_copy['rolling_mean_20']
        df_copy['bb_width'] = np.where(
            df_copy['bb_middle'] != 0,
            (df_copy['bb_upper'] - df_copy['bb_lower']) / df_copy['bb_middle'],
            0
        )
        denominator = df_copy['bb_upper'] - df_copy['bb_lower']
        df_copy['bb_position'] = np.where(
            denominator != 0,
            (df_copy['close'] - df_copy['bb_lower']) / denominator,
            0.5  # neutral position if width is zero
        )
        return df_copy[['bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position']]