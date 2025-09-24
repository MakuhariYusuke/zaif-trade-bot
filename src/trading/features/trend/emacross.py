"""
EMACross feature implementation.
EMA/SMA cross signals for trend detection.

Output columns:
  - ema_sma_cross: Normalized difference between EMA and SMA
  - ema_above_sma: Binary indicator (1 if EMA > SMA, 0 otherwise)
"""

import pandas as pd
from ..base import BaseFeature


class EMACross(BaseFeature):
    """
    EMA/SMA Cross signals for trend detection.
    """

    def __init__(self, **kwargs):
        super().__init__("EMACross", deps=["ema_5", "rolling_mean_20"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['ema_5', 'rolling_mean_20'].
        Returns a DataFrame with EMA/SMA cross signals.
        """
        ema_sma_cross = (df['ema_5'] - df['rolling_mean_20']) / df['rolling_mean_20']
        ema_above_sma = (df['ema_5'] > df['rolling_mean_20']).astype(int)

        return pd.DataFrame({
            'ema_sma_cross': ema_sma_cross,
            'ema_above_sma': ema_above_sma
        })