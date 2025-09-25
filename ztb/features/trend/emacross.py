"""
EMACross feature implementation.
EMA/SMA cross signals for trend detection.

Output columns:
  - ema_sma_cross: Normalized difference between EMA and SMA
  - ema_above_sma: Binary indicator (1 if EMA > SMA, 0 otherwise)
"""

import pandas as pd
from ..base import ParameterizedFeature


class EMACross(ParameterizedFeature):
    """
    EMA/SMA Cross signals for trend detection.
    """

    def __init__(self, **kwargs):
        super().__init__(
            "EMACross",
            deps=["ema_5", "rolling_mean_20"],
            default_params={"fast_period": 5, "slow_period": 20}
        )

    def _compute_with_params(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        Compute EMA/SMA cross signals with configurable periods.
        """
        fast_period = params.get('fast_period', 5)
        slow_period = params.get('slow_period', 20)

        # Use pre-calculated EMAs if available, otherwise calculate
        fast_col = f'ema_{fast_period}'
        slow_col = f'rolling_mean_{slow_period}'

        if fast_col not in df.columns:
            df[fast_col] = df['close'].ewm(span=fast_period).mean()
        if slow_col not in df.columns:
            df[slow_col] = df['close'].rolling(slow_period).mean()

        ema_sma_cross = (df[fast_col] - df[slow_col]) / df[slow_col]
        ema_above_sma = (df[fast_col] > df[slow_col]).astype(int)

        return pd.DataFrame({
            'ema_sma_cross': ema_sma_cross,
            'ema_above_sma': ema_above_sma
        })