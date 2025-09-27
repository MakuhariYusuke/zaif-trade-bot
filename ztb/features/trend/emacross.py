"""
EMACross feature implementation.
EMA/SMA cross signals for trend detection.

Output columns:
  - ema_sma_cross: Normalized difference between EMA and SMA
  - ema_above_sma: Binary indicator (1 if EMA > SMA, 0 otherwise)
"""

import pandas as pd
from typing import Dict, Any, Optional
from ..base import ParameterizedFeature


class EMACross(ParameterizedFeature):
    """
    EMA/SMA Cross signals for trend detection.
    """

    def __init__(self):
        super().__init__(
            "EMACross",
            deps=[],  # Will be set dynamically
            default_params={"fast_period": 5, "slow_period": 20}
        )

    def get_deps(self, params: Optional[Dict[str, Any]] = None):
        if params is None:
            params = self.default_params
        fast_period = params.get('fast_period', 5)
        slow_period = params.get('slow_period', 20)
        return [f"ema_{fast_period}", f"rolling_mean_{slow_period}"]

    def _compute_with_params(self, df: pd.DataFrame, **params: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute EMA/SMA cross signals with configurable periods.
        """
        # Get periods from params or defaults
        fast_period = int(params.get('fast_period', self.default_params['fast_period']))
        slow_period = int(params.get('slow_period', self.default_params['slow_period']))
        fast_col = f'ema_{fast_period}'
        slow_col = f'rolling_mean_{slow_period}'

        # Only compute EMA/SMA if not already present, and avoid overwriting
        if fast_col not in df.columns:
            df[fast_col] = df['close'].ewm(span=fast_period, adjust=False).mean().copy()
        if slow_col not in df.columns:
            df[slow_col] = df['close'].rolling(slow_period).mean().copy()

        # Prevent division by zero by replacing zeros with np.nan
        slow_col_safe = df[slow_col].replace(0, pd.NA)
        ema_sma_cross = (df[fast_col] - slow_col_safe) / slow_col_safe
        ema_above_sma = (df[fast_col] > df[slow_col]).astype(int)

        return pd.DataFrame({
            'ema_sma_cross': ema_sma_cross,
            'ema_above_sma': ema_above_sma
        })