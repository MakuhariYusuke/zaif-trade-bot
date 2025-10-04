"""
EMACross feature implementation.
EMA/SMA cross signals for trend detection.

Output columns:
  - ema_sma_cross: Normalized difference between EMA and SMA
  - ema_above_sma: Binary indicator (1 if EMA > SMA, 0 otherwise)
"""

from typing import Any, Dict, List, Optional, cast

import pandas as pd

from ..base import ParameterizedFeature
from ..registry import FeatureRegistry


@FeatureRegistry.register("EMACross_Diff")
def compute_ema_cross_diff(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Difference (normalized)"""
    feature = EMACross()
    result_df = feature.compute(df)
    return result_df["ema_sma_cross"]


@FeatureRegistry.register("EMACross_Signal")
def compute_ema_cross_signal(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Signal (1 if EMA > SMA, 0 otherwise)"""
    feature = EMACross()
    result_df = feature.compute(df)
    return result_df["ema_above_sma"]


class EMACross(ParameterizedFeature):
    """
    EMA/SMA Cross signals for trend detection.
    """

    def __init__(self) -> None:
        super().__init__(
            "EMACross",
            deps=[],  # Will be set dynamically
            default_params={"fast_period": 5, "slow_period": 20},
        )

    def get_deps(self, params: Optional[Dict[str, Any]] = None) -> List[str]:
        if params is None:
            params = self.default_params
        fast_period = params.get("fast_period", 5)
        slow_period = params.get("slow_period", 20)
        return [f"ema_{fast_period}", f"rolling_mean_{slow_period}"]

    def _compute_with_params(
        self, df: pd.DataFrame, **params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Compute EMA/SMA cross signals with configurable periods.
        """
        fast_period = cast(
            int, params.get("fast_period", self.default_params["fast_period"])
        )
        slow_period = cast(
            int, params.get("slow_period", self.default_params["slow_period"])
        )
        fast_col = f"ema_{fast_period}"
        slow_col = f"rolling_mean_{slow_period}"

        # Only compute EMA/SMA if not already present, and avoid overwriting
        if fast_col not in df.columns:
            df[fast_col] = df["close"].ewm(span=fast_period, adjust=False).mean().copy()
        if slow_col not in df.columns:
            df[slow_col] = df["close"].rolling(slow_period).mean().copy()

        # Prevent division by zero by replacing zeros with np.nan
        slow_col_safe = df[slow_col].replace(0, pd.NA)
        ema_sma_cross = (df[fast_col] - slow_col_safe) / slow_col_safe
        ema_above_sma = (df[fast_col] > df[slow_col]).astype(int)

        return pd.DataFrame(
            {"ema_sma_cross": ema_sma_cross, "ema_above_sma": ema_above_sma}
        )
