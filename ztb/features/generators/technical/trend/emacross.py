"""
EMACross feature implementation.
EMA/SMA cross signals for trend detection with multi-timeframe support.

Output columns:
  - ema_sma_cross: Normalized difference between EMA and SMA
  - ema_above_sma: Binary indicator (1 if EMA > SMA, 0 otherwise)
"""

from typing import Any, cast

import numpy as np
import pandas as pd

from ..base import ParameterizedFeature
from ..registry import FeatureRegistry
from ..timeframe import Timeframe

@FeatureRegistry.register("EMACross_Diff")
def compute_ema_cross_diff(
    df: pd.DataFrame, timeframe: Timeframe | None = None
) -> pd.Series:
    """EMA/SMA Cross Difference (normalized)"""
    feature = EMACross()
    if timeframe is not None:
        # Adjust periods based on timeframe
        from ..timeframe import get_timeframe_params

        tf_params = get_timeframe_params(timeframe)
        feature.default_params = {
            "fast_period": tf_params["short_period"] // 4,  # EMA period
            "slow_period": tf_params["medium_period"] // 4,  # SMA period
        }
    result_df = feature.compute(df)
    return result_df["ema_sma_cross"]

@FeatureRegistry.register("EMACross_Signal")
def compute_ema_cross_signal(
    df: pd.DataFrame, timeframe: Timeframe | None = None
) -> pd.Series:
    """EMA/SMA Cross Signal (1 if EMA > SMA, 0 otherwise)"""
    feature = EMACross()
    if timeframe is not None:
        # Adjust periods based on timeframe
        from ..timeframe import get_timeframe_params

        tf_params = get_timeframe_params(timeframe)
        feature.default_params = {
            "fast_period": tf_params["short_period"] // 4,  # EMA period
            "slow_period": tf_params["medium_period"] // 4,  # SMA period
        }
    result_df = feature.compute(df)
    return result_df["ema_above_sma"]

# === Multi-Timeframe EMACross Features ===

@FeatureRegistry.register("EMACross_Diff_M1")
def compute_ema_cross_diff_m1(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Difference for 1-minute timeframe"""
    return compute_ema_cross_diff(df, timeframe=Timeframe.M1)

@FeatureRegistry.register("EMACross_Diff_M5")
def compute_ema_cross_diff_m5(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Difference for 5-minute timeframe"""
    return compute_ema_cross_diff(df, timeframe=Timeframe.M5)

@FeatureRegistry.register("EMACross_Diff_M15")
def compute_ema_cross_diff_m15(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Difference for 15-minute timeframe"""
    return compute_ema_cross_diff(df, timeframe=Timeframe.M15)

@FeatureRegistry.register("EMACross_Diff_H1")
def compute_ema_cross_diff_h1(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Difference for 1-hour timeframe"""
    return compute_ema_cross_diff(df, timeframe=Timeframe.H1)

@FeatureRegistry.register("EMACross_Diff_H4")
def compute_ema_cross_diff_h4(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Difference for 4-hour timeframe"""
    return compute_ema_cross_diff(df, timeframe=Timeframe.H4)

@FeatureRegistry.register("EMACross_Diff_D1")
def compute_ema_cross_diff_d1(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Difference for daily timeframe"""
    return compute_ema_cross_diff(df, timeframe=Timeframe.D1)

@FeatureRegistry.register("EMACross_Signal_M1")
def compute_ema_cross_signal_m1(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Signal for 1-minute timeframe"""
    return compute_ema_cross_signal(df, timeframe=Timeframe.M1)

@FeatureRegistry.register("EMACross_Signal_M5")
def compute_ema_cross_signal_m5(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Signal for 5-minute timeframe"""
    return compute_ema_cross_signal(df, timeframe=Timeframe.M5)

@FeatureRegistry.register("EMACross_Signal_M15")
def compute_ema_cross_signal_m15(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Signal for 15-minute timeframe"""
    return compute_ema_cross_signal(df, timeframe=Timeframe.M15)

@FeatureRegistry.register("EMACross_Signal_H1")
def compute_ema_cross_signal_h1(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Signal for 1-hour timeframe"""
    return compute_ema_cross_signal(df, timeframe=Timeframe.H1)

@FeatureRegistry.register("EMACross_Signal_H4")
def compute_ema_cross_signal_h4(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Signal for 4-hour timeframe"""
    return compute_ema_cross_signal(df, timeframe=Timeframe.H4)

@FeatureRegistry.register("EMACross_Signal_D1")
def compute_ema_cross_signal_d1(df: pd.DataFrame) -> pd.Series:
    """EMA/SMA Cross Signal for daily timeframe"""
    return compute_ema_cross_signal(df, timeframe=Timeframe.D1)

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

    def get_deps(self, params: dict[str, Any] | None = None) -> list[str]:
        if params is None:
            params = self.default_params
        fast_period = params.get("fast_period", 5)
        slow_period = params.get("slow_period", 20)
        return [f"ema_{fast_period}", f"rolling_mean_{slow_period}"]

    def _compute_with_params(
        self, df: pd.DataFrame, **params: dict[str, Any]
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
            from ztb.utils.talib_wrapper import TaLibWrapper

            talib = TaLibWrapper()
            df[fast_col] = talib.ema(df["close"].values.astype(np.float64), fast_period)
        if slow_col not in df.columns:
            from ztb.utils.talib_wrapper import TaLibWrapper

            talib = TaLibWrapper()
            df[slow_col] = talib.sma(df["close"].values.astype(np.float64), slow_period)

        # Prevent division by zero by replacing zeros with np.nan
        slow_col_safe = df[slow_col].replace(0, pd.NA)
        ema_sma_cross = (df[fast_col] - slow_col_safe) / slow_col_safe
        ema_above_sma = (df[fast_col] > df[slow_col]).astype(int)

        return pd.DataFrame(
            {"ema_sma_cross": ema_sma_cross, "ema_above_sma": ema_above_sma}
        )
