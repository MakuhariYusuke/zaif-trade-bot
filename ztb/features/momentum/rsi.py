"""
RSI (Relative Strength Index) implementation with multi-timeframe support.
RSIの実装 - 複数時間軸対応
"""

from typing import Optional

import pandas as pd

from ztb.features.feature_cache import feature_cache
from ztb.features.registry import FeatureRegistry
from ztb.features.timeframe import Timeframe
from ztb.utils.talib_wrapper import TaLibWrapper


@FeatureRegistry.register("RSI")
def compute_rsi(
    df: pd.DataFrame,
    period: Optional[int] = None,
    timeframe: Optional[Timeframe] = None,
) -> pd.Series:
    """Compute RSI (Relative Strength Index) - Optimized version with Ta-Lib support"""
    # Determine period based on timeframe
    if timeframe is not None:
        from ztb.features.timeframe import get_timeframe_params

        tf_params = get_timeframe_params(timeframe)
        period = period or (
            tf_params["short_period"] // 2
        )  # RSI typically uses shorter periods
    else:
        period = period or 14

    if not FeatureRegistry.is_cache_enabled():
        return _compute_rsi_optimized(df, period)

    cache_key = f"rsi_{feature_cache.generate_dataframe_hash(df, ['close'], {'period': period})}"

    def compute() -> pd.Series:
        return _compute_rsi_optimized(df, period)

    return feature_cache.get_or_compute(cache_key, compute)


# === Multi-Timeframe RSI Features ===


@FeatureRegistry.register("RSI_M1")
def compute_rsi_m1(df: pd.DataFrame) -> pd.Series:
    """RSI for 1-minute timeframe"""
    return compute_rsi(df, timeframe=Timeframe.M1)


@FeatureRegistry.register("RSI_M5")
def compute_rsi_m5(df: pd.DataFrame) -> pd.Series:
    """RSI for 5-minute timeframe"""
    return compute_rsi(df, timeframe=Timeframe.M5)


@FeatureRegistry.register("RSI_M15")
def compute_rsi_m15(df: pd.DataFrame) -> pd.Series:
    """RSI for 15-minute timeframe"""
    return compute_rsi(df, timeframe=Timeframe.M15)


@FeatureRegistry.register("RSI_H1")
def compute_rsi_h1(df: pd.DataFrame) -> pd.Series:
    """RSI for 1-hour timeframe"""
    return compute_rsi(df, timeframe=Timeframe.H1)


@FeatureRegistry.register("RSI_H4")
def compute_rsi_h4(df: pd.DataFrame) -> pd.Series:
    """RSI for 4-hour timeframe"""
    return compute_rsi(df, timeframe=Timeframe.H4)


@FeatureRegistry.register("RSI_D1")
def compute_rsi_d1(df: pd.DataFrame) -> pd.Series:
    """RSI for daily timeframe"""
    return compute_rsi(df, timeframe=Timeframe.D1)


def _compute_rsi_optimized(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Optimized RSI computation using Ta-Lib when available, fallback to custom implementation"""
    close_prices = df["close"]

    # Use Ta-Lib if available, otherwise use custom implementation
    talib = TaLibWrapper()
    rsi_values = talib.rsi(close_prices, period)

    # Convert to pandas Series and handle NaN values
    rsi_series = pd.Series(rsi_values, index=df.index)

    # Fill initial NaN values with 50 (neutral RSI)
    rsi_series = rsi_series.fillna(50)

    return rsi_series
