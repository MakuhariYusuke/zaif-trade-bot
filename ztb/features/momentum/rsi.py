"""
RSI (Relative Strength Index) implementation.
RSIの実装
"""

import numpy as np
import pandas as pd

from ztb.features.feature_cache import feature_cache
from ztb.features.registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper


@FeatureRegistry.register("RSI")
def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index) - Optimized version with Ta-Lib support"""
    if not FeatureRegistry.is_cache_enabled():
        return _compute_rsi_optimized(df, period)

    cache_key = f"rsi_{feature_cache.generate_dataframe_hash(df, ['close'], {'period': period})}"

    def compute() -> pd.Series:
        return _compute_rsi_optimized(df, period)

    return feature_cache.get_or_compute(cache_key, compute)


def _compute_rsi_optimized(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Optimized RSI computation using Ta-Lib when available, fallback to custom implementation"""
    close_prices = np.asarray(df["close"].values)

    # Use Ta-Lib if available, otherwise use custom implementation
    rsi_values = TaLibWrapper.rsi(close_prices, period)

    # Convert to pandas Series and handle NaN values
    rsi_series = pd.Series(rsi_values, index=df.index)

    # Fill initial NaN values with 50 (neutral RSI)
    rsi_series = rsi_series.fillna(50)

    return rsi_series
