"""
RSI (Relative Strength Index) implementation.
RSIの実装
"""

import pandas as pd

from ztb.features.feature_cache import feature_cache
from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("RSI")
def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index) - Optimized version"""
    if not FeatureRegistry.is_cache_enabled():
        return _compute_rsi_optimized(df, period)

    cache_key = f"rsi_{feature_cache.generate_dataframe_hash(df, ['close'], {'period': period})}"

    def compute() -> pd.Series:
        return _compute_rsi_optimized(df, period)

    return feature_cache.get_or_compute(cache_key, compute)


def _compute_rsi_optimized(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Optimized RSI computation using pandas only"""
    # Calculate price changes
    delta = df["close"].astype(float).diff()

    # Calculate gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Calculate average gains and losses using EMA for smoothing
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0, 1e-8)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)
