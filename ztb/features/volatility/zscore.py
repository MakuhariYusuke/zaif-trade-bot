"""
Z-Score implementation.
標準化スコア - ボラティリティ指標
"""

import pandas as pd
import numpy as np
from ztb.features.registry import FeatureRegistry
from ztb.features.feature_cache import feature_cache


@FeatureRegistry.register("ZScore")
def compute_zscore(df: pd.DataFrame, window: int = 20) -> pd.Series[float]:
    """Compute Z-Score of returns - Optimized version"""
    if not FeatureRegistry.is_cache_enabled():
        return _compute_zscore_numpy(df, window)

    cache_key = f"zscore_{feature_cache.generate_dataframe_hash(df, ['close'], {'window': window})}"

    def compute():
        return _compute_zscore_numpy(df, window)

    return feature_cache.get_or_compute(cache_key, compute)


def _compute_zscore_numpy(df: pd.DataFrame, window: int = 20) -> pd.Series[float]:
    """Optimized Z-Score computation using pandas rolling"""
    returns = df['close'].pct_change().fillna(0)

    if len(returns) < window:
        return pd.Series(np.zeros(len(returns)), index=df.index)

    # Use pandas rolling for efficient computation
    mean = returns.rolling(window=window).mean()
    std = returns.rolling(window=window).std(ddof=1)

    # Avoid division by zero
    std = std.replace(0, 1e-8)
    zscore = (returns - mean) / std

    return zscore.fillna(0)  # type: ignore