"""
Z-Score implementation.
標準化スコア - ボラティリティ指標
"""

import pandas as pd
import numpy as np
from ztb.features.registry import FeatureRegistry
import hashlib


# Simple cache for DataFrame-based computations
_cache = {}


def _get_df_hash(df: pd.DataFrame, window: int) -> str:
    """Generate hash for DataFrame + parameters"""
    # Use only close prices and window for hash
    close_values = df['close'].astype(float).values
    data_str = f"{window}_{close_values.tobytes()}"
    return hashlib.md5(data_str.encode()).hexdigest()


@FeatureRegistry.register("ZScore")
def compute_zscore(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute Z-Score of returns - Optimized version"""
    if not FeatureRegistry.is_cache_enabled():
        return _compute_zscore_numpy(df, window)
    
    cache_key = f"zscore_{_get_df_hash(df, window)}"
    
    if cache_key in _cache:
        return _cache[cache_key].copy()
    
    result = _compute_zscore_numpy(df, window)
    _cache[cache_key] = result.copy()
    return result


def _compute_zscore_numpy(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Optimized Z-Score computation using pure numpy operations"""
    returns = df['close'].pct_change().fillna(0).values
    
    if len(returns) < window:
        return pd.Series(np.zeros(len(returns)), index=df.index)
    
    # Calculate rolling mean and std using numpy
    mean = np.full(len(returns), np.nan)
    std = np.full(len(returns), np.nan)
    
    for i in range(window - 1, len(returns)):
        window_data = returns[i - window + 1:i + 1]
        mean[i] = np.mean(window_data)
        std[i] = np.std(window_data, ddof=1)  # sample std
    
    # Avoid division by zero
    std = np.where(std == 0, 1e-8, std)
    zscore = (returns - mean) / std
    
    return pd.Series(zscore, index=df.index).fillna(0)