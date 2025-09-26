"""
ROC (Rate of Change) implementation.
ROCの実装
"""

import pandas as pd
from ztb.features.registry import FeatureRegistry
import hashlib


# Simple cache for DataFrame-based computations
_cache = {}


def _get_df_hash(df: pd.DataFrame, period: int) -> str:
    """Generate hash for DataFrame + parameters"""
    # Use close prices and period for hash
    close_values = df['close'].astype(float).values
    data_str = f"{period}_{close_values.tobytes()}"
    return hashlib.md5(data_str.encode()).hexdigest()


@FeatureRegistry.register("ROC")
def compute_roc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Compute ROC (Rate of Change)"""
    if not FeatureRegistry.is_cache_enabled():
        roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        return roc.fillna(0)
    
    cache_key = f"roc_{_get_df_hash(df, period)}"
    
    if cache_key in _cache:
        return _cache[cache_key].copy()
    
    roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
    result = roc.fillna(0)
    
    _cache[cache_key] = result.copy()
    return result