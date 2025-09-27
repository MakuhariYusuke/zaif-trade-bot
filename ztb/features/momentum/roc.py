"""
ROC (Rate of Change) implementation.
ROCの実装
"""

import pandas as pd
from ztb.features.registry import FeatureRegistry
from ztb.features.feature_cache import feature_cache


@FeatureRegistry.register("ROC")
def compute_roc(df: pd.DataFrame, period: int = 10) -> pd.Series[float]:
    """Compute ROC (Rate of Change)"""
    if not FeatureRegistry.is_cache_enabled():
        roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        return roc.fillna(0)

    cache_key = f"roc_{feature_cache.generate_dataframe_hash(df, ['close'], {'period': period})}"

    def compute() -> pd.Series:
        roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        return roc.fillna(0)

    return feature_cache.get_or_compute(cache_key, compute)