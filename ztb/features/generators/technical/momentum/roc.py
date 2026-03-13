"""
ROC (Rate of Change) implementation.
ROCの実装
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ztb.features.core.registry import FeatureRegistry
from ztb.features.processors.caching.cache import feature_cache
from ztb.utils.talib_wrapper import TaLibWrapper

@FeatureRegistry.register("ROC")
def compute_roc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Compute ROC (Rate of Change) using Ta-Lib wrapper"""
    if not FeatureRegistry.is_cache_enabled():
        result = TaLibWrapper.roc(df["close"].values.astype(np.float64), period)
        return pd.Series(result, index=df.index)

    cache_key = f"roc_{feature_cache.generate_dataframe_hash(df, ['close'], {'period': period})}"

    def compute() -> pd.Series:
        result = TaLibWrapper.roc(df["close"].values.astype(np.float64), period)
        return pd.Series(result, index=df.index)

    return feature_cache.get_or_compute(cache_key, compute)
