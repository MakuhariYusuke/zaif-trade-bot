"""
OBV (On-Balance Volume) implementation.
OBVの実装
"""

from typing import cast

import numpy as np
import pandas as pd

from ztb.features.core.registry import FeatureRegistry
from ztb.features.processors.caching.cache import feature_cache
from ztb.utils.talib_wrapper import TaLibWrapper


@FeatureRegistry.register("OBV")
def compute_obv(df: pd.DataFrame) -> pd.Series:
    """Compute OBV (On-Balance Volume) using Ta-Lib wrapper"""
    if not FeatureRegistry.is_cache_enabled():
        result = TaLibWrapper.obv(
            df["close"].values.astype(np.float64),
            df["volume"].values.astype(np.float64),
        )
        return pd.Series(result, name="OBV", index=df.index)

    cache_key = f"obv_{feature_cache.generate_dataframe_hash(df, ['close', 'volume'])}"

    def compute() -> pd.Series:
        result = TaLibWrapper.obv(
            df["close"].values.astype(np.float64),
            df["volume"].values.astype(np.float64),
        )
        return cast(pd.Series, pd.Series(result, name="OBV", index=df.index))

    return feature_cache.get_or_compute(cache_key, compute)
