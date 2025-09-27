"""
OBV (On-Balance Volume) implementation.
OBVの実装
"""

import numpy as np
import pandas as pd
from typing import cast
from ztb.features.registry import FeatureRegistry
from ztb.features.feature_cache import feature_cache


@FeatureRegistry.register("OBV")
def compute_obv(df: pd.DataFrame) -> pd.Series[float]:
    """Compute OBV (On-Balance Volume)"""
    if not FeatureRegistry.is_cache_enabled():
        direction = np.sign(df['close'].diff().fillna(0))
        signed_volume = pd.Series(direction * df['volume'], index=df.index)
        obv = signed_volume.fillna(0).cumsum()
        return pd.Series(obv, name='OBV', index=df.index)  # type: ignore

    cache_key = f"obv_{feature_cache.generate_dataframe_hash(df, ['close', 'volume'])}"

    def compute() -> pd.Series:
        direction = np.sign(df['close'].diff().fillna(0))
        signed_volume = pd.Series(direction * df['volume'], index=df.index)
        obv = signed_volume.fillna(0).cumsum()
        return cast(pd.Series, pd.Series(obv, name='OBV', index=df.index))

    return feature_cache.get_or_compute(cache_key, compute)