"""
OBV (On-Balance Volume) implementation.
OBVの実装
"""

import numpy as np
import pandas as pd
from ztb.features.registry import FeatureRegistry
import hashlib


# Simple cache for DataFrame-based computations
_cache = {}


def _get_df_hash(df: pd.DataFrame) -> str:
    """Generate hash for DataFrame"""
    # Use close and volume for hash
    close_values = df['close'].astype(float).values
    volume_values = df['volume'].astype(float).values
    data_str = f"{close_values.tobytes()}_{volume_values.tobytes()}"
    return hashlib.md5(data_str.encode()).hexdigest()


@FeatureRegistry.register("OBV")
def compute_obv(df: pd.DataFrame) -> pd.Series:
    """Compute OBV (On-Balance Volume)"""
    if not FeatureRegistry.is_cache_enabled():
        direction = np.sign(df['close'].diff().fillna(0))
        signed_volume = pd.Series(direction * df['volume'], index=df.index)
        obv = signed_volume.fillna(0).cumsum()
        return pd.Series(obv, name='OBV', index=df.index)
    
    cache_key = f"obv_{_get_df_hash(df)}"
    
    if cache_key in _cache:
        return _cache[cache_key].copy()
    
    direction = np.sign(df['close'].diff().fillna(0))
    signed_volume = pd.Series(direction * df['volume'], index=df.index)
    obv = signed_volume.fillna(0).cumsum()
    result = pd.Series(obv, name='OBV', index=df.index)
    
    _cache[cache_key] = result.copy()
    return result