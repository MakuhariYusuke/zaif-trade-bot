"""
FeatureCache: Unified caching for feature computations.

Provides a common caching mechanism for DataFrame-based feature calculations.
"""

import hashlib
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


class FeatureCache:
    """Unified cache for feature computations"""

    def __init__(self) -> None:
        self._cache: Dict[str, pd.Series] = {}

    def generate_dataframe_hash(
        self,
        df: pd.DataFrame,
        columns: List[str],
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate hash for DataFrame + parameters"""
        values = []
        for col in columns:
            if col in df.columns:
                col_values = df[col].astype(float).values
                values.append(col_values.tobytes())  # type: ignore
            else:
                values.append(b"")  # Empty for missing columns

        param_str = "_".join(f"{k}:{v}" for k, v in (params or {}).items())
        data_str = (
            f"{param_str}_{'_'.join(str(v) for v in values)}"
            if param_str
            else "_".join(str(v) for v in values)
        )
        return hashlib.md5(data_str.encode()).hexdigest()

    def get_or_compute(
        self, key: str, compute_func: Callable[[], pd.Series]
    ) -> pd.Series:
        """Get from cache or compute and cache"""
        if key in self._cache:
            return self._cache[key].copy()
        result = compute_func()
        self._cache[key] = result.copy()
        return result

    def clear(self) -> None:
        """Clear all cached data"""
        self._cache.clear()

    def size(self) -> int:
        """Get number of cached items"""
        return len(self._cache)


# Global instance
feature_cache = FeatureCache()
