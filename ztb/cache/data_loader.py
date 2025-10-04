"""
DataLoader: Unified data loading with caching.

Provides a common interface for loading data with automatic caching.
"""

import pickle
from pathlib import Path
from typing import Callable, Dict, Optional, cast

import pandas as pd

from ztb.utils.errors import safe_operation


class DataLoader:
    """Unified data loader with caching support"""

    def __init__(self, cache_dir: str = "data/cache") -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_with_cache(
        self, key: str, load_func: Callable[[], pd.DataFrame]
    ) -> pd.DataFrame:
        """Load data with caching"""
        return safe_operation(
            logger=None,  # Use default logger
            operation=lambda: self._load_with_cache_impl(key, load_func),
            context="data_loading_with_cache",
            default_result=pd.DataFrame(),  # Return empty DataFrame on failure
        )

    def _load_with_cache_impl(
        self, key: str, load_func: Callable[[], pd.DataFrame]
    ) -> pd.DataFrame:
        """Implementation of load with cache."""
        cache_path = self.cache_dir / f"{key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return cast(pd.DataFrame, pickle.load(f))
            except Exception:
                # If cache is corrupted, remove it
                cache_path.unlink(missing_ok=True)

        data = load_func()
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            # If caching fails, continue without error
            pass
        return data

    def load_multiple(
        self, keys_and_loaders: Dict[str, Callable[[], pd.DataFrame]]
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple datasets with caching"""
        return safe_operation(
            logger=None,  # Use default logger
            operation=lambda: self._load_multiple_impl(keys_and_loaders),
            context="multiple_data_loading",
            default_result={},  # Return empty dict on failure
        )

    def _load_multiple_impl(
        self, keys_and_loaders: Dict[str, Callable[[], pd.DataFrame]]
    ) -> Dict[str, pd.DataFrame]:
        """Implementation of load multiple."""
        result = {}
        for key, loader in keys_and_loaders.items():
            result[key] = self.load_with_cache(key, loader)
        return result

    def clear_cache(self, key: Optional[str] = None) -> None:
        """Clear cache for specific key or all"""
        if key:
            cache_path = self.cache_dir / f"{key}.pkl"
            cache_path.unlink(missing_ok=True)
        else:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink(missing_ok=True)

    def list_cached(self) -> list[str]:
        """List cached keys"""
        return [f.stem for f in self.cache_dir.glob("*.pkl")]
