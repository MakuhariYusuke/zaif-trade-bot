"""
CacheDataLoader: Unified data loading with caching.

Provides a common interface for loading data with automatic caching.
Enhanced with TTLCache memory management integration.
"""

import pickle
from pathlib import Path
from typing import Callable, Dict, Optional, cast

import pandas as pd

from ztb.utils.errors import safe_operation

from .memory_cache import default_memory_manager


class CacheDataLoader:
    """Unified data loader with caching support"""

    def __init__(
        self, cache_dir: str = "data/cache", enable_memory_cache: bool = True
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_memory_cache = enable_memory_cache
        self.memory_manager = default_memory_manager if enable_memory_cache else None

    def load_with_cache(
        self, key: str, load_func: Callable[[], pd.DataFrame]
    ) -> pd.DataFrame:
        """Load data with caching"""
        return cast(
            pd.DataFrame,
            safe_operation(
                logger=None,  # Use default logger
                operation=lambda: self._load_with_cache_impl(key, load_func),
                context="data_loading_with_cache",
                default_result=pd.DataFrame(),  # Return empty DataFrame on failure
            ),
        )

    def _load_with_cache_impl(
        self, key: str, load_func: Callable[[], pd.DataFrame]
    ) -> pd.DataFrame:
        """Implementation of load with cache."""
        cache_key = f"{self.cache_dir.resolve()}::{key}"
        # Check memory cache first
        if self.memory_manager:
            cached_data = self.memory_manager.get_cached_training_data(cache_key)
            if cached_data is not None:
                return cached_data

        cache_path = self.cache_dir / f"{key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    data = cast(pd.DataFrame, pickle.load(f))
                    # Cache in memory for future use
                    if self.memory_manager:
                        self.memory_manager.cache_training_data(cache_key, data)
                    return data
            except Exception:
                # If cache is corrupted, remove it
                cache_path.unlink(missing_ok=True)

        data = load_func()
        try:
            # Write to a temporary file then atomically rename to avoid partial writes or
            # platform-specific file locking issues (especially on Windows).
            tmp_path = cache_path.with_suffix(".pkl.tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(data, f)
                f.flush()
                try:
                    # Ensure data is flushed to disk
                    import os

                    os.fsync(f.fileno())
                except Exception:
                    # os.fsync may not be available in some environments; ignore if it fails
                    pass
            # Atomic replacement
            tmp_path.replace(cache_path)
            # Also cache in memory
            if self.memory_manager:
                self.memory_manager.cache_training_data(cache_key, data)
        except Exception:
            # If caching fails, continue without error
            pass
        return data

    def load_multiple(
        self, keys_and_loaders: Dict[str, Callable[[], pd.DataFrame]]
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple datasets with caching"""
        return cast(
            Dict[str, pd.DataFrame],
            safe_operation(
                logger=None,  # Use default logger
                operation=lambda: self._load_multiple_impl(keys_and_loaders),
                context="multiple_data_loading",
                default_result={},  # Return empty dict on failure
            ),
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


# Backwards-compatible alias to reduce churn in legacy scripts/tests.
DataLoader = CacheDataLoader
