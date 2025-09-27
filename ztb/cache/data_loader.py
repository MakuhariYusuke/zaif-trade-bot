"""
DataLoader: Unified data loading with caching.

Provides a common interface for loading data with automatic caching.
"""

import pickle
from pathlib import Path
from typing import Dict, Callable, Optional
import pandas as pd


class DataLoader:
    """Unified data loader with caching support"""

    def __init__(self, cache_dir: str = "data/cache") -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_with_cache(self, key: str, load_func: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        """Load data with caching"""
        cache_path = self.cache_dir / f"{key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                # If cache is corrupted, remove it
                cache_path.unlink(missing_ok=True)

        data = load_func()
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            # If caching fails, continue without error
            pass
        return data

    def load_multiple(self, keys_and_loaders: Dict[str, Callable[[], pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """Load multiple datasets with caching"""
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