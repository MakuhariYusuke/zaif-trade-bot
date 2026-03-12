"""
Improved data loading utilities for ZTB.

This module provides enhanced data loading with prefetching, memory-mapped files,
caching, and parallel processing for improved performance.
"""

import asyncio
import logging
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import pandas as pd

from ztb.io.advanced_csv import read_csv_async, read_csv_cached, read_csv_mmap
from ztb.io.data_loader import DataLoader
from ztb.trading.environment.constants import BYTES_PER_MB
from ztb.utils.cache_utils import cached_with_ttl
from ztb.utils.errors import ZTBError

logger = logging.getLogger(__name__)

class ImprovedDataLoader:
    """
    Enhanced data loader with performance optimizations.

    Features:
    - Memory-mapped file loading for large datasets
    - Asynchronous prefetching
    - Parallel feature computation
    - Incremental loading and caching
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        max_workers: int = 4,
        prefetch_buffer_size: int = 1000,
        enable_memory_mapping: bool = True,
        enable_async_loading: bool = True,
        max_cache_entries: int = 16,
    ):
        warnings.warn(
            "ImprovedDataLoader is deprecated; use ztb.io.data_loader.DataLoader "
            "for standard CSV loading. This class remains for legacy mmap/async paths.",
            DeprecationWarning,
            stacklevel=2,
        )
        """
        Initialize improved data loader.

        Args:
            cache_dir: Directory for caching
            max_workers: Max threads for parallel processing
            prefetch_buffer_size: Size of prefetch buffer
            enable_memory_mapping: Use memory-mapped files
            enable_async_loading: Enable async operations
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.prefetch_buffer_size = prefetch_buffer_size
        self.enable_memory_mapping = enable_memory_mapping
        self.enable_async_loading = enable_async_loading
        try:
            self.max_cache_entries = max(1, int(max_cache_entries))
        except (TypeError, ValueError):
            self.max_cache_entries = 16

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._prefetch_queue = asyncio.Queue(maxsize=prefetch_buffer_size)
        self._cache: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        self._mmap_files: "OrderedDict[str, object]" = OrderedDict()

    def _evict_oldest_cache(self) -> None:
        """Evict oldest cached entry to keep memory bounded."""
        if not self._cache:
            return
        cache_key, _ = self._cache.popitem(last=False)
        mmap_key = cache_key.replace("mmap_", "", 1)
        mmap_obj = self._mmap_files.pop(mmap_key, None)
        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except Exception:
                pass

    def load_csv_memory_mapped(
        self, file_path: str | Path, chunk_size: int | None = None, **kwargs
    ) -> pd.DataFrame:
        """
        Load CSV using memory mapping for large files.

        Args:
            file_path: Path to CSV file
            chunk_size: Size for chunked reading
            **kwargs: Additional pandas read_csv arguments

        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ZTBError(f"File not found: {file_path}")

        cache_key = f"mmap_{file_path.stem}_{file_path.stat().st_mtime}"
        if cache_key in self._cache:
            cached = self._cache.pop(cache_key)
            self._cache[cache_key] = cached
            return cached.copy()

        try:
            df = read_csv_mmap(
                file_path,
                chunk_size=chunk_size,
                enable_mmap=self.enable_memory_mapping,
                min_mmap_bytes=10 * BYTES_PER_MB,
                **kwargs,
            )

            # Cache result
            self._cache[cache_key] = df.copy()
            if len(self._cache) > self.max_cache_entries:
                self._evict_oldest_cache()
            logger.info(f"Loaded data from {file_path} with {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise ZTBError(f"Data loading failed: {e}")

    async def load_csv_async(
        self, file_path: str | Path, **kwargs
    ) -> pd.DataFrame:
        """
        Load CSV asynchronously.

        Args:
            file_path: Path to CSV file
            **kwargs: Additional pandas arguments

        Returns:
            Loaded DataFrame
        """
        if not self.enable_async_loading:
            return self.load_csv_memory_mapped(file_path, **kwargs)

        return await read_csv_async(
            file_path,
            executor=self.executor,
            enable_mmap=self.enable_memory_mapping,
            min_mmap_bytes=10 * BYTES_PER_MB,
            **kwargs,
        )

    def compute_features_parallel(
        self,
        data: pd.DataFrame,
        feature_functions: dict[str, callable],
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute features in parallel.

        Args:
            data: Input DataFrame
            feature_functions: dict of feature name -> function
            max_workers: Number of parallel workers

        Returns:
            DataFrame with computed features
        """
        if max_workers is None:
            max_workers = self.max_workers

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(func, data): name
                for name, func in feature_functions.items()
            }

            for future in futures:
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.error(f"Failed to compute feature {name}: {e}")
                    results[name] = pd.Series([np.nan] * len(data), index=data.index)

        # Combine results
        feature_df = pd.DataFrame(results)
        return pd.concat([data, feature_df], axis=1)

    def incremental_feature_computation(
        self,
        data: pd.DataFrame,
        feature_functions: dict[str, callable],
        cache_key: str,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """
        Compute features incrementally with caching.

        Args:
            data: Input DataFrame
            feature_functions: Feature computation functions
            cache_key: Unique cache identifier
            force_recompute: Force recomputation

        Returns:
            DataFrame with features
        """
        cache_file = self.cache_dir / f"{cache_key}_features.pkl"

        if not force_recompute and cache_file.exists():
            try:
                cached_data = pd.read_pickle(cache_file)
                # Check if data has changed
                if len(cached_data) == len(data) and cached_data.index.equals(
                    data.index
                ):
                    logger.info(f"Loaded cached features for {cache_key}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cached features: {e}")

        # Compute features
        logger.info(f"Computing features for {cache_key}")
        start_time = time.time()

        result_df = self.compute_features_parallel(data, feature_functions)

        # Cache result
        try:
            result_df.to_pickle(cache_file)
            logger.info(f"Cached features to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache features: {e}")

        elapsed = time.time() - start_time
        logger.info(f"Feature computation took {elapsed:.2f}s")

        return result_df

    async def prefetch_data(self, file_paths: list[str | Path], **kwargs) -> None:
        """
        Prefetch multiple files asynchronously.

        Args:
            file_paths: list of file paths to prefetch
            **kwargs: Loading arguments
        """
        if not self.enable_async_loading:
            return

        tasks = [
            read_csv_async(
                path,
                executor=self.executor,
                enable_mmap=self.enable_memory_mapping,
                min_mmap_bytes=10 * BYTES_PER_MB,
                **kwargs,
            )
            for path in file_paths
        ]

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Prefetched {len(file_paths)} files")

    def cleanup(self):
        """Clean up resources."""
        for mm in self._mmap_files.values():
            try:
                mm.close()
            except Exception:
                pass
        self._mmap_files.clear()
        self._cache.clear()
        self.executor.shutdown(wait=True)
        logger.info("DataLoader cleaned up")

# Convenience functions
@lru_cache(maxsize=32)
def get_cached_data_loader(
    cache_dir: str = "./cache/data", max_workers: int = 4
) -> ImprovedDataLoader:
    """Get cached data loader instance."""
    return ImprovedDataLoader(cache_dir=cache_dir, max_workers=max_workers)

@cached_with_ttl(ttl_seconds=300)  # 5 minute cache
def load_market_data_cached(
    file_path: str, loader: ImprovedDataLoader | None = None
) -> pd.DataFrame:
    """
    Load market data with caching.

    Args:
        file_path: Path to data file
        loader: Data loader instance (deprecated; ignored)

    Returns:
        Loaded DataFrame
    """
    return read_csv_cached(file_path)
