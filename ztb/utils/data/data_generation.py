"""
Data generation utilities for ZTB.

This module provides functions for generating synthetic market data
and loading sample datasets for testing and experimentation.
"""

import gzip
import hashlib
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ztb.io.advanced_csv import prefetch_csv, read_csvs_async
from ztb.io.data_loader import DataLoader
from ztb.utils.cache_utils import cached_with_ttl
from ztb.utils.errors import safe_operation
from ztb.utils.path_utils import ensure_dir
from ztb.utils.performance_utils import timed

logger = logging.getLogger(__name__)

class DataGenerator:
    """
    Class for generating and managing synthetic market data.

    Provides methods for generating synthetic OHLCV data, loading datasets
    with caching, and managing data persistence.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        enable_memory_cache: bool = True,
        default_seed: int = 42,
    ):
        """
        Initialize DataGenerator.

        Args:
            cache_dir: Directory for disk caching
            enable_memory_cache: Whether to use in-memory caching
            default_seed: Default random seed for reproducibility
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enable_memory_cache = enable_memory_cache
        self.default_seed = default_seed
        self._memory_cache: dict[str, Any] = {}

        if self.cache_dir and self.cache_dir.exists():
            ensure_dir(self.cache_dir)

    @timed
    def generate_synthetic_market_data(
        self,
        n_samples: int = 10000,
        version: str = "v2",
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic market data for testing.

        Args:
            n_samples: Number of data points to generate
            version: Data generation version ("v1" or "v2")
            seed: Random seed for reproducibility (uses default if None)

        Returns:
            DataFrame with OHLCV data
        """
        actual_seed = seed if seed is not None else self.default_seed

        return safe_operation(
            self._generate_synthetic_market_data_impl,
            logger=logger,
            context=f"Generating synthetic data (version={version}, n_samples={n_samples})",
            fallback=pd.DataFrame(),
            n_samples=n_samples,
            version=version,
            seed=actual_seed,
        )

    @timed
    def _generate_synthetic_market_data_impl(
        self,
        n_samples: int,
        version: str,
        seed: int,
    ) -> pd.DataFrame:
        """
        Implementation of synthetic market data generation.
        """
        # Check memory cache first
        cache_key = f"synthetic_{version}_{n_samples}_{seed}"
        if self.enable_memory_cache and cache_key in self._memory_cache:
            return self._memory_cache[cache_key].copy()  # type: ignore

        if version == "v2":
            price = self._generate_price_series_v2(n_samples, seed)
        else:
            price = self._generate_price_series_v1(n_samples, seed)

        df = self._generate_ohlcv_from_price(price, n_samples, seed)

        # Cache the result
        if self.enable_memory_cache:
            self._memory_cache[cache_key] = df.copy()

        return df

    @lru_cache(maxsize=64)
    def _generate_price_series_v2(self, n_samples: int, seed: int) -> np.ndarray:
        """Generate price series using v2 algorithm with latent factors."""
        np.random.seed(seed)
        t = np.linspace(0, 100, n_samples)

        # Latent factors that features can correlate with
        cycle = 0.1 * np.sin(2 * np.pi * t / 50)  # Cyclical component
        momentum = 0.05 * np.cumsum(np.random.normal(0, 0.01, n_samples))  # Momentum
        volatility = 0.02 * np.abs(
            np.random.normal(0, 0.01, n_samples)
        )  # Volatility factor

        # Price influenced by latent factors
        # Randomize trend direction to create balanced market conditions
        trend_direction = np.random.choice(
            [-1, 0, 1], p=[0.3, 0.4, 0.3]
        )  # 30% down, 40% sideways, 30% up
        trend_magnitude = np.random.uniform(0.005, 0.015)  # Random trend strength
        trend = trend_direction * trend_magnitude * t

        latent_influence = 0.3 * cycle + 0.2 * momentum + 0.1 * volatility
        noise = np.random.normal(0, 0.003, n_samples)
        price = 100 * np.exp(trend + latent_influence + noise)
        return price

    def _generate_price_series_v1(self, n_samples: int, seed: int) -> np.ndarray:
        """Generate price series using v1 algorithm with simple trend."""
        np.random.seed(seed)
        t = np.linspace(0, 100, n_samples)
        trend_direction = np.random.choice(
            [-1, 0, 1], p=[0.3, 0.4, 0.3]
        )  # 30% down, 40% sideways, 30% up
        trend_magnitude = np.random.uniform(0.01, 0.03)  # Random trend strength
        trend = trend_direction * trend_magnitude * t  # Randomized trend
        noise = np.random.normal(0, 0.005, n_samples)  # Less noise
        price = 100 * np.exp(trend + noise)
        return price

    def _generate_ohlcv_from_price(
        self, price: np.ndarray, n_samples: int, seed: int
    ) -> pd.DataFrame:
        """Generate OHLCV DataFrame from price series."""
        np.random.seed(seed + 1)  # Different seed for OHLCV
        volume = np.random.lognormal(12, 0.5, n_samples)  # Higher volume

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=n_samples, freq="1H"),
                "open": price * (1 + np.random.normal(0, 0.002, n_samples)),
                "high": price * (1 + np.random.normal(0, 0.005, n_samples)),
                "low": price * (1 - np.random.normal(0, 0.005, n_samples)),
                "close": price,
                "volume": volume,
            }
        )

        # Ensure high >= max(open, close) and low <= min(open, close)
        df["high"] = np.maximum(df[["open", "close"]].max(axis="columns"), df["high"])
        df["low"] = np.minimum(df[["open", "close"]].min(axis="columns"), df["low"])
        return df

    def load_dataset(
        self,
        dataset: str = "synthetic",
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Load sample market data for testing with caching support.

        Args:
            dataset: Dataset type ("synthetic", "synthetic-v2", "coingecko")
            force_reload: Force reload even if cached

        Returns:
            DataFrame with market data
        """
        return safe_operation(
            self._load_dataset_impl,
            logger=logger,
            context=f"Loading dataset {dataset}",
            fallback=pd.DataFrame(),
            dataset=dataset,
            force_reload=force_reload,
        )

    def _load_dataset_impl(
        self,
        dataset: str,
        force_reload: bool,
    ) -> pd.DataFrame:
        """Implementation of dataset loading"""
        # Create cache key
        cache_key = f"{dataset}_{'forced' if force_reload else 'cached'}"

        # Check memory cache first
        if (
            not force_reload
            and self.enable_memory_cache
            and cache_key in self._memory_cache
        ):
            logger.info(f"Loading {dataset} from memory cache")
            return cast(pd.DataFrame, self._memory_cache[cache_key].copy())

        # Check disk cache
        if self.cache_dir and not force_reload:
            cache_path = self._get_cache_path(dataset)
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        df = pickle.load(f)
                    logger.info(f"Loading {dataset} from disk cache: {cache_path}")
                    if self.enable_memory_cache:
                        self._memory_cache[cache_key] = df.copy()
                    return cast(pd.DataFrame, df)
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")

        # Generate/load fresh data
        if dataset == "synthetic-v2":
            df = self.generate_synthetic_market_data(version="v2")
        else:
            df = self.generate_synthetic_market_data(version="v1")

        # Cache the result
        if self.enable_memory_cache:
            self._memory_cache[cache_key] = df.copy()

        # Save to disk cache
        if self.cache_dir:
            cache_path = self._get_cache_path(dataset)
            ensure_dir(cache_path.parent)
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(df, f)
                logger.info(f"Saved {dataset} to disk cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

        return df

    def _get_cache_path(self, dataset: str) -> Path:
        """Generate cache file path"""
        if not self.cache_dir:
            raise ValueError("Cache directory not set")
        # Create hash of dataset name for filename
        dataset_hash = hashlib.md5(dataset.encode()).hexdigest()[:8]
        return self.cache_dir / f"dataset_{dataset_hash}.pkl"

    def clear_cache(self) -> None:
        """
        Clear data cache.
        """
        if self.enable_memory_cache:
            self._memory_cache.clear()
            logger.info("Cleared memory cache")

        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("dataset_*.pkl"):
                cache_file.unlink()
            logger.info(f"Cleared disk cache: {self.cache_dir}")

    def preload_datasets(self, datasets: list[str], max_workers: int = 2) -> None:
        """
        Preload multiple datasets in parallel for faster subsequent access.

        Args:
            datasets: list of dataset names to preload
            max_workers: Maximum number of parallel workers
        """
        logger.info(f"Preloading {len(datasets)} datasets with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for dataset in datasets:
                future = executor.submit(self.load_dataset, dataset, False)
                futures.append((dataset, future))

            for dataset, future in futures:
                try:
                    df = future.result()
                    logger.info(f"Preloaded {dataset}: {len(df)} samples")
                except Exception as e:
                    logger.error(f"Failed to preload {dataset}: {e}")

    def save_parquet_chunked(
        self,
        df: pd.DataFrame,
        base_path: Path,
        partition_cols: list[str] | None = None,
        compression: str = "zstd",
        chunk_rows: int = 1000000,
    ) -> list[str]:
        """
        Save DataFrame to Parquet files in chunks with optional partitioning.

        Args:
            df: DataFrame to save
            base_path: Base path for output files
            partition_cols: Columns to partition by (e.g., ['year', 'month'])
            compression: Compression algorithm ('zstd', 'snappy', 'gzip')
            chunk_rows: Number of rows per chunk

        Returns:
            list of saved file paths
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        saved_files = []

        if partition_cols:
            # Partitioned save
            for partition_values, group_df in df.groupby(partition_cols):
                partition_path = base_path / "/".join(
                    f"{col}={str(val)}"
                    for col, val in zip(partition_cols, partition_values)
                )

                ensure_dir(partition_path)

                # Chunk large partitions
                for i, chunk_start in enumerate(range(0, len(group_df), chunk_rows)):
                    chunk_df = group_df.iloc[chunk_start : chunk_start + chunk_rows]
                    file_path = partition_path / f"part_{i:04d}.parquet"

                    table = pa.Table.from_pandas(chunk_df)
                    pq.write_table(table, file_path, compression=compression)
                    saved_files.append(str(file_path))

        else:
            # Non-partitioned save with chunking
            for i, chunk_start in enumerate(range(0, len(df), chunk_rows)):
                chunk_df = df.iloc[chunk_start : chunk_start + chunk_rows]
                file_path = base_path / f"part_{i:04d}.parquet"

                table = pa.Table.from_pandas(chunk_df)
                pq.write_table(table, file_path, compression=compression)
                saved_files.append(str(file_path))

        logger.info(f"Saved {len(saved_files)} Parquet files to {base_path}")
        return saved_files

    @cached_with_ttl(ttl_seconds=1800)  # Cache for 30 minutes
    def load_parquet_pattern(
        self,
        pattern: str,
        columns: list[str] | None = None,
        filters: list[Any] | None = None,
    ) -> pd.DataFrame:
        """
        Load Parquet files matching a pattern with optional column selection and filtering.

        Args:
            pattern: Glob pattern for Parquet files (e.g., "data/**/*.parquet")
            columns: Columns to load (None for all)
            filters: Row filters for predicate pushdown

        Returns:
            Combined DataFrame
        """
        return safe_operation(
            self._load_parquet_pattern_impl,
            logger=logger,
            context="Loading parquet pattern",
            fallback=pd.DataFrame(),
            pattern=pattern,
            columns=columns,
            filters=filters,
        )

    def _load_parquet_pattern_impl(
        self,
        pattern: str,
        columns: list[str] | None = None,
        filters: list[Any] | None = None,
    ) -> pd.DataFrame:
        """Implementation of parquet pattern loading"""
        import pyarrow.parquet as pq

        dfs = []
        for file_path in Path().glob(pattern):
            try:
                table = pq.read_table(file_path, columns=columns, filters=filters)
                df = table.to_pandas()
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not dfs:
            raise FileNotFoundError(
                f"No Parquet files found matching pattern: {pattern}"
            )

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} rows from {len(dfs)} Parquet files")
        return combined_df

    def save_parquet_monthly_chunked(
        self, df: pd.DataFrame, path: str, chunk: str = "M", compression: str = "zstd"
    ) -> list[str]:
        """
        Save DataFrame to Parquet files in monthly chunks with zstd compression.

        Args:
            df: DataFrame to save (must have datetime index)
            path: Base path for output files
            chunk: Chunk frequency ('M' for monthly, 'W' for weekly, etc.)
            compression: Compression algorithm

        Returns:
            list of saved file paths
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        saved_files = []
        base_path = Path(path)
        ensure_dir(base_path)

        # Group by chunk period
        for period, group_df in df.groupby(pd.Grouper(freq=chunk.replace("M", "ME"))):
            # Create filename with period
            if chunk == "M":
                filename = f"{period.year}-{period.month:02d}.parquet"  # type: ignore
            elif chunk == "W":
                filename = f"{period.year}-W{period.week:02d}.parquet"  # type: ignore
            else:
                filename = f"{period.strftime('%Y%m%d')}.parquet"  # type: ignore

            file_path = base_path / filename

            # Convert to PyArrow table and save
            table = pa.Table.from_pandas(group_df)
            pq.write_table(table, file_path, compression=compression)
            saved_files.append(str(file_path))

            logger.info(f"Saved {len(group_df)} rows to {file_path}")

        logger.info(f"Saved {len(saved_files)} Parquet files to {base_path}")
        return saved_files

    def generate_synthetic_data(
        self,
        n_rows: int = 5000,
        freq: str = "1H",
        episode_length: int | None = None,
        volume_range: tuple[float, float] = (1000, 10000),
    ) -> pd.DataFrame:
        """Generate synthetic data for training."""
        return safe_operation(
            self._generate_synthetic_data_impl,
            logger=logger,
            context="Generating synthetic training data",
            fallback=pd.DataFrame(),
            n_rows=n_rows,
            freq=freq,
            episode_length=episode_length,
            volume_range=volume_range,
        )

    def _generate_synthetic_data_impl(
        self,
        n_rows: int,
        freq: str,
        episode_length: int | None,
        volume_range: tuple[float, float],
    ) -> pd.DataFrame:
        """Implementation of synthetic training data generation."""
        np.random.seed(self.default_seed)
        dates = pd.date_range("2024-01-01", periods=n_rows, freq=freq)

        returns = np.random.normal(0, 0.02, n_rows)
        price = 100 * np.exp(np.cumsum(returns))

        high = price * (1 + np.random.uniform(0, 0.03, n_rows))
        low = price * (1 - np.random.uniform(0, 0.03, n_rows))
        close = price
        volume = np.random.uniform(volume_range[0], volume_range[1], n_rows)

        # Episode ID: change every episode_length steps (None means fixed 0)
        if episode_length is not None:
            episode_ids = np.repeat(
                np.arange(n_rows // episode_length + 1), episode_length
            )[:n_rows]
        else:
            episode_ids = np.zeros(n_rows, dtype=int)

        df = pd.DataFrame(
            {
                "ts": dates.astype("int64") // 10**9,
                "close": close,
                "high": high,
                "low": low,
                "volume": volume,
                "exchange": "synthetic",
                "pair": "BTC/USD",
                "episode_id": episode_ids,
            }
        )

        return df

    def load_data_with_memory_map(self, file_path: str) -> np.memmap:
        """
        Load numpy data with memory mapping.

        Args:
            file_path: Path to .npy file
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return cast(np.memmap, np.load(path, mmap_mode="r"))

    def load_compressed_data(self, file_path: str, compression: str = "gzip") -> Any:
        """
        Load compressed pickled data.

        Args:
            file_path: Path to compressed pickle
            compression: Compression type (gzip)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if compression != "gzip":
            raise ValueError("Unsupported compression")
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    async def load_data_async(
        self, file_paths: list[str], max_workers: int = 4, **kwargs
    ) -> list[pd.DataFrame]:
        """
        Load multiple CSV files asynchronously.
        """
        return await read_csvs_async(file_paths, max_workers=max_workers, **kwargs)

    def prefetch_data(
        self, file_paths: list[str], prefetch_size: int = 2, **kwargs
    ) -> ThreadPoolExecutor:
        """
        Prefetch data files and warm in-memory cache.
        """
        executor = prefetch_csv(file_paths[:prefetch_size], max_workers=prefetch_size, **kwargs)
        if self.enable_memory_cache:
            for path in file_paths[:prefetch_size]:
                cache_key = f"prefetch_{hashlib.md5(path.encode()).hexdigest()}"
                try:
                    df = DataLoader.load_csv_optimized(path, **kwargs)
                    self._memory_cache[cache_key] = df
                except Exception:
                    pass
        return executor

def load_sample_data(n_samples: int = 100, version: str = "v1") -> pd.DataFrame:
    """Convenience helper used by tests to quickly get a sample dataset.

    Args:
        n_samples: number of rows to generate (default 100)
        version: generator version

    Returns:
        pd.DataFrame with synthetic OHLCV data
    """
    dg = DataGenerator()
    return dg.generate_synthetic_market_data(n_samples=n_samples, version=version)

# Global cache for backward compatibility
_data_cache: dict[str, Any] = {}

def generate_synthetic_market_data(
    n_samples: int = 10000, version: str = "v2", seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic market data for testing.

    Args:
        n_samples: Number of data points to generate
        version: Data generation version ("v1" or "v2")
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV data
    """
    generator = DataGenerator()
    return generator.generate_synthetic_market_data(n_samples, version, seed)

def load_dataset(
    dataset: str = "synthetic",
    cache_dir: str | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Load sample market data for testing with caching support.

    Args:
        dataset: Dataset type ("synthetic", "synthetic-v2", "coingecko")
        cache_dir: Directory to store cached data
        force_reload: Force reload even if cached

    Returns:
        DataFrame with market data
    """
    generator = DataGenerator(cache_dir=cache_dir)
    return generator.load_dataset(dataset, force_reload)

def preload_datasets(
    datasets: list[str], cache_dir: str = "data/cache", max_workers: int = 2
) -> None:
    """
    Preload multiple datasets in parallel for faster subsequent access.

    Args:
        datasets: list of dataset names to preload
        cache_dir: Cache directory
        max_workers: Maximum number of parallel workers
    """
    generator = DataGenerator(cache_dir=cache_dir)
    generator.preload_datasets(datasets, max_workers)

def clear_cache(cache_dir: str | None = None) -> None:
    """
    Clear data cache.

    Args:
        cache_dir: Cache directory to clear (if None, clears memory cache only)
    """
    generator = DataGenerator(cache_dir=cache_dir)
    generator.clear_cache()

def save_parquet_chunked(
    df: pd.DataFrame,
    base_path: Path,
    partition_cols: list[str] | None = None,
    compression: str = "zstd",
    chunk_rows: int = 1000000,
) -> list[str]:
    """
    Save DataFrame to Parquet files in chunks with optional partitioning.

    Args:
        df: DataFrame to save
        base_path: Base path for output files
        partition_cols: Columns to partition by (e.g., ['year', 'month'])
        compression: Compression algorithm ('zstd', 'snappy', 'gzip')
        chunk_rows: Number of rows per chunk

    Returns:
        list of saved file paths
    """
    generator = DataGenerator()
    return generator.save_parquet_chunked(
        df, base_path, partition_cols, compression, chunk_rows
    )

def load_parquet_pattern(
    pattern: str,
    columns: list[str] | None = None,
    filters: list[Any] | None = None,
) -> pd.DataFrame:
    """
    Load Parquet files matching a pattern with optional column selection and filtering.

    Args:
        pattern: Glob pattern for Parquet files (e.g., "data/**/*.parquet")
        columns: Columns to load (None for all)
        filters: Row filters for predicate pushdown

    Returns:
        Combined DataFrame
    """
    generator = DataGenerator()
    return generator.load_parquet_pattern(pattern, columns, filters)

def save_parquet_monthly_chunked(
    df: pd.DataFrame, path: str, chunk: str = "M", compression: str = "zstd"
) -> list[str]:
    """
    Save DataFrame to Parquet files in monthly chunks with zstd compression.

    Args:
        df: DataFrame to save (must have datetime index)
        path: Base path for output files
        chunk: Chunk frequency ('M' for monthly, 'W' for weekly, etc.)
        compression: Compression algorithm

    Returns:
        list of saved file paths
    """
    generator = DataGenerator()
    return generator.save_parquet_monthly_chunked(df, path, chunk, compression)

def generate_synthetic_data(
    n_rows: int = 5000,
    freq: str = "1H",
    episode_length: int | None = None,
    volume_range: tuple[float, float] = (1000, 10000),
) -> pd.DataFrame:
    """Generate synthetic data for training."""
    generator = DataGenerator()
    return generator.generate_synthetic_data(n_rows, freq, episode_length, volume_range)
