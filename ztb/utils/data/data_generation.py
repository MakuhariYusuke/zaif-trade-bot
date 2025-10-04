"""
Data generation utilities for ZTB.

This module provides functions for generating synthetic market data
and loading sample datasets for testing and experimentation.
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from ztb.utils.errors import safe_operation

logger = logging.getLogger(__name__)

# Global cache
_data_cache: Dict[str, Any] = {}


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
    # Check cache first
    cache_key = f"synthetic_{version}_{n_samples}_{seed}"
    if cache_key in _data_cache:
        return _data_cache[cache_key].copy()  # type: ignore

    np.random.seed(seed)

    if version == "v2":
        # Improved synthetic data with latent factors for realistic correlations
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
    else:
        # Original synthetic data with randomized trend
        t = np.linspace(0, 100, n_samples)
        trend_direction = np.random.choice(
            [-1, 0, 1], p=[0.3, 0.4, 0.3]
        )  # 30% down, 40% sideways, 30% up
        trend_magnitude = np.random.uniform(0.01, 0.03)  # Random trend strength
        trend = trend_direction * trend_magnitude * t  # Randomized trend
        noise = np.random.normal(0, 0.005, n_samples)  # Less noise
        price = 100 * np.exp(trend + noise)

    # Generate OHLCV data
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
    df["high"] = np.maximum(df[["open", "close"]].max(axis=1), df["high"])
    df["low"] = np.minimum(df[["open", "close"]].min(axis=1), df["low"])

    # Cache the result
    _data_cache[cache_key] = df.copy()

    return df


def load_sample_data(
    dataset: str = "synthetic",
    cache_dir: Optional[str] = None,
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
    # Create cache key
    cache_key = f"{dataset}_{'forced' if force_reload else 'cached'}"

    # Check memory cache first
    if not force_reload and cache_key in _data_cache:
        logger.info(f"Loading {dataset} from memory cache")
        return cast(pd.DataFrame, _data_cache[cache_key].copy())

    # Check disk cache
    if cache_dir and not force_reload:
        cache_path = _get_cache_path(cache_dir, dataset)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    df = pickle.load(f)
                logger.info(f"Loading {dataset} from disk cache: {cache_path}")
                _data_cache[cache_key] = df.copy()
                return cast(pd.DataFrame, df)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

    # Generate/load fresh data
    # Generate synthetic data
    if dataset == "synthetic-v2":
        df = generate_synthetic_market_data(version="v2")
    else:
        df = generate_synthetic_market_data(version="v1")

    # Cache the result
    _data_cache[cache_key] = df.copy()

    # Save to disk cache
    if cache_dir:
        cache_path = _get_cache_path(cache_dir, dataset)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(df, f)
            logger.info(f"Saved {dataset} to disk cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    return df


def _get_cache_path(cache_dir: str, dataset: str) -> Path:
    """Generate cache file path"""
    # Create hash of dataset name for filename
    dataset_hash = hashlib.md5(dataset.encode()).hexdigest()[:8]
    return Path(cache_dir) / f"dataset_{dataset_hash}.pkl"


def preload_datasets(
    datasets: List[str], cache_dir: str = "data/cache", max_workers: int = 2
) -> None:
    """
    Preload multiple datasets in parallel for faster subsequent access.

    Args:
        datasets: List of dataset names to preload
        cache_dir: Cache directory
        max_workers: Maximum number of parallel workers
    """
    from concurrent.futures import ThreadPoolExecutor

    logger.info(f"Preloading {len(datasets)} datasets with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for dataset in datasets:
            future = executor.submit(load_sample_data, dataset, cache_dir, False)
            futures.append((dataset, future))

        for dataset, future in futures:
            try:
                df = future.result()
                logger.info(f"Preloaded {dataset}: {len(df)} samples")
            except Exception as e:
                logger.error(f"Failed to preload {dataset}: {e}")


def clear_cache(cache_dir: Optional[str] = None) -> None:
    """
    Clear data cache.

    Args:
        cache_dir: Cache directory to clear (if None, clears memory cache only)
    """
    global _data_cache
    _data_cache.clear()
    logger.info("Cleared memory cache")

    if cache_dir:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            for cache_file in cache_path.glob("dataset_*.pkl"):
                cache_file.unlink()
            logger.info(f"Cleared disk cache: {cache_dir}")


def save_parquet_chunked(
    df: pd.DataFrame,
    base_path: Path,
    partition_cols: Optional[List[str]] = None,
    compression: str = "zstd",
    chunk_rows: int = 1000000,
) -> List[str]:
    """
    Save DataFrame to Parquet files in chunks with optional partitioning.

    Args:
        df: DataFrame to save
        base_path: Base path for output files
        partition_cols: Columns to partition by (e.g., ['year', 'month'])
        compression: Compression algorithm ('zstd', 'snappy', 'gzip')
        chunk_rows: Number of rows per chunk

    Returns:
        List of saved file paths
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    saved_files = []
    # base_path is already a Path object

    if partition_cols:
        # Partitioned save
        for partition_values, group_df in df.groupby(partition_cols):
            partition_path = base_path / "/".join(
                f"{col}={val}" for col, val in zip(partition_cols, partition_values)
            )

            partition_path.mkdir(parents=True, exist_ok=True)

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


def load_parquet_pattern(
    pattern: str,
    columns: Optional[List[str]] = None,
    filters: Optional[List[Any]] = None,
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
        raise FileNotFoundError(f"No Parquet files found matching pattern: {pattern}")

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} rows from {len(dfs)} Parquet files")
    return cast(pd.DataFrame, combined_df)


def save_parquet_monthly_chunked(
    df: pd.DataFrame, path: str, chunk: str = "M", compression: str = "zstd"
) -> List[str]:
    """
    Save DataFrame to Parquet files in monthly chunks with zstd compression.

    Args:
        df: DataFrame to save (must have datetime index)
        path: Base path for output files
        chunk: Chunk frequency ('M' for monthly, 'W' for weekly, etc.)
        compression: Compression algorithm

    Returns:
        List of saved file paths
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    saved_files = []
    base_path = Path(path)
    base_path.mkdir(parents=True, exist_ok=True)

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
    n_rows: int = 5000,
    freq: str = "1H",
    episode_length: Optional[int] = 1000,
    volume_range: tuple = (1000, 10000),
) -> pd.DataFrame:
    """合成データを生成（学習用）"""
    return safe_operation(
        logger=None,  # Use default logger
        operation=lambda: _generate_synthetic_data_impl(
            n_rows, freq, episode_length, volume_range
        ),
        context="synthetic_data_generation",
        default_result=pd.DataFrame(),  # Return empty DataFrame on failure
    )


def _generate_synthetic_data_impl(
    n_rows: int = 5000,
    freq: str = "1H",
    episode_length: Optional[int] = 1000,
    volume_range: tuple = (1000, 10000),
) -> pd.DataFrame:
    """Implementation of synthetic data generation."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_rows, freq=freq)

    returns = np.random.normal(0, 0.02, n_rows)
    price = 100 * np.exp(np.cumsum(returns))

    high = price * (1 + np.random.uniform(0, 0.03, n_rows))
    low = price * (1 - np.random.uniform(0, 0.03, n_rows))
    close = price
    volume = np.random.uniform(volume_range[0], volume_range[1], n_rows)

    # エピソードID: episode_lengthステップごとに変更（Noneの場合は0固定）
    if episode_length is not None:
        episode_ids = np.repeat(
            np.arange(n_rows // episode_length + 1), episode_length
        )[:n_rows]
    else:
        episode_ids = np.zeros(n_rows, dtype=int)

    df = pd.DataFrame(
        {
            "ts": dates.view("int64") // 10**9,
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
