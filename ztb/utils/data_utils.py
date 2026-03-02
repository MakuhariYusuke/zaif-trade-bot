#!/usr/bin/env python3
"""
Data loading utilities for consistent data handling across the codebase.

Deprecated: Prefer ztb.io.data_loader.DataLoader for CSV/JSON/Parquet loading and
data handling utilities. This module remains for backward compatibility and
will be removed after legacy callers are migrated.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Iterator, Literal, cast

import pandas as pd

from ztb.io.data_loader import DataLoader
from ztb.utils.signal_utils import suppress_signals

logger = logging.getLogger(__name__)

warnings.warn(
    "ztb.utils.data_utils is deprecated; prefer ztb.io.data_loader.DataLoader "
    "and dedicated utilities in ztb.io.",
    DeprecationWarning,
    stacklevel=2,
)

def load_csv_data_cached(
    file_path: str | Path,
    force_refresh: bool = False,
    cache_format: Literal["feather", "parquet"] = "feather",
    **kwargs: Any
) -> pd.DataFrame:
    """
    Load CSV data with caching to avoid repeated timestamp parsing.
    
    This function creates a cached version of the CSV file in feather/parquet format
    with timestamps already converted. This avoids pandas' C extension issues on Windows.
    
    Args:
        file_path: Path to the CSV file
        force_refresh: Force recreation of cache
        cache_format: Cache file format ('feather' or 'parquet')
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        Loaded DataFrame with timestamps already converted
    """
    warnings.warn(
        "load_csv_data_cached is deprecated; use ztb.io.data_loader.DataLoader "
        "directly (or a cache-specific utility).",
        DeprecationWarning,
        stacklevel=2,
    )
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Determine cache path (append, not replace suffix)
    cache_ext = f".cached.{cache_format}"
    cache_path = file_path.parent / (file_path.name + cache_ext)
    
    # Check if cache exists and is newer than source
    use_cache = (
        not force_refresh
        and cache_path.exists()
        and cache_path.stat().st_mtime > file_path.stat().st_mtime
    )
    
    if use_cache:
        try:
            logger.info(f"Loading cached data from {cache_path}")
            if cache_format == "feather":
                df = pd.read_feather(cache_path)
            else:
                df = pd.read_parquet(cache_path)
            logger.info(f"✅ Cache loaded: {df.shape}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, regenerating...")
    
    # Load from CSV and create cache
    logger.info(f"Loading CSV and creating cache: {file_path}")
    
    try:
        policy = os.getenv("ZTB_SIGINT_POLICY", "default")
        with suppress_signals(policy=policy, enabled=True):
            # Load CSV
            df = cast(pd.DataFrame, DataLoader.load_csv(file_path, **kwargs))

            # Convert timestamp if present
            if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(
                df["timestamp"]
            ):
                logger.info("Converting timestamps...")
                # Use basic pd.to_datetime in a controlled environment
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Save to cache
            logger.info(f"Saving cache to {cache_path}")
            if cache_format == "feather":
                df.to_feather(cache_path)
            else:
                df.to_parquet(cache_path, index=False)

            logger.info(f"✅ Cache created: {cache_path}")
            return df

    except Exception as e:
        raise ValueError(f"Failed to load/cache data from {file_path}: {e}") from e

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.

    Args:
        df: Input DataFrame

    Returns:
        Memory-optimized DataFrame
    """
    warnings.warn(
        "optimize_dataframe_memory is deprecated; prefer specialized utilities "
        "or DataLoader.load_csv_optimized where appropriate.",
        DeprecationWarning,
        stacklevel=2,
    )
    df_optimized = df.copy()

    for col in df_optimized.select_dtypes(include=["int64"]).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="integer")

    for col in df_optimized.select_dtypes(include=["float64"]).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="float")

    # Convert object columns to category if they have few unique values
    for col in df_optimized.select_dtypes(include=["object"]).columns:
        if (
            df_optimized[col].nunique() / len(df_optimized) < 0.5
        ):  # Less than 50% unique
            df_optimized[col] = df_optimized[col].astype("category")

    return df_optimized

def safe_merge_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str] | None = None,
    how: Literal["left", "right", "outer", "inner"] = "left",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Safely merge DataFrames with memory optimization.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Column(s) to merge on
        how: Type of merge ('left', 'right', 'outer', 'inner')
        **kwargs: Additional arguments for pd.merge

    Returns:
        Merged DataFrame
    """
    warnings.warn(
        "safe_merge_dataframes is deprecated; prefer dedicated merge utilities "
        "or pandas operations in-place.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        result = pd.merge(left, right, on=on, how=how, **kwargs)
        return optimize_dataframe_memory(result)
    except Exception as e:
        raise ValueError(f"Failed to merge DataFrames: {e}") from e

def load_csv_data(file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Load CSV data with consistent error handling.

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be loaded
    """
    warnings.warn(
        "load_csv_data is deprecated; use ztb.io.data_loader.DataLoader.load_csv",
        DeprecationWarning,
        stacklevel=2,
    )
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        df = cast(pd.DataFrame, DataLoader.load_csv(file_path, **kwargs))
        if df.empty and kwargs.get("nrows") != 0:
            raise ValueError(f"Loaded data is empty: {file_path}")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load data from {file_path}: {e}") from e

def load_csv_data_strict(file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Load CSV data and raise on failure.

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        Loaded DataFrame
    """
    warnings.warn(
        "load_csv_data_strict is deprecated; use ztb.io.data_loader.DataLoader.load_csv_strict",
        DeprecationWarning,
        stacklevel=2,
    )
    return DataLoader.load_csv_strict(file_path, **kwargs)

def load_csv_data_iter(
    file_path: str | Path, chunksize: int, **kwargs: Any
) -> Iterator[pd.DataFrame]:
    """
    Load CSV data in chunks with consistent error handling.

    Args:
        file_path: Path to the CSV file
        chunksize: Number of rows per chunk
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        Iterator of DataFrames

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be loaded
    """
    warnings.warn(
        "load_csv_data_iter is deprecated; use ztb.io.data_loader.DataLoader.load_csv_iter",
        DeprecationWarning,
        stacklevel=2,
    )
    return DataLoader.load_csv_iter(file_path, chunksize, **kwargs)

def load_csv_data_optimized(
    file_path: str | Path,
    usecols: list[str] | None = None,
    dtype: dict[str, Any] | None = None,
    parse_dates: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load CSV data with memory optimization.

    Args:
        file_path: Path to the CSV file
        usecols: list of columns to read (None for all columns)
        dtype: Dictionary mapping column names to dtypes for memory optimization
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        Loaded DataFrame with optimized memory usage

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be loaded
    """
    warnings.warn(
        "load_csv_data_optimized is deprecated; use ztb.io.data_loader.DataLoader.load_csv_optimized",
        DeprecationWarning,
        stacklevel=2,
    )
    return DataLoader.load_csv_optimized(
        file_path,
        usecols=usecols,
        dtype=dtype,
        parse_dates=parse_dates,
        **kwargs,
    )

logger = logging.getLogger(__name__)
