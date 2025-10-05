#!/usr/bin/env python3
"""
Data loading utilities for consistent data handling across the codebase.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Iterator, cast, Optional, Union, Literal


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory-optimized DataFrame
    """
    df_optimized = df.copy()
    
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
    
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    # Convert object columns to category if they have few unique values
    for col in df_optimized.select_dtypes(include=['object']).columns:
        if df_optimized[col].nunique() / len(df_optimized) < 0.5:  # Less than 50% unique
            df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized


def safe_merge_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Optional[Union[str, list[str]]] = None,
    how: Literal['left', 'right', 'outer', 'inner'] = 'left',
    **kwargs: Any
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
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        df = cast(pd.DataFrame, pd.read_csv(file_path, **kwargs))
        if df.empty:
            raise ValueError(f"Loaded data is empty: {file_path}")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load data from {file_path}: {e}") from e


def load_csv_data_iter(file_path: str | Path, chunksize: int, **kwargs: Any) -> Iterator[pd.DataFrame]:
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
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        return pd.read_csv(file_path, chunksize=chunksize, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to load data from {file_path}: {e}") from e