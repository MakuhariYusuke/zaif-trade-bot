#!/usr/bin/env python3
"""
Data loading utilities for consistent data handling across the codebase.
"""

from pathlib import Path
from typing import Any, Iterator

import pandas as pd


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
        df = pd.read_csv(file_path, **kwargs)
        if df.empty:
            raise ValueError(f"Loaded data is empty: {file_path}")
        return df  # type: ignore[no-any-return]
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
        return pd.read_csv(file_path, chunksize=chunksize, **kwargs)  # type: ignore
    except Exception as e:
        raise ValueError(f"Failed to load data from {file_path}: {e}") from e