#!/usr/bin/env python3
"""
Analysis Utilities

Common utilities for data analysis and reporting.
Provides consistent analysis patterns across the project.
"""

from pathlib import Path

import pandas as pd

from ztb.utils.data_utils import load_csv_data
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_analysis_data(
    data_path: str | Path, date_columns: list | None = None, **kwargs
) -> pd.DataFrame:
    """
    Load data for analysis with consistent handling.

    Args:
        data_path: Path to data file (CSV, etc.)
        date_columns: Columns to parse as dates
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        Loaded DataFrame
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    try:
        if data_path.suffix.lower() == ".csv":
            df = load_csv_data(data_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

        logger.info(f"Loaded {len(df)} rows from {data_path}")
        return df

    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        raise

def print_basic_stats(df: pd.DataFrame, title: str = "Data Statistics") -> None:
    """
    Print basic statistics for a DataFrame.

    Args:
        df: DataFrame to analyze
        title: Title for the statistics output
    """
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")

    # Numeric columns stats
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        print(f"Numeric Columns: {len(numeric_cols)}")
        print(f"Data Types:\n{df.dtypes}")

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing Values: {missing.sum()} total")
        print("Top missing columns:")
        missing_pct = (missing / len(df) * 100).round(2)
        for col, pct in missing_pct[missing_pct > 0].head().items():
            print(f"  {col}: {pct}%")

    print(f"{'='*60}\n")
