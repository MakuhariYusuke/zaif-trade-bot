#!/usr/bin/env python3
"""
Data loading utilities for consistent data handling across the codebase.
"""

from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Union, cast

import pandas as pd


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.

    Args:
        df: Input DataFrame

    Returns:
        Memory-optimized DataFrame
    """
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
    on: Optional[Union[str, list[str]]] = None,
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
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        return cast(
            Iterator[pd.DataFrame],
            pd.read_csv(file_path, chunksize=chunksize, **kwargs),
        )
    except Exception as e:
        raise ValueError(f"Failed to load data from {file_path}: {e}") from e


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
        usecols: List of columns to read (None for all columns)
        dtype: Dictionary mapping column names to dtypes for memory optimization
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        Loaded DataFrame with optimized memory usage

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be loaded
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        # Default dtype optimizations for common trading data columns
        if dtype is None:
            dtype = {
                # Price and volume data - use float32 for memory efficiency
                "close": "float32",
                "high": "float32",
                "low": "float32",
                "open": "float32",
                "volume": "float32",
                "qty": "float32",
                "price": "float32",
                # Technical indicators - float32 is usually sufficient
                "rsi": "float32",
                "sma_short": "float32",
                "sma_long": "float32",
                "ADX": "float32",
                "ATR": "float32",
                "ATR_simplified": "float32",
                "BB_Lower": "float32",
                "BB_Middle": "float32",
                "BB_Position": "float32",
                "BB_Upper": "float32",
                "BB_Width": "float32",
                "CCI": "float32",
                "DOW": "float32",
                "Donchian_Pos_2": "float32",
                "Donchian_Slope_20": "float32",
                "Donchian_Width_Rel_20": "float32",
                "EMACross_Diff": "float32",
                "EMACross_Signal": "float32",
                "HV": "float32",
                "HeikinAshi_Close": "float32",
                "HeikinAshi_High": "float32",
                "HeikinAshi_Low": "float32",
                "HeikinAshi_Open": "float32",
                "HourOfDay": "int32",
                "Ichimoku_Chikou": "float32",
                "Ichimoku_Cloud_Thickness": "float32",
                "Ichimoku_Composite_Signal": "float32",
                "Ichimoku_Cross": "float32",
                "Ichimoku_Diff_Norm": "float32",
                "Ichimoku_Kijun": "float32",
                "Ichimoku_Price_Cloud_Distance": "float32",
                "Ichimoku_Senkou_A": "float32",
                "Ichimoku_Senkou_B": "float32",
                "Ichimoku_Tenkan": "float32",
                "Ichimoku_Trend": "float32",
                "KAMA": "float32",
                "Kalman_Estimate": "float32",
                "Kalman_Residual": "float32",
                "Kalman_Residual_Norm": "float32",
                "MACD": "float32",
                "MFI": "float32",
                "MinusDI": "float32",
                "OBV": "float32",
                "PlusDI": "float32",
                "PriceVolumeCorr": "float32",
                "ROC": "float32",
                "RSI": "float32",
                "ReturnMA_Medium": "float32",
                "ReturnMA_Short": "float32",
                "ReturnStdDev": "float32",
                "Stochastic": "float32",
                "Supertrend": "float32",
                "Supertrend_Direction": "float32",
                "TEMA": "float32",
                "VWAP": "float32",
                "ZScore": "float32",
                "atr_10": "float32",
                "ema_5": "float32",
                "rolling_mean_20": "float32",
                # Integer columns
                "win": "int32",
            }

        # Read header to determine available columns for parse_dates
        header_df = pd.read_csv(file_path, nrows=0)
        available_columns = list(header_df.columns)

        # Default parse_dates for timestamp columns
        if parse_dates is None:
            parse_dates = [
                col for col in ["timestamp", "ts"] if col in available_columns
            ]

        df = cast(
            pd.DataFrame,
            pd.read_csv(
                file_path,
                usecols=usecols,
                dtype=cast(Any, dtype),
                parse_dates=parse_dates,
                **kwargs,
            ),
        )

        if df.empty:
            raise ValueError(f"Loaded data is empty: {file_path}")

        return df
    except Exception as e:
        raise ValueError(f"Failed to load optimized data from {file_path}: {e}") from e
