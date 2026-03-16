#!/usr/bin/env python3
"""
Data validation utilities for ZTB system.

This module provides common data validation functions used across the codebase.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ztb.types.common import ConfigDict

logger = logging.getLogger(__name__)

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    column_types: dict[str, str] | None = None,
    min_rows: int = 1,
) -> bool:
    """
    Validate DataFrame structure and content for trading data.

    Performs comprehensive validation of pandas DataFrames used in trading
    applications, checking for required columns, data types, and minimum
    row counts. This function is critical for ensuring data quality before
    training or inference.

    Args:
        df: DataFrame to validate. Must contain trading data with proper structure.
        required_columns: list of column names that must be present in the DataFrame.
                          Common examples: ['close', 'volume', 'timestamp']
        column_types: Optional mapping of column names to expected numpy/pandas dtypes.
                     Examples: {'close': 'float64', 'volume': 'int64'}
        min_rows: Minimum number of rows required for valid data. Defaults to 1.

    Returns:
        bool: True if all validation checks pass, False if any check fails.
             Failed validations are logged as errors.

    Raises:
        No exceptions raised - validation failures are logged and return False.

    Example:
        >>> df = pd.DataFrame({'close': [100, 101, 102], 'volume': [1000, 1100, 900]})
        >>> validate_dataframe(df, ['close', 'volume'], min_rows=3)
        True
    """
    if df is None or df.empty:
        logger.error("DataFrame is None or empty")
        return False

    if len(df) < min_rows:
        logger.error(f"DataFrame has {len(df)} rows, minimum {min_rows} required")
        return False

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False

    if column_types:
        for col, expected_type in column_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    logger.warning(
                        f"Column {col} has type {actual_type}, expected {expected_type}"
                    )

    return True

def validate_numeric_array(
    data: np.ndarray[Any, np.dtype[np.floating[Any]]] | pd.Series | list[float],
    name: str = "data",
    allow_nan: bool = True,
    min_value: float | None = None,
    max_value: float | None = None,
) -> bool:
    """
    Validate numeric array data.

    Args:
        data: Numeric data to validate
        name: Name for logging purposes
        allow_nan: Whether NaN values are allowed
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        True if validation passes, False otherwise
    """
    try:
        arr = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as e:
        logger.error(f"Cannot convert {name} to numeric array: {e}")
        return False

    if not allow_nan and np.any(np.isnan(arr)):
        logger.error(f"{name} contains NaN values")
        return False

    if min_value is not None and np.any(arr < min_value):
        logger.error(f"{name} contains values below minimum {min_value}")
        return False

    if max_value is not None and np.any(arr > max_value):
        logger.error(f"{name} contains values above maximum {max_value}")
        return False

    return True

def validate_config_dict(
    config: ConfigDict,
    required_keys: list[str],
    validators: dict[str, Callable[[Any], bool]] | None = None,
) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate
        required_keys: list of required keys
        validators: Optional dict of validator functions for specific keys

    Returns:
        True if validation passes, False otherwise
    """
    missing_keys = set(required_keys) - set(config.keys())
    if missing_keys:
        logger.error(f"Missing required configuration keys: {missing_keys}")
        return False

    if validators:
        for key, validator in validators.items():
            if key in config and not validator(config[key]):
                logger.error(f"Configuration key '{key}' failed validation")
                return False

    return True

def sanitize_numeric_value(
    value: Any,
    default: float = 0.0,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """
    Sanitize a numeric value with bounds checking.

    Args:
        value: Value to sanitize
        default: Default value if conversion fails
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Sanitized numeric value
    """
    try:
        result = float(value)
    except (ValueError, TypeError):
        logger.warning(f"Cannot convert {value} to float, using default {default}")
        result = default

    if min_val is not None:
        result = max(result, min_val)
    if max_val is not None:
        result = min(result, max_val)

    return result

def validate_file_path(file_path: str | Path, must_exist: bool = True) -> bool:
    """
    Validate file path.

    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist

    Returns:
        True if validation passes, False otherwise
    """
    from pathlib import Path

    path = Path(file_path)

    if must_exist and not path.exists():
        logger.error(f"File does not exist: {file_path}")
        return False

    if must_exist and not path.is_file():
        logger.error(f"Path is not a file: {file_path}")
        return False

    return True
