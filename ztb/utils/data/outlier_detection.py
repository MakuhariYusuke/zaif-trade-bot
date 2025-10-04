"""
Outlier detection utilities for ZTB.

This module provides standardized functions for detecting outliers in data
using various statistical methods (IQR, Z-score, etc.).
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ztb.utils.errors import safe_operation

logger = logging.getLogger(__name__)


def detect_outliers_iqr(
    data: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, float, float]:
    """
    Detect outliers using IQR (Interquartile Range) method.

    Args:
        data: Input DataFrame
        column: Column name to analyze

    Returns:
        Tuple of (outliers DataFrame, lower_bound, upper_bound)
    """
    return safe_operation(
        logger,
        lambda: _detect_outliers_iqr_impl(data, column),
        f"detect_outliers_iqr({column})",
        (pd.DataFrame(), 0.0, 0.0)
    )


def _detect_outliers_iqr_impl(
    data: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, float, float]:
    """Implementation of IQR outlier detection."""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def detect_outliers_zscore(
    data: pd.DataFrame, column: str, threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect outliers using Z-score method.

    Args:
        data: Input DataFrame
        column: Column name to analyze
        threshold: Z-score threshold (default: 3.0)

    Returns:
        DataFrame containing outliers
    """
    return safe_operation(
        logger,
        lambda: _detect_outliers_zscore_impl(data, column, threshold),
        f"detect_outliers_zscore({column})",
        pd.DataFrame()
    )


def _detect_outliers_zscore_impl(
    data: pd.DataFrame, column: str, threshold: float = 3.0
) -> pd.DataFrame:
    """Implementation of Z-score outlier detection."""
    series = data[column]

    # Calculate z-scores, handling NaN values
    z_scores_raw = stats.zscore(series, nan_policy="omit")

    # Convert to numpy array, filling NaN with 0
    z_scores = np.nan_to_num(z_scores_raw, nan=0.0)

    # Find outliers
    outlier_mask = np.abs(z_scores) > threshold
    outliers = data[outlier_mask]

    return outliers


def detect_outliers_isolation_forest(
    data: pd.DataFrame,
    columns: list[str],
    contamination: float = 0.1
) -> pd.DataFrame:
    """
    Detect outliers using Isolation Forest method.

    Args:
        data: Input DataFrame
        columns: List of column names to analyze
        contamination: Expected proportion of outliers (default: 0.1)

    Returns:
        DataFrame containing outliers
    """
    return safe_operation(
        logger,
        lambda: _detect_outliers_isolation_forest_impl(data, columns, contamination),
        f"detect_outliers_isolation_forest({columns})",
        pd.DataFrame()
    )


def _detect_outliers_isolation_forest_impl(
    data: pd.DataFrame,
    columns: list[str],
    contamination: float = 0.1
) -> pd.DataFrame:
    """Implementation of Isolation Forest outlier detection."""
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        logger.warning("scikit-learn not available, skipping Isolation Forest detection")
        return pd.DataFrame()

    # Prepare data
    X = data[columns].fillna(data[columns].mean())

    # Fit isolation forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_predictions = iso_forest.fit_predict(X)

    # Get outliers (prediction == -1)
    outliers = data[outlier_predictions == -1]

    return outliers


def remove_outliers(
    data: pd.DataFrame,
    column: str,
    method: str = "iqr",
    **kwargs
) -> pd.DataFrame:
    """
    Remove outliers from DataFrame using specified method.

    Args:
        data: Input DataFrame
        column: Column name to clean
        method: Detection method ("iqr", "zscore", "isolation_forest")
        **kwargs: Additional parameters for detection method

    Returns:
        DataFrame with outliers removed
    """
    return safe_operation(
        logger,
        lambda: _remove_outliers_impl(data, column, method, **kwargs),
        f"remove_outliers({column}, {method})",
        data.copy()
    )


def _remove_outliers_impl(
    data: pd.DataFrame,
    column: str,
    method: str = "iqr",
    **kwargs
) -> pd.DataFrame:
    """Implementation of outlier removal."""
    if method == "iqr":
        outliers, _, _ = detect_outliers_iqr(data, column)
    elif method == "zscore":
        outliers = detect_outliers_zscore(data, column, **kwargs)
    elif method == "isolation_forest":
        outliers = detect_outliers_isolation_forest(data, [column], **kwargs)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    # Remove outliers
    clean_data = data[~data.index.isin(outliers.index)]
    return clean_data