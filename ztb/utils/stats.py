"""
stats.py: Statistical utility functions.

Aggregates functions like calculate_skew, calculate_kurtosis, nan_ratio, correlation.

Usage:
    from ztb.utils.stats import calculate_skew, calculate_kurtosis, nan_ratio, correlation

    skew = calculate_skew(df)
    kurt = calculate_kurtosis(df)
    nan_r = nan_ratio(df)
    corr = correlation(a, b)
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple


def calculate_skew(data: Union[pd.Series, pd.DataFrame]) -> Union[float, int]:
    """Calculate skewness of data"""
    if isinstance(data, pd.DataFrame):
        # For DataFrame, calculate mean skewness across numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 0.0
        skews = data[numeric_cols].skew()
        return skews.mean()
    else:
        # For Series
        return float(data.skew())


def calculate_kurtosis(data: Union[pd.Series, pd.DataFrame]) -> Union[float, int]:
    """Calculate kurtosis of data"""
    if isinstance(data, pd.DataFrame):
        # For DataFrame, calculate mean kurtosis across numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 0.0
        kurts = data[numeric_cols].kurtosis()
        return kurts.mean()
    else:
        # For Series
        return float(data.kurtosis())


def nan_ratio(data: Union[pd.Series, pd.DataFrame]) -> float:
    """Calculate NaN ratio (0.0 to 1.0)"""
    if isinstance(data, pd.DataFrame):
        total_cells = data.shape[0] * data.shape[1]
        nan_cells = data.isnull().sum().sum()
        return nan_cells / total_cells if total_cells > 0 else 0.0
    else:
        # For Series
        total = len(data)
        nan_count = data.isnull().sum()
        return nan_count / total if total > 0 else 0.0


def correlation(a: pd.Series, b: pd.Series) -> float:
    """Calculate Pearson correlation between two series"""
    return a.corr(b)