"""
Unit tests for stats utilities
"""
import pandas as pd
import numpy as np
from ztb.utils.stats import calculate_skew, calculate_kurtosis, nan_ratio, correlation


def test_calculate_skew():
    """Test calculate_skew returns float"""
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_skew(data)
    assert isinstance(result, float)


def test_calculate_kurtosis():
    """Test calculate_kurtosis returns float"""
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_kurtosis(data)
    assert isinstance(result, float)


def test_nan_ratio():
    """Test nan_ratio returns float"""
    data = pd.Series([1, np.nan, 3, np.nan, 5])
    result = nan_ratio(data)
    assert isinstance(result, float)
    assert result == 0.4  # 2 NaN out of 5


def test_correlation():
    """Test correlation returns float"""
    a = pd.Series([1, 2, 3, 4, 5])
    b = pd.Series([1, 2, 3, 4, 5])
    result = correlation(a, b)
    assert isinstance(result, float)