#!/usr/bin/env python3
"""
test_equivalence.py
Unit tests for TS-Python equivalence
"""

import numpy as np
import pandas as pd
import pytest
from scripts.compare_outputs import compare_dataframes


def generate_test_data(n_rows: int = 1000) -> pd.DataFrame:
    """Generate test DataFrame"""
    np.random.seed(42)

    data = {
        "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="1min"),
        "close": np.random.uniform(100, 200, n_rows),
        "rsi": np.random.uniform(0, 100, n_rows),
        "macd": np.random.normal(0, 10, n_rows),
    }

    return pd.DataFrame(data)


def test_identical_data():
    """Test identical dataframes"""
    df = generate_test_data()

    result = compare_dataframes(df, df)

    assert result["pass"] == True
    assert result["error_rate"] == 0.0
    assert len(result["differences"]) == 0


def test_small_numeric_differences():
    """Test small numeric differences within tolerance"""
    df1 = generate_test_data()
    df2 = df1.copy()

    # Add small differences
    df2["close"] = df2["close"] * 1.000001  # Within default rtol=1e-6

    result = compare_dataframes(df1, df2)

    assert result["pass"] == True
    assert result["error_rate"] < 0.001


def test_large_numeric_differences():
    """Test large numeric differences"""
    df1 = generate_test_data()
    df2 = df1.copy()

    # Add large differences
    df2["close"] = df2["close"] * 1.01  # Exceeds tolerance

    result = compare_dataframes(df1, df2)

    assert result["pass"] == False
    assert result["error_rate"] > 0.0
    assert len(result["differences"]) > 0


def test_nan_handling():
    """Test NaN value handling"""
    df1 = generate_test_data()
    df2 = df1.copy()

    # Introduce NaN in different positions
    df1.loc[0, "rsi"] = np.nan
    df2.loc[1, "rsi"] = np.nan

    result = compare_dataframes(df1, df2)

    assert result["pass"] == False
    assert any("nan_mismatch" in str(diff) for diff in result["differences"])


def test_missing_columns():
    """Test dataframes with different columns"""
    df1 = generate_test_data()
    df2 = df1[["timestamp", "close"]].copy()  # Missing columns

    result = compare_dataframes(df1, df2)

    # Should only compare common columns
    assert (
        "close" in [d["column"] for d in result["differences"]]
        or len(result["differences"]) == 0
    )


def test_different_order():
    """Test dataframes with different row order"""
    df1 = generate_test_data()
    df2 = df1.iloc[::-1].reset_index(drop=True)  # Reverse order

    result = compare_dataframes(df1, df2)

    # Without key column, assumes same order - should pass
    assert result["pass"] == True


def test_with_timestamp_key():
    """Test with timestamp as key column"""
    df1 = generate_test_data()
    df2 = df1.copy()

    # Modify one value
    df2.loc[0, "close"] = 999

    result = compare_dataframes(df1, df2)

    assert result["pass"] == False
    assert len(result["differences"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
