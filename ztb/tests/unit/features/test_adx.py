"""
Tests for ADX feature implementation.
"""

import numpy as np
import pandas as pd

from ztb.features.trend.adx import compute_adx, compute_minus_di, compute_plus_di


class TestADX:
    """Test ADX feature implementation."""

    def test_adx_basic_computation(self):
        """Test basic ADX computation with sufficient data."""
        # Create sample OHLC data
        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n)) * 2
        low = close - np.abs(np.random.randn(n)) * 2

        df = pd.DataFrame({"high": high, "low": low, "close": close})

        adx_result = compute_adx(df, period=14)
        plus_di_result = compute_plus_di(df, period=14)
        minus_di_result = compute_minus_di(df, period=14)

        # Check results are pandas Series
        assert isinstance(adx_result, pd.Series)
        assert isinstance(plus_di_result, pd.Series)
        assert isinstance(minus_di_result, pd.Series)

        # Check output shape
        assert len(adx_result) == len(df), "ADX output length should match input"
        assert len(plus_di_result) == len(
            df
        ), "Plus DI output length should match input"
        assert len(minus_di_result) == len(
            df
        ), "Minus DI output length should match input"

        # Check for reasonable values (ADX typically 0-100)
        adx_values = adx_result.dropna()
        if len(adx_values) > 0:
            assert all(
                0 <= val <= 100 for val in adx_values
            ), "ADX values should be between 0 and 100"

    def test_adx_insufficient_data(self):
        """Test ADX with insufficient data."""
        # Very small dataset
        df = pd.DataFrame(
            {"high": [100, 101, 102], "low": [99, 98, 97], "close": [100, 100, 100]}
        )

        adx_result = compute_adx(df, period=14)
        plus_di_result = compute_plus_di(df, period=14)
        minus_di_result = compute_minus_di(df, period=14)

        # Should still produce output, but values might be NaN
        assert len(adx_result) == len(df), "ADX output length should match input"
        assert len(plus_di_result) == len(
            df
        ), "Plus DI output length should match input"
        assert len(minus_di_result) == len(
            df
        ), "Minus DI output length should match input"

    def test_adx_edge_cases(self):
        """Test ADX with edge cases."""
        # Flat market (no movement)
        df = pd.DataFrame({"high": [100] * 20, "low": [100] * 20, "close": [100] * 20})

        adx_result = compute_adx(df, period=14)
        plus_di_result = compute_plus_di(df, period=14)
        minus_di_result = compute_minus_di(df, period=14)

        # Should handle flat market gracefully
        assert len(adx_result) == len(df)
        assert len(plus_di_result) == len(df)
        assert len(minus_di_result) == len(df)

        # ADX should be low or NaN in flat market
        adx_values = adx_result.dropna()
        if len(adx_values) > 0:
            assert all(val >= 0 for val in adx_values), "ADX should be non-negative"

    def test_adx_output_columns(self):
        """Test that ADX produces expected output columns."""
        df = pd.DataFrame(
            {
                "high": np.random.uniform(95, 105, 30),
                "low": np.random.uniform(95, 105, 30),
                "close": np.random.uniform(95, 105, 30),
            }
        )

        adx_result = compute_adx(df, period=10)
        plus_di_result = compute_plus_di(df, period=10)
        minus_di_result = compute_minus_di(df, period=10)

        # Check that results are Series (not DataFrame columns)
        assert isinstance(adx_result, pd.Series)
        assert isinstance(plus_di_result, pd.Series)
        assert isinstance(minus_di_result, pd.Series)

        # Check that we have some non-NaN values
        assert not adx_result.isnull().all(), "ADX should have some valid values"
        assert (
            not plus_di_result.isnull().all()
        ), "Plus DI should have some valid values"
        assert (
            not minus_di_result.isnull().all()
        ), "Minus DI should have some valid values"
