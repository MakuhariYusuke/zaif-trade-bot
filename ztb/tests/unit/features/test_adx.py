"""
Tests for ADX feature implementation.
"""

import numpy as np
import pandas as pd

from ztb.features.trend.adx import ADX


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

        adx_feature = ADX(period=14)
        result = adx_feature.compute(df)

        # Check output columns exist
        expected_columns = [f"adx_14", f"plus_di_14", f"minus_di_14"]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        # Check output shape
        assert len(result) == len(df), "Output length should match input"

        # Check for reasonable values (ADX typically 0-100)
        adx_values = result[f"adx_14"].dropna()
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

        adx_feature = ADX(period=14)
        result = adx_feature.compute(df)

        # Should still produce output, but values might be NaN
        assert len(result) == len(df), "Output length should match input"
        assert not result.empty, "Should produce some output"

    def test_adx_edge_cases(self):
        """Test ADX with edge cases."""
        # Flat market (no movement)
        df = pd.DataFrame({"high": [100] * 20, "low": [100] * 20, "close": [100] * 20})

        adx_feature = ADX(period=14)
        result = adx_feature.compute(df)

        # Should handle flat market gracefully
        assert len(result) == len(df)
        # ADX should be low or NaN in flat market
        adx_values = result[f"adx_14"].dropna()
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

        adx_feature = ADX(period=10)
        result = adx_feature.compute(df)

        expected_columns = ["adx_10", "plus_di_10", "minus_di_10"]
        for col in expected_columns:
            assert col in result.columns, f"Expected column {col} not found"

        # Check that we have some non-NaN values
        total_values = len(result)
        nan_count = result.isnull().sum().sum()
        assert nan_count < total_values, "Should have some valid values"
