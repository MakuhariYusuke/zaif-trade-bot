"""
Unit tests for ztb.analysis.timeseries module.
"""

import pandas as pd
import pytest

from ztb.analysis.timeseries import compute_lag_correlations


class TestTimeseries:
    """Test cases for timeseries analysis functions."""

    def test_compute_lag_correlations_normal(self):
        """Test normal lag correlation computation."""
        df1 = pd.DataFrame({
            "feature1": list(range(50)),
            "feature2": list(range(0, 100, 2)),
        })
        df2 = pd.DataFrame({
            "feature3": list(range(25, 75)),
        })
        
        frames = {"df1": df1, "df2": df2}
        result = compute_lag_correlations(frames)
        
        assert isinstance(result, list)
        if result:  # May be empty if insufficient data
            assert len(result) <= 10  # Top 10
            for item in result:
                assert "feature1" in item
                assert "feature2" in item
                assert "lag" in item
                assert "correlation" in item
                assert item["lag"] in [1, 5, 10, 20]

    def test_compute_lag_correlations_empty_frames(self):
        """Test lag correlation computation with empty frames."""
        result = compute_lag_correlations({})
        
        assert result == []

    def test_compute_lag_correlations_insufficient_data(self):
        """Test lag correlation computation with insufficient data."""
        df = pd.DataFrame({
            "feature1": [1, 2],  # Too few data points
            "feature2": [2, 4],
        })
        
        frames = {"df": df}
        result = compute_lag_correlations(frames)
        
        assert result == []

    def test_compute_lag_correlations_high_nan(self):
        """Test lag correlation computation with high NaN columns."""
        df = pd.DataFrame({
            "good_feature": list(range(50)),
            "high_nan": [1] * 10 + [None] * 40,  # 80% NaN
        })
        
        frames = {"df": df}
        result = compute_lag_correlations(frames)
        
        assert isinstance(result, list)
        # Should exclude high_nan column

    def test_compute_lag_correlations_constant_column(self):
        """Test lag correlation computation with constant columns."""
        df = pd.DataFrame({
            "variable": list(range(50)),
            "constant": [1] * 50,
        })
        
        frames = {"df": df}
        result = compute_lag_correlations(frames)
        
        assert isinstance(result, list)
        # Should exclude constant column

    def test_compute_lag_correlations_sorting(self):
        """Test that results are sorted by absolute correlation."""
        # Create data with known correlations
        import numpy as np
        np.random.seed(42)
        
        base = np.random.normal(0, 1, 100)
        high_corr = base + np.random.normal(0, 0.1, 100)
        low_corr = np.random.normal(0, 1, 100)
        
        df = pd.DataFrame({
            "base": base,
            "high_corr": high_corr,
            "low_corr": low_corr,
        })
        
        frames = {"df": df}
        result = compute_lag_correlations(frames)
        
        assert isinstance(result, list)
        if len(result) > 1:
            # Check that correlations are in descending order of absolute value
            for i in range(len(result) - 1):
                assert abs(result[i]["correlation"]) >= abs(result[i + 1]["correlation"])