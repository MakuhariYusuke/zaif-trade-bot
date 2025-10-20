"""
Unit tests for ztb.analysis.wave3_diag module.
"""

import pandas as pd

from ztb.analysis.wave3_diag import (
    calculate_correlations,
    calculate_mutual_info,
    calculate_vif,
    check_leaks,
    generate_synthetic_data,
)


class TestWave3Diag:
    """Test cases for wave3 diagnostic functions."""

    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        df = generate_synthetic_data(n_rows=100)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "ts" in df.columns
        assert "close" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "volume" in df.columns
        assert "episode_id" in df.columns

    def test_calculate_correlations_normal(self):
        """Test normal correlation calculation."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 4, 6, 8, 10],
                "feature3": [1, 3, 5, 7, 9],
            }
        )

        result = calculate_correlations(df)

        assert isinstance(result, dict)
        assert "pearson" in result
        assert "spearman" in result
        assert isinstance(result["pearson"], pd.DataFrame)
        assert isinstance(result["spearman"], pd.DataFrame)

    def test_calculate_correlations_empty(self):
        """Test correlation calculation with empty DataFrame."""
        df = pd.DataFrame()
        result = calculate_correlations(df)

        assert isinstance(result, dict)
        assert "pearson" in result
        assert "spearman" in result
        assert result["pearson"].empty
        assert result["spearman"].empty

    def test_calculate_vif_normal(self):
        """Test normal VIF calculation."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 4, 6, 8, 10],
                "feature3": [1, 3, 5, 7, 9],
            }
        )

        result = calculate_vif(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "vif" in result.columns

    def test_calculate_vif_insufficient_features(self):
        """Test VIF calculation with insufficient features."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
            }
        )

        result = calculate_vif(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["vif"] == 1.0

    def test_calculate_mutual_info_normal(self):
        """Test normal mutual information calculation."""
        df = pd.DataFrame(
            {
                "close": [100, 102, 101, 103, 105],
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 4, 6, 8, 10],
            }
        )

        result = calculate_mutual_info(df, [1])

        assert isinstance(result, dict)
        assert "h1" in result
        assert isinstance(result["h1"], pd.DataFrame)
        assert len(result["h1"]) >= 2

    def test_calculate_mutual_info_no_horizons(self):
        """Test mutual information calculation with empty horizons."""
        df = pd.DataFrame(
            {
                "close": [100, 102, 101, 103, 105],
                "feature1": [1, 2, 3, 4, 5],
            }
        )

        result = calculate_mutual_info(df, [])

        assert result == {}

    def test_check_leaks_normal(self):
        """Test normal leak checking."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 4, 6, 8, 10],
                "close": [100, 102, 101, 103, 105],
                "return": [0, 1, 0, 1, 0],
            }
        )

        result = check_leaks(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 2
        assert "corr_current" in result.columns
        assert "corr_future" in result.columns

    def test_check_leaks_no_target(self):
        """Test leak checking without target column."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 4, 6, 8, 10],
            }
        )

        result = check_leaks(df)

        # Should process features but correlations will be NaN due to missing close column
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert all(pd.isna(result["corr_current"]))
        assert all(pd.isna(result["corr_future"]))
