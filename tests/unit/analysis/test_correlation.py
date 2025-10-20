"""
Unit tests for ztb.analysis.correlation module.
"""

import pandas as pd

from ztb.analysis.correlation import compute_correlations


class TestCorrelation:
    """Test cases for correlation analysis functions."""

    def test_compute_correlations_normal(self):
        """Test normal correlation computation."""
        df1 = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 4, 6, 8, 10],
            }
        )
        df2 = pd.DataFrame(
            {
                "feature3": [1, 3, 5, 7, 9],
            }
        )

        frames = {"df1": df1, "df2": df2}
        result = compute_correlations(frames)

        assert "pearson" in result
        assert "spearman" in result
        assert isinstance(result["pearson"], pd.DataFrame)
        assert isinstance(result["spearman"], pd.DataFrame)
        assert result["pearson"].shape[0] > 0
        assert result["spearman"].shape[0] > 0

    def test_compute_correlations_empty_frames(self):
        """Test correlation computation with empty frames."""
        result = compute_correlations({})

        assert result == {"pearson": None, "spearman": None}

    def test_compute_correlations_high_nan(self):
        """Test correlation computation with high NaN columns."""
        df = pd.DataFrame(
            {
                "good_feature1": [1, 2, 3, 4, 5],
                "good_feature2": [2, 4, 6, 8, 10],
                "high_nan": [1, None, None, None, None],  # 80% NaN
            }
        )

        frames = {"df": df}
        result = compute_correlations(frames)

        # Should exclude high_nan column
        assert result["pearson"] is not None
        assert "df_good_feature1" in result["pearson"].columns
        assert "df_good_feature2" in result["pearson"].columns
        assert "df_high_nan" not in result["pearson"].columns

    def test_compute_correlations_constant_column(self):
        """Test correlation computation with constant columns."""
        df = pd.DataFrame(
            {
                "variable1": [1, 2, 3, 4, 5],
                "variable2": [2, 4, 6, 8, 10],
                "constant": [1, 1, 1, 1, 1],
            }
        )

        frames = {"df": df}
        result = compute_correlations(frames)

        # Should exclude constant column
        assert result["pearson"] is not None
        assert "df_variable1" in result["pearson"].columns
        assert "df_variable2" in result["pearson"].columns
        assert "df_constant" not in result["pearson"].columns

    def test_compute_correlations_nan_strategy_drop(self):
        """Test correlation computation with nan_strategy='drop'."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, None, 4, 5],
                "feature2": [2, None, 6, 8, 10],
            }
        )

        frames = {"df": df}
        result = compute_correlations(frames, nan_strategy="drop")

        assert result["pearson"] is not None
        # Should have fewer rows due to NaN dropping

    def test_compute_correlations_nan_strategy_fill(self):
        """Test correlation computation with nan_strategy='fill'."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, None, 4, 5],
                "feature2": [2, None, 6, 8, 10],
            }
        )

        frames = {"df": df}
        result = compute_correlations(frames, nan_strategy="fill", fill_value=0.0)

        assert result["pearson"] is not None
        # Should fill NaN with 0.0

    def test_compute_correlations_nan_strategy_none(self):
        """Test correlation computation with nan_strategy='none'."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, None, 4, 5],
                "feature2": [2, None, 6, 8, 10],
            }
        )

        frames = {"df": df}
        result = compute_correlations(frames, nan_strategy="none")

        assert result["pearson"] is not None
        # Should leave NaN as is

    def test_compute_correlations_insufficient_columns(self):
        """Test correlation computation with insufficient valid columns."""
        df = pd.DataFrame(
            {
                "constant": [1, 1, 1, 1, 1],
                "high_nan": [None, None, None, None, None],
            }
        )

        frames = {"df": df}
        result = compute_correlations(frames)

        assert result == {"pearson": None, "spearman": None}

    def test_compute_correlations_duplicate_columns(self):
        """Test correlation computation with duplicate column names."""
        df1 = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4, 5],
            }
        )
        df2 = pd.DataFrame(
            {
                "feature": [2, 4, 6, 8, 10],  # Same column name
            }
        )

        frames = {"df1": df1, "df2": df2}
        result = compute_correlations(frames)

        assert result["pearson"] is not None
        # Should have prefixed column names
        assert any("df1_feature" in col for col in result["pearson"].columns)
        assert any("df2_feature" in col for col in result["pearson"].columns)
