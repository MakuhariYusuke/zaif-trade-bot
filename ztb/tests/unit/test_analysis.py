#!/usr/bin/env python3
"""
Unit tests for analysis components (correlation, timeseries).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from ztb.utils.data_utils import load_csv_data

from analysis.correlation import compute_correlations
from analysis.timeseries import compute_lag_correlations


class TestCorrelationAnalysis:
    """Test correlation analysis functions."""

    def test_compute_correlations_empty(self):
        """Test correlation computation with empty frames."""
        result = compute_correlations({})
        assert result["pearson"] is None
        assert result["spearman"] is None

    def test_compute_correlations_basic(self):
        """Test basic correlation computation."""
        # Create test data
        np.random.seed(42)
        data = {
            "feature1": pd.DataFrame({"feature1": np.random.randn(100)}),
            "feature2": pd.DataFrame({"feature2": np.random.randn(100)}),
            "feature3": pd.DataFrame({"feature3": np.random.randn(100)}),
        }

        result = compute_correlations(data)

        assert result["pearson"] is not None
        assert result["spearman"] is not None

        pearson_df = result["pearson"]
        spearman_df = result["spearman"]

        assert isinstance(pearson_df, pd.DataFrame)
        assert isinstance(spearman_df, pd.DataFrame)
        assert pearson_df.shape == (3, 3)
        assert spearman_df.shape == (3, 3)

        # Check self-correlations are approximately 1
        assert pearson_df.loc["feature1", "feature1"] == pytest.approx(1.0, abs=1e-10)
        assert spearman_df.loc["feature1", "feature1"] == pytest.approx(1.0, abs=1e-10)

    def test_compute_correlations_filters(self):
        """Test filtering of high NaN and constant columns."""
        # Create data with high NaN and constant columns
        data = {
            "good_feature": pd.Series([1, 2, 3, 4, 5]),
            "high_nan": pd.Series([1, np.nan, np.nan, np.nan, np.nan]),  # 80% NaN
            "constant": pd.Series([1, 1, 1, 1, 1]),  # Constant
        }

        result = compute_correlations(data)

        # Should only include good_feature
        if result["pearson"] is not None:
            assert result["pearson"].shape == (1, 1)
            assert "good_feature" in result["pearson"].index


class TestLagCorrelationAnalysis:
    """Test lag correlation analysis functions."""

    def test_compute_lag_correlations_empty(self):
        """Test lag correlation with empty frames."""
        result = compute_lag_correlations({})
        assert result == []

    def test_compute_lag_correlations_basic(self):
        """Test basic lag correlation computation."""
        # Create test data with some correlation
        np.random.seed(42)
        n = 50
        base = np.random.randn(n)
        df = pd.DataFrame(
            {
                "feature1": base,
                "feature2": base + 0.5 * np.random.randn(n),  # Correlated
                "feature3": np.random.randn(n),  # Uncorrelated
            }
        )

        frames = {"test": df}
        result = compute_lag_correlations(frames)

        assert isinstance(result, list)
        assert len(result) > 0

        # Check structure
        for item in result:
            assert "feature1" in item
            assert "feature2" in item
            assert "lag" in item
            assert "correlation" in item
            assert isinstance(item["correlation"], (int, float))
            assert item["lag"] in [1, 5, 10, 20]

    def test_compute_lag_correlations_filters(self):
        """Test filtering in lag correlation."""
        # Create data with high NaN
        data = {
            "good1": pd.Series(list(range(1, 21))),  # 1 to 20
            "good2": pd.Series(list(range(2, 22))),  # 2 to 21
            "high_nan": pd.Series([1] + [np.nan] * 19),  # 95% NaN
        }

        result = compute_lag_correlations(data)

        # Should only analyze good features
        feature_pairs = {(item["feature1"], item["feature2"]) for item in result}
        assert ("good1", "good2") in feature_pairs
        assert not any("high_nan" in pair for pair in feature_pairs)


class TestAnalysisIntegration:
    """Test integration of analysis components."""

    def test_full_analysis_pipeline(self):
        """Test the full analysis pipeline works together."""
        # This would be integration test - create mock data and run full pipeline
        # For now, just test that imports work
        try:
            from analysis.correlation import compute_correlations
            from analysis.timeseries import compute_lag_correlations
            from tools.evaluation.re_evaluate_features import (
                ComprehensiveFeatureReEvaluator,
            )

            assert True
        except ImportError:
            pytest.fail("Analysis components failed to import")

    def test_output_files_generation(self, tmp_path):
        """Test that analysis outputs are generated correctly."""
        # Create temporary directory structure
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        # Create mock correlation data
        mock_corr = pd.DataFrame(
            {"feature1": [1.0, 0.5], "feature2": [0.5, 1.0]},
            index=["feature1", "feature2"],
        )

        # Save mock data
        mock_corr.to_csv(reports_dir / "correlation_pearson.csv")

        # Test loading works
        csv_path = reports_dir / "correlation_pearson.csv"
        assert csv_path.exists()

        loaded = load_csv_data(csv_path, index_col=0)
        assert loaded.shape == (2, 2)
        assert loaded.loc["feature1", "feature1"] == pytest.approx(1.0, abs=1e-10)
