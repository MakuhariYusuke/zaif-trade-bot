"""Tests for drift_detection module."""

import numpy as np
import pandas as pd
import pytest

from ztb.utils.drift_detection import (
    calculate_psi,
    calculate_ks,
    detect_drift_single_feature,
    detect_drift_all_features,
)


class TestCalculatePSI:
    """Test PSI calculation."""
    
    def test_identical_distributions(self):
        """Identical distributions should have PSI ≈ 0."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)
        
        psi = calculate_psi(expected, actual)
        
        # Should be very small (close to 0)
        assert psi < 0.1
    
    def test_shifted_distribution(self):
        """Shifted distribution should have high PSI."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(2, 1, 1000)  # Mean shifted by 2
        
        psi = calculate_psi(expected, actual)
        
        # Should be high (>= 0.2)
        assert psi > 0.2
    
    def test_constant_values(self):
        """Constant values should have PSI = 0."""
        expected = np.ones(1000)
        actual = np.ones(1000)
        
        psi = calculate_psi(expected, actual)
        
        assert psi == 0.0
    
    def test_nan_handling(self):
        """NaN values should be ignored."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)
        actual[::10] = np.nan  # Add some NaNs
        
        psi = calculate_psi(expected, actual)
        
        # Should still work
        assert not np.isnan(psi)
        assert psi < 0.1


class TestCalculateKS:
    """Test KS test calculation."""
    
    def test_identical_distributions(self):
        """Identical distributions should have high p-value."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)
        
        stat, p = calculate_ks(expected, actual)
        
        # p-value should be high (distributions are similar)
        assert p > 0.01
    
    def test_different_distributions(self):
        """Different distributions should have low p-value."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(2, 1, 1000)  # Mean shifted by 2
        
        stat, p = calculate_ks(expected, actual)
        
        # p-value should be very low (distributions are different)
        assert p < 0.01
    
    def test_statistic_range(self):
        """KS statistic should be between 0 and 1."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(1, 1, 1000)
        
        stat, p = calculate_ks(expected, actual)
        
        assert 0 <= stat <= 1


class TestDetectDriftSingleFeature:
    """Test drift detection for single feature."""
    
    def test_no_drift(self):
        """No drift case."""
        np.random.seed(42)
        train_values = np.random.normal(0, 1, 1000)
        eval_values = np.random.normal(0, 1, 1000)
        
        result = detect_drift_single_feature(
            train_values, eval_values, "test_feature"
        )
        
        assert result["drift_detected"] is False
        assert result["psi_drift"] is False
        assert result["ks_drift"] is False
        assert result["feature_name"] == "test_feature"
    
    def test_psi_drift(self):
        """PSI drift case."""
        np.random.seed(42)
        train_values = np.random.normal(0, 1, 1000)
        eval_values = np.random.normal(2, 1, 1000)  # Shifted
        
        result = detect_drift_single_feature(
            train_values, eval_values, "test_feature"
        )
        
        assert result["drift_detected"] is True
        assert result["psi_drift"] is True
        assert result["psi"] > 0.2
    
    def test_ks_drift(self):
        """KS drift case."""
        np.random.seed(42)
        train_values = np.random.normal(0, 1, 1000)
        eval_values = np.random.normal(1.5, 1, 1000)  # Shifted
        
        result = detect_drift_single_feature(
            train_values, eval_values, "test_feature"
        )
        
        assert result["drift_detected"] is True
        assert result["ks_drift"] is True
        assert result["ks_p_value"] < 0.01
    
    def test_statistics_included(self):
        """Result should include basic statistics."""
        np.random.seed(42)
        train_values = np.random.normal(0, 1, 1000)
        eval_values = np.random.normal(0, 1, 1000)
        
        result = detect_drift_single_feature(
            train_values, eval_values, "test_feature"
        )
        
        assert "train_mean" in result
        assert "eval_mean" in result
        assert "train_std" in result
        assert "eval_std" in result
        
        # Should be approximately 0 and 1
        assert -0.2 < result["train_mean"] < 0.2
        assert 0.9 < result["train_std"] < 1.1


class TestDetectDriftAllFeatures:
    """Test drift detection for multiple features."""
    
    def test_multiple_features(self):
        """Detect drift across multiple features."""
        np.random.seed(42)
        
        # Create train dataset
        train_df = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.normal(5, 2, 1000),
            "feature3": np.random.normal(10, 3, 1000),
        })
        
        # Create eval dataset (feature2 drifted)
        eval_df = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 1000),  # No drift
            "feature2": np.random.normal(8, 2, 1000),  # Drifted
            "feature3": np.random.normal(10, 3, 1000),  # No drift
        })
        
        result_df = detect_drift_all_features(train_df, eval_df)
        
        assert len(result_df) == 3
        assert "feature1" in result_df["feature_name"].values
        assert "feature2" in result_df["feature_name"].values
        assert "feature3" in result_df["feature_name"].values
        
        # feature2 should have drift
        feature2_row = result_df[result_df["feature_name"] == "feature2"].iloc[0]
        assert feature2_row["drift_detected"] == True  # Use == for boolean comparison
    
    def test_empty_dataframe(self):
        """Empty dataframes should return empty result."""
        train_df = pd.DataFrame()
        eval_df = pd.DataFrame()
        
        result_df = detect_drift_all_features(train_df, eval_df)
        
        assert len(result_df) == 0
    
    def test_mismatched_columns(self):
        """Should only process common columns."""
        train_df = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(5, 2, 100),
        })
        
        eval_df = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature3": np.random.normal(10, 3, 100),  # Different feature
        })
        
        result_df = detect_drift_all_features(train_df, eval_df)
        
        # Should only process feature1 (common to both)
        assert len(result_df) == 1
        assert result_df.iloc[0]["feature_name"] == "feature1"


class TestCustomThresholds:
    """Test custom thresholds."""
    
    def test_custom_psi_threshold(self):
        """Custom PSI threshold."""
        np.random.seed(42)
        train_values = np.random.normal(0, 1, 1000)
        eval_values = np.random.normal(0.5, 1, 1000)  # Slight shift
        
        # Default threshold (0.2)
        result_default = detect_drift_single_feature(
            train_values, eval_values, "test", psi_threshold=0.2
        )
        
        # Strict threshold (0.05)
        result_strict = detect_drift_single_feature(
            train_values, eval_values, "test", psi_threshold=0.05
        )
        
        # Strict threshold more likely to detect drift
        if result_strict["psi_drift"]:
            # If strict detects drift, default might or might not
            assert True
        else:
            # If strict doesn't detect drift, default shouldn't either
            assert not result_default["psi_drift"]
    
    def test_custom_ks_threshold(self):
        """Custom KS p-value threshold."""
        np.random.seed(42)
        train_values = np.random.normal(0, 1, 1000)
        eval_values = np.random.normal(0.3, 1, 1000)  # Slight shift
        
        # Default threshold (p < 0.01)
        result_default = detect_drift_single_feature(
            train_values, eval_values, "test", ks_p_threshold=0.01
        )
        
        # Lenient threshold (p < 0.1)
        result_lenient = detect_drift_single_feature(
            train_values, eval_values, "test", ks_p_threshold=0.1
        )
        
        # Lenient threshold more likely to detect drift
        if result_default["ks_drift"]:
            # If default detects drift, lenient must too
            assert result_lenient["ks_drift"]
