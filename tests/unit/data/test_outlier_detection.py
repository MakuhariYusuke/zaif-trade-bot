"""
Tests for outlier detection functionality.

This module tests various outlier detection methods and processing techniques.
"""

import numpy as np
import pandas as pd
import pytest

from ztb.data.outlier_detection import OutlierDetector


class TestOutlierDetector:
    """Test cases for OutlierDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = OutlierDetector(random_seed=42)

    def test_init_with_seed(self):
        """Test initialization with random seed."""
        detector = OutlierDetector(random_seed=123)
        assert detector.random_seed == 123

    def test_init_without_seed(self):
        """Test initialization without random seed."""
        detector = OutlierDetector()
        assert detector.random_seed is None

    def test_detect_outliers_z_score_method(self):
        """Test outlier detection using Z-score method."""
        # Create data with clear outliers
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # 100.0 is outlier

        result = self.detector.detect_outliers(data, method="z_score", threshold=2.0)

        assert isinstance(result, dict)
        assert "outlier_flags" in result
        assert "outlier_indices" in result
        assert "method" in result
        assert result["method"] == "z_score"
        assert len(result["outlier_flags"]) == len(data)
        assert sum(result["outlier_flags"]) > 0  # Should detect at least one outlier

    def test_detect_outliers_iqr_method(self):
        """Test outlier detection using IQR method."""
        # Create data with outliers
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0])

        result = self.detector.detect_outliers(data, method="iqr", multiplier=1.5)

        assert isinstance(result, dict)
        assert "outlier_flags" in result
        assert result["method"] == "iqr"
        assert len(result["outlier_flags"]) == len(data)

    def test_detect_outliers_modified_z_score_method(self):
        """Test outlier detection using modified Z-score method."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 50.0])

        result = self.detector.detect_outliers(
            data, method="modified_z_score", threshold=3.5
        )

        assert isinstance(result, dict)
        assert result["method"] == "modified_z_score"
        assert len(result["outlier_flags"]) == len(data)

    def test_detect_outliers_isolation_forest_method(self):
        """Test outlier detection using Isolation Forest method."""
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [100.0]])

        result = self.detector.detect_outliers(
            data, method="isolation_forest", contamination=0.1
        )

        assert isinstance(result, dict)
        assert result["method"] == "isolation_forest"
        assert len(result["outlier_flags"]) == len(data)

    def test_detect_outliers_lof_method(self):
        """Test outlier detection using Local Outlier Factor method."""
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [100.0]])

        result = self.detector.detect_outliers(data, method="lof", n_neighbors=2)

        assert isinstance(result, dict)
        assert result["method"] == "lof"
        assert len(result["outlier_flags"]) == len(data)

    def test_detect_outliers_stl_decomposition_method(self):
        """Test outlier detection using STL decomposition method."""
        # Create time series data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        trend = np.linspace(100, 200, 100)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(100) / 7)  # Weekly pattern
        noise = np.random.normal(0, 1, 100)
        data = trend + seasonal + noise
        data[50] = 500  # Add outlier

        ts_data = pd.Series(data, index=dates)

        result = self.detector.detect_outliers(ts_data, method="stl_decomposition")

        assert isinstance(result, dict)
        assert result["method"] == "stl_decomposition"
        assert len(result["outlier_flags"]) == len(ts_data)

    def test_detect_outliers_arima_residual_method(self):
        """Test outlier detection using ARIMA residual method."""
        # Create time series data
        np.random.seed(42)
        data = np.random.normal(100, 5, 50)
        data[25] = 200  # Add outlier

        result = self.detector.detect_outliers(
            data, method="arima_residual", order=(1, 0, 1)
        )

        assert isinstance(result, dict)
        assert result["method"] == "arima_residual"
        assert len(result["outlier_flags"]) == len(data)

    def test_detect_outliers_invalid_method(self):
        """Test outlier detection with invalid method."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError):
            self.detector.detect_outliers(data, method="invalid_method")

    def test_detect_outliers_empty_data(self):
        """Test outlier detection with empty data."""
        data = np.array([])

        result = self.detector.detect_outliers(data, method="z_score")

        assert isinstance(result, dict)
        assert len(result["outlier_flags"]) == 0
        assert len(result["outlier_indices"]) == 0

    def test_combine_outlier_flags(self):
        """Test combining outlier flags from multiple methods."""
        flags_dict = {
            "method1": np.array([True, False, True, False, False]),
            "method2": np.array([False, True, True, False, False]),
            "method3": np.array([True, True, False, False, False]),
        }

        combined_flags = self.detector._combine_outlier_flags(flags_dict)

        assert isinstance(combined_flags, np.ndarray)
        assert len(combined_flags) == 5
        assert combined_flags[0] == True  # Detected by method1 and method3
        assert combined_flags[1] == True  # Detected by method2 and method3
        assert combined_flags[2] == True  # Detected by method1 and method2

    def test_z_score_detection_normal_data(self):
        """Test Z-score detection with normal data."""
        np.random.seed(42)
        data = np.random.normal(100, 10, 100)

        outlier_flags, outlier_indices = self.detector._detect_z_score(
            data, threshold=3.0
        )

        assert isinstance(outlier_flags, np.ndarray)
        assert isinstance(outlier_indices, np.ndarray)
        assert len(outlier_flags) == len(data)
        # Normal data should have few outliers
        assert (
            sum(outlier_flags) <= 5
        )  # Allow up to 5 outliers for statistical variation

    def test_z_score_detection_with_outliers(self):
        """Test Z-score detection with known outliers."""
        data = np.array([100.0, 105.0, 95.0, 200.0, 98.0, 102.0])  # 200.0 is outlier

        outlier_flags, outlier_indices = self.detector._detect_z_score(
            data, threshold=2.0
        )

        assert isinstance(outlier_flags, np.ndarray)
        assert sum(outlier_flags) > 0  # Should detect the outlier

    def test_iqr_detection_with_outliers(self):
        """Test IQR detection with outliers."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0])

        outlier_flags, outlier_indices = self.detector._detect_iqr(data, multiplier=1.5)

        assert isinstance(outlier_flags, np.ndarray)
        assert sum(outlier_flags) > 0  # Should detect outliers

    def test_modified_z_score_detection(self):
        """Test modified Z-score detection."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 50.0])

        outlier_flags, outlier_indices = self.detector._detect_modified_z_score(
            data, threshold=3.5
        )

        assert isinstance(outlier_flags, np.ndarray)
        assert len(outlier_flags) == len(data)
