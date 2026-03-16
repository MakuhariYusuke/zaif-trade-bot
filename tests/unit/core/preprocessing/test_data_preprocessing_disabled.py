"""
Unit tests for data preprocessing components

データ前処理コンポーネントの単体テスト
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ztb.core.preprocessing.data_preprocessing import (
    AnomalyDetector,
    NoiseFilter,
    SyntheticDataGenerator,
    preprocess_data,
)


class TestNoiseFilter:
    """NoiseFilter unit tests"""

    @pytest.fixture
    def sample_dataframe(self):
        """Sample dataframe with some noise"""
        np.random.seed(42)
        n_samples = 100

        # Create clean data
        clean_data = np.sin(
            np.linspace(0, 4 * np.pi, n_samples)
        ) + 0.1 * np.random.randn(n_samples)

        # Add some outliers
        data_with_noise = clean_data.copy()
        data_with_noise[10] = 10.0  # Outlier
        data_with_noise[50] = -8.0  # Outlier

        return pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2023-01-01", periods=n_samples, freq="1min"
                ),
                "price": data_with_noise,
                "volume": np.random.uniform(1000, 10000, n_samples),
            }
        )

    def test_initialization(self):
        """Test NoiseFilter initialization"""
        config = {"zscore_threshold": 2.5, "iqr_multiplier": 2.0}
        filter_obj = NoiseFilter(config=config)

        assert filter_obj.zscore_threshold == 2.5
        assert filter_obj.iqr_multiplier == 2.0

    def test_initialization_default_config(self):
        """Test NoiseFilter initialization with default config"""
        filter_obj = NoiseFilter()

        assert filter_obj.zscore_threshold == 3.0
        assert filter_obj.iqr_multiplier == 1.5

    def test_filter_zscore(self, sample_dataframe):
        """Test Z-score filtering"""
        filter_obj = NoiseFilter()
        result = filter_obj.filter_zscore(sample_dataframe, ["price"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
        assert "price" in result.columns

        # Check that extreme outliers were modified
        original_extreme = sample_dataframe["price"].iloc[10]  # Should be 10.0
        filtered_extreme = result["price"].iloc[10]
        assert abs(filtered_extreme - original_extreme) > 0.1  # Should be modified

    def test_filter_iqr(self, sample_dataframe):
        """Test IQR filtering"""
        filter_obj = NoiseFilter()
        result = filter_obj.filter_iqr(sample_dataframe, ["price"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)

        # Check that outliers were handled
        assert not np.isnan(result["price"]).any()  # No NaN values should remain

    def test_apply_filters(self, sample_dataframe):
        """Test applying all filters"""
        filter_obj = NoiseFilter()
        result = filter_obj.apply_filters(sample_dataframe)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
        assert list(result.columns) == list(sample_dataframe.columns)

    def test_apply_filters_specific_columns(self, sample_dataframe):
        """Test applying filters to specific columns"""
        filter_obj = NoiseFilter()
        result = filter_obj.apply_filters(sample_dataframe, ["price"])

        assert isinstance(result, pd.DataFrame)
        assert "price" in result.columns
        assert "volume" in result.columns  # Other columns should remain unchanged

    def test_apply_filters_no_numeric_columns(self):
        """Test applying filters with no numeric columns"""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1min"),
                "category": ["A", "B", "C"] * 3 + ["A"],
            }
        )

        filter_obj = NoiseFilter()
        result = filter_obj.apply_filters(df)

        # Should return original dataframe unchanged
        pd.testing.assert_frame_equal(result, df)


class TestAnomalyDetector:
    """AnomalyDetector unit tests"""

    @pytest.fixture
    def sample_dataframe(self):
        """Sample dataframe for anomaly detection"""
        np.random.seed(42)
        n_samples = 100

        return pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n_samples),
                "feature2": np.random.normal(5, 2, n_samples),
                "feature3": np.random.uniform(0, 10, n_samples),
            }
        )

    def test_initialization(self):
        """Test AnomalyDetector initialization"""
        config = {"contamination": 0.05}
        detector = AnomalyDetector(config=config)

        assert detector.config == config
        assert "isolation_forest" in detector.methods
        assert "local_outlier_factor" in detector.methods
        assert "statistical" in detector.methods

    def test_detect_anomalies_statistical(self, sample_dataframe):
        """Test statistical anomaly detection"""
        detector = AnomalyDetector()

        # Add some clear anomalies
        df_with_anomalies = sample_dataframe.copy()
        df_with_anomalies.loc[10, "feature1"] = 10.0  # Clear outlier
        df_with_anomalies.loc[50, "feature2"] = -20.0  # Clear outlier

        result_df, anomaly_mask = detector.detect_anomalies(
            df_with_anomalies, method="statistical"
        )

        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(anomaly_mask, pd.Series)
        assert len(result_df) == len(df_with_anomalies)
        assert len(anomaly_mask) == len(df_with_anomalies)

        # Should detect some anomalies
        assert anomaly_mask.sum() > 0

    @patch("sklearn.ensemble.IsolationForest")
    def test_detect_anomalies_isolation_forest(self, mock_iso_forest, sample_dataframe):
        """Test isolation forest anomaly detection"""
        # Mock sklearn components
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.array(
            [1] * 98 + [-1, -1]
        )  # Last 2 are anomalies
        mock_iso_forest.return_value = mock_instance

        detector = AnomalyDetector()
        result_df, anomaly_mask = detector.detect_anomalies(
            sample_dataframe, method="isolation_forest"
        )

        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(anomaly_mask, pd.Series)
        assert anomaly_mask.sum() == 2  # Should detect 2 anomalies

    @patch("sklearn.neighbors.LocalOutlierFactor")
    def test_detect_anomalies_lof(self, mock_lof, sample_dataframe):
        """Test LOF anomaly detection"""
        # Mock sklearn components
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.array(
            [1] * 95 + [-1] * 5
        )  # Last 5 are anomalies
        mock_lof.return_value = mock_instance

        detector = AnomalyDetector()
        result_df, anomaly_mask = detector.detect_anomalies(
            sample_dataframe, method="local_outlier_factor"
        )

        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(anomaly_mask, pd.Series)
        assert anomaly_mask.sum() == 5  # Should detect 5 anomalies

    def test_detect_anomalies_unknown_method(self, sample_dataframe):
        """Test unknown detection method falls back to statistical"""
        detector = AnomalyDetector()

        # Should not raise error, should use statistical method
        result_df, anomaly_mask = detector.detect_anomalies(
            sample_dataframe, method="unknown_method"
        )

        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(anomaly_mask, pd.Series)

    def test_detect_anomalies_specific_columns(self, sample_dataframe):
        """Test anomaly detection on specific columns"""
        detector = AnomalyDetector()

        result_df, anomaly_mask = detector.detect_anomalies(
            sample_dataframe, columns=["feature1", "feature2"]
        )

        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(anomaly_mask, pd.Series)
        # Should only consider specified columns for anomaly detection


class TestSyntheticDataGenerator:
    """SyntheticDataGenerator unit tests"""

    @pytest.fixture
    def sample_dataframe(self):
        """Sample dataframe for synthetic data generation"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="1min")

        return pd.DataFrame(
            {
                "timestamp": dates,
                "price": np.random.uniform(100, 110, 50),
                "volume": np.random.uniform(1000, 10000, 50),
            }
        )

    def test_initialization(self):
        """Test SyntheticDataGenerator initialization"""
        config = {"random_state": 123}
        generator = SyntheticDataGenerator(config=config)

        assert generator.config == config
        assert generator.random_state == 123

    def test_generate_gaussian_noise(self, sample_dataframe):
        """Test Gaussian noise generation"""
        generator = SyntheticDataGenerator()
        result = generator.generate_gaussian_noise(
            sample_dataframe, ["price"], noise_level=0.1
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
        assert "price" in result.columns
        assert "volume" in result.columns

        # Price should be modified with noise
        assert not result["price"].equals(sample_dataframe["price"])
        # Volume should remain unchanged
        pd.testing.assert_series_equal(result["volume"], sample_dataframe["volume"])

    def test_generate_time_series(self, sample_dataframe):
        """Test time series generation"""
        generator = SyntheticDataGenerator()
        result = generator.generate_time_series(sample_dataframe, n_periods=25)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 25
        assert "price" in result.columns
        assert "volume" in result.columns

        # Should have timestamp column
        assert "timestamp" in result.columns
        assert len(result["timestamp"]) == 25

    @patch("sklearn.neighbors.NearestNeighbors")
    def test_generate_smote_like(self, mock_nn, sample_dataframe):
        """Test SMOTE-like data generation"""
        # Mock sklearn components
        mock_instance = MagicMock()
        mock_instance.fit.return_value = None
        # Mock k_neighbors to return some indices
        mock_instance.kneighbors.return_value = (
            np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
            np.array([[1, 2, 3, 4, 5]]),
        )
        mock_nn.return_value = mock_instance

        generator = SyntheticDataGenerator()
        result = generator.generate_smote_like(sample_dataframe, "price", n_samples=10)

        assert isinstance(result, pd.DataFrame)
        # Should include original data plus synthetic data
        assert len(result) >= len(sample_dataframe)


class TestPreprocessData:
    """Test the main preprocess_data function"""

    @pytest.fixture
    def sample_dataframe(self):
        """Sample dataframe for preprocessing"""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="1min"),
                "price": np.random.uniform(100, 110, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            }
        )

    @patch("ztb.core.preprocessing.data_preprocessing.NoiseFilter")
    @patch("ztb.core.preprocessing.data_preprocessing.AnomalyDetector")
    @patch("ztb.core.preprocessing.data_preprocessing.SyntheticDataGenerator")
    def test_preprocess_data_all_enabled(
        self, mock_generator, mock_detector, mock_filter, sample_dataframe
    ):
        """Test preprocess_data with all features enabled"""
        # Setup mocks
        mock_filter_instance = MagicMock()
        mock_filter_instance.apply_filters.return_value = sample_dataframe
        mock_filter.return_value = mock_filter_instance

        mock_detector_instance = MagicMock()
        mock_detector_instance.detect_anomalies.return_value = (
            sample_dataframe,
            pd.Series(False, index=sample_dataframe.index),
        )
        mock_detector.return_value = mock_detector_instance

        mock_generator_instance = MagicMock()
        mock_generator_instance.generate_time_series.return_value = sample_dataframe
        mock_generator.return_value = mock_generator_instance

        config = {
            "apply_noise_filter": True,
            "apply_anomaly_detection": True,
            "generate_synthetic": True,
            "synthetic_periods": 50,
        }

        result = preprocess_data(sample_dataframe, config)

        assert isinstance(result, pd.DataFrame)
        # Verify all components were called
        mock_filter.assert_called_once()
        mock_detector.assert_called_once()
        mock_generator.assert_called_once()

    def test_preprocess_data_defaults(self, sample_dataframe):
        """Test preprocess_data with default configuration"""
        result = preprocess_data(sample_dataframe)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)

    def test_preprocess_data_empty_config(self, sample_dataframe):
        """Test preprocess_data with empty config"""
        result = preprocess_data(sample_dataframe, {})

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
