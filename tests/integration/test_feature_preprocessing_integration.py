"""
Integration tests for SAC v446 components

SAC v446コンポーネントの統合テスト
"""

import sys
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, "src")

from ztb.features.unified_feature import V4FeatureExtractor

# from ztb.core.preprocessing.data_preprocessing import (
#     NoiseFilter,
#     AnomalyDetector,
#     SyntheticDataGenerator,
#     preprocess_data
# )


class TestFeaturePreprocessingIntegration:
    """Integration tests for feature extraction and preprocessing"""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        np.random.seed(42)
        n_samples = 1000

        # Generate realistic market data
        base_price = 100.0
        prices = []
        current_price = base_price

        for i in range(n_samples):
            # Random walk with some trend
            change = np.random.normal(0, 0.01)  # 1% volatility
            current_price *= 1 + change
            prices.append(current_price)

        # Create OHLCV data
        data = []
        for i in range(n_samples):
            price = prices[i]
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.lognormal(10, 1)  # Log-normal volume

            data.append(
                {
                    "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=i),
                    "open": price * (1 + np.random.normal(0, 0.002)),
                    "high": high,
                    "low": low,
                    "close": price,
                    "volume": volume,
                }
            )

        return pd.DataFrame(data)

    @pytest.fixture
    def mock_unified_feature_engineer(self):
        """Mock UnifiedFeatureEngineer for testing"""
        mock_engineer = Mock()
        mock_engineer.generate_features.return_value = pd.DataFrame(
            {
                "feature1": np.random.randn(1000),
                "feature2": np.random.randn(1000),
                "feature3": np.random.randn(1000),
                "feature4": np.random.randn(1000),
                "feature5": np.random.randn(1000),
            }
        )
        return mock_engineer

    def test_v4_feature_extractor_initialization(self, sample_market_data):
        """Test V4FeatureExtractor initialization"""
        extractor = V4FeatureExtractor()

        assert extractor is not None
        assert hasattr(extractor, "extract_features")

    @patch("ztb.features.unified_feature.UnifiedFeatureEngineer")
    def test_v4_feature_extractor_with_mock(
        self, mock_unified_engineer_class, sample_market_data
    ):
        """Test V4FeatureExtractor with mocked UnifiedFeatureEngineer"""
        # Setup mock
        mock_instance = Mock()
        mock_instance.generate_features.return_value = pd.DataFrame(
            {
                "feature1": np.random.randn(len(sample_market_data)),
                "feature2": np.random.randn(len(sample_market_data)),
                "feature3": np.random.randn(len(sample_market_data)),
                "feature4": np.random.randn(len(sample_market_data)),
                "feature5": np.random.randn(len(sample_market_data)),
            }
        )
        mock_unified_engineer_class.return_value = mock_instance

        # Test extraction
        extractor = V4FeatureExtractor()
        features = extractor.extract_features(sample_market_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_market_data)
        assert "feature1" in features.columns
        assert "feature2" in features.columns

        # Verify mock was called with correct parameters
        mock_unified_engineer_class.assert_called_once()
        mock_instance.generate_features.assert_called_once_with(
            sample_market_data, feature_set="curated", model_type="sac"
        )

    def test_noise_filter_integration(self, sample_market_data):
        """Test NoiseFilter integration"""
        # Add some noise to the data
        noisy_data = sample_market_data.copy()
        noisy_data["close"] += np.random.normal(0, 0.1, len(noisy_data))

        filter_obj = NoiseFilter()
        filtered_data = filter_obj.apply_filters(noisy_data)

        assert isinstance(filtered_data, pd.DataFrame)
        assert len(filtered_data) <= len(noisy_data)  # May remove some data
        assert all(col in filtered_data.columns for col in noisy_data.columns)

    @patch("sklearn.ensemble.IsolationForest")
    def test_anomaly_detector_integration(
        self, mock_isolation_forest, sample_market_data
    ):
        """Test AnomalyDetector integration with mocked sklearn"""
        # Setup mock
        mock_instance = Mock()
        mock_instance.fit_predict.return_value = np.ones(len(sample_market_data))
        mock_isolation_forest.return_value = mock_instance

        detector = AnomalyDetector()
        clean_data, anomaly_mask = detector.detect_anomalies(
            sample_market_data, method="isolation_forest"
        )

        assert isinstance(clean_data, pd.DataFrame)
        assert isinstance(anomaly_mask, pd.Series)
        assert len(clean_data) == len(sample_market_data)

        # Verify sklearn was used
        mock_isolation_forest.assert_called_once()
        mock_instance.fit_predict.assert_called_once()

    @patch("sklearn.neighbors.LocalOutlierFactor")
    def test_anomaly_detector_lof_integration(self, mock_lof, sample_market_data):
        """Test AnomalyDetector with LOF method"""
        # Setup mock
        mock_instance = Mock()
        mock_instance.fit_predict.return_value = np.ones(len(sample_market_data))
        mock_lof.return_value = mock_instance

        detector = AnomalyDetector()
        clean_data, anomaly_mask = detector.detect_anomalies(
            sample_market_data, method="local_outlier_factor"
        )

        assert isinstance(clean_data, pd.DataFrame)
        mock_lof.assert_called_once()

    def test_synthetic_data_generator_integration(self, sample_market_data):
        """Test SyntheticDataGenerator integration"""
        generator = SyntheticDataGenerator()
        synthetic_data = generator.generate_time_series(
            sample_market_data, n_periods=500
        )

        assert isinstance(synthetic_data, pd.DataFrame)
        assert len(synthetic_data) == 500
        assert all(
            col in synthetic_data.columns
            for col in ["open", "high", "low", "close"]
            if col in sample_market_data.columns
        )

        # Check synthetic data has similar statistics
        for col in ["open", "high", "low", "close"]:
            real_mean = sample_market_data[col].mean()
            synth_mean = synthetic_data[col].mean()
            # Should be reasonably close (within 20% relative difference)
            assert abs(real_mean - synth_mean) / real_mean < 0.2

    @patch("sklearn.ensemble.IsolationForest")
    @patch("ztb.features.unified_feature.UnifiedFeatureEngineer")
    def test_full_preprocessing_pipeline(
        self, mock_unified_engineer_class, mock_isolation_forest, sample_market_data
    ):
        """Test full preprocessing pipeline integration"""
        # Setup mocks
        mock_unified_instance = Mock()
        # Mock will be called with processed data (which may include synthetic data)
        mock_unified_instance.generate_features.side_effect = (
            lambda df, **kwargs: pd.DataFrame(
                {
                    "feature1": np.random.randn(len(df)),
                    "feature2": np.random.randn(len(df)),
                    "feature3": np.random.randn(len(df)),
                }
            )
        )
        mock_unified_engineer_class.return_value = mock_unified_instance

        mock_if_instance = Mock()
        mock_if_instance.fit_predict.return_value = np.ones(len(sample_market_data))
        mock_isolation_forest.return_value = mock_if_instance

        # Test preprocessing
        config = {
            "apply_noise_filter": True,
            "apply_anomaly_detection": True,
            "generate_synthetic": True,
            "synthetic_periods": 200,
        }
        processed_data = preprocess_data(sample_market_data, config)

        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) >= len(
            sample_market_data
        )  # May include synthetic data

        # Test feature extraction separately
        extractor = V4FeatureExtractor()
        features = extractor.extract_features(processed_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(
            processed_data
        )  # Should match processed data length

    def test_preprocessing_pipeline_error_handling(self, sample_market_data):
        """Test error handling in preprocessing pipeline"""
        # Test with invalid data
        invalid_data = sample_market_data.copy()
        invalid_data["close"] = np.nan  # Add NaN values

        # Should handle NaN values gracefully
        config = {"apply_noise_filter": True, "apply_anomaly_detection": True}
        processed_data = preprocess_data(invalid_data, config)

        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0

    def test_feature_preprocessing_consistency(self, sample_market_data):
        """Test consistency between feature extraction and preprocessing"""
        # Create multiple runs with same seed
        np.random.seed(42)

        # First run
        config1 = {
            "apply_noise_filter": True,
            "apply_anomaly_detection": True,
            "generate_synthetic": False,
        }
        processed_data1 = preprocess_data(sample_market_data, config1)

        np.random.seed(42)

        # Second run
        config2 = {
            "apply_noise_filter": True,
            "apply_anomaly_detection": True,
            "generate_synthetic": False,
        }
        processed_data2 = preprocess_data(sample_market_data, config2)

        # Results should be identical with same seed
        pd.testing.assert_frame_equal(processed_data1, processed_data2)

    def test_memory_efficiency_integration(self, sample_market_data):
        """Test memory efficiency of integration operations"""
        # Large dataset
        large_data = pd.concat([sample_market_data] * 10, ignore_index=True)

        # Should not crash with reasonable memory usage
        config = {
            "apply_noise_filter": True,
            "apply_anomaly_detection": True,
            "generate_synthetic": False,
        }
        processed_data = preprocess_data(large_data, config)

        assert len(processed_data) > 0

        # Clean up
        del large_data, processed_data

    def test_data_types_preservation(self, sample_market_data):
        """Test that data types are preserved through pipeline"""
        original_dtypes = sample_market_data.dtypes

        config = {
            "apply_noise_filter": True,
            "apply_anomaly_detection": True,
            "generate_synthetic": False,
        }
        processed_data = preprocess_data(sample_market_data, config)

        # Check that numeric columns maintain numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            if col in processed_data.columns:
                assert np.issubdtype(
                    processed_data[col].dtype, np.number
                ), f"Column {col} should remain numeric"
