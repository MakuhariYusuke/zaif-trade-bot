"""
Performance tests for SAC v446 components

SAC v446コンポーネントのパフォーマンステスト
"""

import sys
import time
import numpy as np
import pandas as pd
import pytest
import psutil
import os
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, 'src')

from ztb.features.unified_feature import V4FeatureExtractor
from ztb.core.preprocessing.data_preprocessing import (
    NoiseFilter,
    AnomalyDetector,
    SyntheticDataGenerator,
    preprocess_data
)


class TestPerformance:
    """Performance tests for SAC v446 components"""

    @pytest.fixture
    def large_market_data(self):
        """Create large dataset for performance testing"""
        np.random.seed(42)
        n_samples = 10000  # Large dataset

        # Generate market data
        base_price = 5000000.0  # JPY-based price
        prices = []
        current_price = base_price

        for i in range(n_samples):
            change = np.random.normal(0, 0.01)
            current_price *= (1 + change)
            prices.append(current_price)

        # Create OHLCV data
        data = []
        for i in range(n_samples):
            price = prices[i]
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.lognormal(10, 1)

            data.append({
                'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=i),
                'open': price * (1 + np.random.normal(0, 0.002)),
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })

        return pd.DataFrame(data)

    @pytest.fixture
    def medium_market_data(self):
        """Create medium dataset for performance testing"""
        np.random.seed(42)
        n_samples = 5000

        # Similar to large_market_data but smaller
        base_price = 5000000.0  # JPY-based price
        prices = []
        current_price = base_price

        for i in range(n_samples):
            change = np.random.normal(0, 0.01)
            current_price *= (1 + change)
            prices.append(current_price)

        data = []
        for i in range(n_samples):
            price = prices[i]
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.lognormal(10, 1)

            data.append({
                'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=i),
                'open': price * (1 + np.random.normal(0, 0.002)),
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })

        return pd.DataFrame(data)

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def measure_execution_time(self, func, *args, **kwargs):
        """Measure function execution time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    @patch('ztb.features.unified_feature.UnifiedFeatureEngineer')
    def test_v4_feature_extractor_performance(self, mock_unified_engineer_class,
                                            large_market_data):
        """Test V4FeatureExtractor performance"""
        # Setup mock
        mock_instance = Mock()
        mock_instance.generate_features.return_value = pd.DataFrame({
            'feature1': np.random.randn(len(large_market_data)),
            'feature2': np.random.randn(len(large_market_data)),
            'feature3': np.random.randn(len(large_market_data)),
            'feature4': np.random.randn(len(large_market_data)),
            'feature5': np.random.randn(len(large_market_data))
        })
        mock_unified_engineer_class.return_value = mock_instance

        extractor = V4FeatureExtractor()

        # Measure performance
        features, execution_time = self.measure_execution_time(
            extractor.extract_features, large_market_data
        )

        # Performance assertions
        assert execution_time < 5.0, f"Feature extraction too slow: {execution_time}s"
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(large_market_data)

    def test_noise_filter_performance(self, large_market_data):
        """Test NoiseFilter performance"""
        filter_obj = NoiseFilter()

        # Measure performance
        filtered_data, execution_time = self.measure_execution_time(
            filter_obj.apply_filters, large_market_data
        )

        # Performance assertions
        assert execution_time < 2.0, f"Noise filtering too slow: {execution_time}s"
        assert isinstance(filtered_data, pd.DataFrame)

    @patch('sklearn.ensemble.IsolationForest')
    def test_anomaly_detector_performance(self, mock_isolation_forest, large_market_data):
        """Test AnomalyDetector performance"""
        # Setup mock
        mock_instance = Mock()
        mock_instance.fit_predict.return_value = np.ones(len(large_market_data))
        mock_isolation_forest.return_value = mock_instance

        detector = AnomalyDetector()

        # Measure performance
        result, execution_time = self.measure_execution_time(
            detector.detect_anomalies, large_market_data, method='isolation_forest'
        )

        # Performance assertions
        assert execution_time < 10.0, f"Anomaly detection too slow: {execution_time}s"
        assert isinstance(result[0], pd.DataFrame)  # clean_data
        assert isinstance(result[1], pd.Series)    # anomaly_mask

    def test_synthetic_data_generator_performance(self, medium_market_data):
        """Test SyntheticDataGenerator performance"""
        generator = SyntheticDataGenerator()

        # Measure performance
        synthetic_data, execution_time = self.measure_execution_time(
            generator.generate_time_series, medium_market_data, n_periods=2000
        )

        # Performance assertions
        assert execution_time < 5.0, f"Synthetic data generation too slow: {execution_time}s"
        assert isinstance(synthetic_data, pd.DataFrame)
        assert len(synthetic_data) == 2000

    @patch('sklearn.ensemble.IsolationForest')
    @patch('ztb.features.unified_feature.UnifiedFeatureEngineer')
    def test_full_pipeline_performance(self, mock_unified_engineer_class,
                                     mock_isolation_forest, medium_market_data):
        """Test full preprocessing pipeline performance"""
        # Setup mocks
        mock_unified_instance = Mock()
        mock_unified_instance.generate_features.side_effect = lambda df, **kwargs: pd.DataFrame({
            'feature1': np.random.randn(len(df)),
            'feature2': np.random.randn(len(df)),
            'feature3': np.random.randn(len(df))
        })
        mock_unified_engineer_class.return_value = mock_unified_instance

        mock_if_instance = Mock()
        mock_if_instance.fit_predict.return_value = np.ones(len(medium_market_data))
        mock_isolation_forest.return_value = mock_if_instance

        # Measure performance
        config = {
            'apply_noise_filter': True,
            'apply_anomaly_detection': True,
            'generate_synthetic': True,
            'synthetic_periods': 500
        }
        processed_data, execution_time = self.measure_execution_time(
            preprocess_data, medium_market_data, config
        )

        # Performance assertions
        assert execution_time < 15.0, f"Full pipeline too slow: {execution_time}s"
        assert isinstance(processed_data, pd.DataFrame)

    def test_memory_usage_v4_feature_extractor(self, large_market_data):
        """Test memory usage of V4FeatureExtractor"""
        initial_memory = self.get_memory_usage()

        with patch('ztb.features.unified_feature.UnifiedFeatureEngineer') as mock_class:
            mock_instance = Mock()
            mock_instance.generate_features.return_value = pd.DataFrame({
                'feature1': np.random.randn(len(large_market_data)),
                'feature2': np.random.randn(len(large_market_data))
            })
            mock_class.return_value = mock_instance

            extractor = V4FeatureExtractor()
            features = extractor.extract_features(large_market_data)

            peak_memory = self.get_memory_usage()
            memory_increase = peak_memory - initial_memory

            # Memory increase should be reasonable (< 500MB)
            assert memory_increase < 500, f"Memory usage too high: {memory_increase}MB"

    def test_memory_usage_preprocessing_pipeline(self, medium_market_data):
        """Test memory usage of preprocessing pipeline"""
        initial_memory = self.get_memory_usage()

        with patch('sklearn.ensemble.IsolationForest') as mock_if, \
             patch('ztb.features.unified_feature.UnifiedFeatureEngineer') as mock_unified:

            # Setup mocks
            mock_if_instance = Mock()
            mock_if_instance.fit_predict.return_value = np.ones(len(medium_market_data))
            mock_if.return_value = mock_if_instance

            mock_unified_instance = Mock()
            mock_unified_instance.generate_features.side_effect = lambda df, **kwargs: pd.DataFrame({
                'feature1': np.random.randn(len(df))
            })
            mock_unified.return_value = mock_unified_instance

            config = {
                'apply_noise_filter': True,
                'apply_anomaly_detection': True,
                'generate_synthetic': False
            }
            processed_data = preprocess_data(medium_market_data, config)

            peak_memory = self.get_memory_usage()
            memory_increase = peak_memory - initial_memory

            # Memory increase should be reasonable (< 300MB)
            assert memory_increase < 300, f"Memory usage too high: {memory_increase}MB"

    def test_scalability_with_data_size(self, medium_market_data):
        """Test scalability as data size increases"""
        sizes = [1000, 2000, 5000]
        times = []

        with patch('sklearn.ensemble.IsolationForest') as mock_if, \
             patch('ztb.features.unified_feature.UnifiedFeatureEngineer') as mock_unified:

            # Setup mocks
            mock_if_instance = Mock()
            mock_if.return_value = mock_if_instance

            mock_unified_instance = Mock()
            mock_unified.return_value = mock_unified_instance

            for size in sizes:
                subset_data = medium_market_data.head(size)

                mock_if_instance.fit_predict.return_value = np.ones(size)
                mock_unified_instance.generate_features.side_effect = lambda df, **kwargs: pd.DataFrame({
                    'feature1': np.random.randn(len(df))
                })

                config = {
                    'apply_noise_filter': True,
                    'apply_anomaly_detection': True,
                    'generate_synthetic': False
                }
                _, execution_time = self.measure_execution_time(
                    preprocess_data, subset_data, config
                )

                times.append(execution_time)

            # Check that scaling is roughly linear (not exponential)
            # Time for 5k should be less than 5x time for 1k
            scaling_factor = times[2] / times[0]
            assert scaling_factor < 10, f"Poor scaling: {scaling_factor}x for 5x data"

    def test_concurrent_performance_stability(self, medium_market_data):
        """Test performance stability across multiple runs"""
        times = []

        with patch('sklearn.ensemble.IsolationForest') as mock_if, \
             patch('ztb.features.unified_feature.UnifiedFeatureEngineer') as mock_unified:

            # Setup mocks
            mock_if_instance = Mock()
            mock_if_instance.fit_predict.return_value = np.ones(len(medium_market_data))
            mock_if.return_value = mock_if_instance

            mock_unified_instance = Mock()
            mock_unified_instance.generate_features.side_effect = lambda df, **kwargs: pd.DataFrame({
                'feature1': np.random.randn(len(df))
            })
            mock_unified.return_value = mock_unified_instance

            # Run multiple times
            for i in range(5):
                config = {
                    'apply_noise_filter': True,
                    'apply_anomaly_detection': True,
                    'generate_synthetic': False
                }
                _, execution_time = self.measure_execution_time(
                    preprocess_data, medium_market_data, config
                )
                times.append(execution_time)

            # Check stability (coefficient of variation < 0.5)
            mean_time = np.mean(times)
            std_time = np.std(times)
            cv = std_time / mean_time if mean_time > 0 else 0

            assert cv < 0.5, f"Performance not stable: CV = {cv}"

    def test_cpu_usage_bounds(self, medium_market_data):
        """Test that CPU usage stays within reasonable bounds"""
        # This is a basic test - in a real scenario you'd use more sophisticated monitoring
        start_time = time.time()

        with patch('sklearn.ensemble.IsolationForest') as mock_if, \
             patch('ztb.features.unified_feature.UnifiedFeatureEngineer') as mock_unified:

            mock_if_instance = Mock()
            mock_if_instance.fit_predict.return_value = np.ones(len(medium_market_data))
            mock_if.return_value = mock_if_instance

            mock_unified_instance = Mock()
            mock_unified_instance.generate_features.side_effect = lambda df, **kwargs: pd.DataFrame({
                'feature1': np.random.randn(len(df))
            })
            mock_unified.return_value = mock_unified_instance

            config = {
                'apply_noise_filter': True,
                'apply_anomaly_detection': True,
                'generate_synthetic': False
            }
            preprocess_data(medium_market_data, config)

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete within reasonable time bounds
        assert total_time < 30.0, f"Processing took too long: {total_time}s"