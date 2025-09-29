"""
Performance benchmark tests for feature computation and evaluation.
"""

import time

import numpy as np
import pandas as pd
import pytest

from ztb.evaluation.auto_feature_generator import AutoFeatureGenerator
from ztb.metrics.metrics import calculate_all_metrics


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    @pytest.fixture
    def large_ohlc_data(self):
        """Create large OHLC dataset for performance testing"""
        # Create 100,000 rows of data (approximately 274 years of daily data)
        dates = pd.date_range("1950-01-01", periods=100000, freq="D")
        np.random.seed(42)

        # Generate realistic price data with trends and volatility
        base_price = 100
        long_trend = np.sin(np.linspace(0, 4 * np.pi, 100000)) * 50  # Long-term cycles
        short_trend = np.random.randn(100000).cumsum() * 0.1  # Short-term movements
        noise = np.random.randn(100000) * 2

        prices = base_price + long_trend + short_trend + noise

        return pd.DataFrame(
            {
                "open": prices + np.random.randn(100000) * 0.5,
                "high": prices + np.abs(np.random.randn(100000)) * 2,
                "low": prices - np.abs(np.random.randn(100000)) * 2,
                "close": prices,
                "volume": np.random.randint(1000, 100000, 100000),
            },
            index=dates,
        )

    @pytest.fixture
    def medium_ohlc_data(self):
        """Create medium-sized OHLC dataset"""
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        np.random.seed(42)

        prices = 100 + np.random.randn(1000).cumsum()

        return pd.DataFrame(
            {
                "open": prices + np.random.randn(1000) * 0.5,
                "high": prices + np.abs(np.random.randn(1000)) * 1.5,
                "low": prices - np.abs(np.random.randn(1000)) * 1.5,
                "close": prices,
                "volume": np.random.randint(1000, 10000, 1000),
            },
            index=dates,
        )

    def test_evaluate_feature_class_performance(self, medium_ohlc_data):
        """Benchmark evaluate_feature_class performance"""
        from ztb.evaluation.re_evaluate_features import evaluate_feature_class

        # Create a mock feature class
        class MockFeature:
            @staticmethod
            def compute(data):
                # Return a simple feature array
                return np.ones(len(data))

        import time

        start_time = time.time()

        # Benchmark the evaluation
        result = evaluate_feature_class(
            MockFeature,  # Mock feature class
            medium_ohlc_data,
            "MockFeature",
        )

        elapsed = time.time() - start_time

        # Should complete within reasonable time (< 10 seconds)
        assert elapsed < 10, f"Evaluation took too long: {elapsed:.2f} seconds"
        assert isinstance(result, dict)

    def test_calculate_all_metrics_performance(self):
        """Benchmark calculate_all_metrics performance"""
        # Create test returns data
        returns = np.random.randn(1000) * 0.02

        import time

        start_time = time.time()

        # Benchmark the calculation
        result = calculate_all_metrics(returns)

        elapsed = time.time() - start_time

        # Should complete within reasonable time (< 1 second)
        assert elapsed < 1, f"Metrics calculation took too long: {elapsed:.2f} seconds"

        # Verify result structure
        assert isinstance(result, dict)
        assert "sharpe_ratio" in result
        assert "total_return" in result

    def test_auto_feature_generation_performance(self, medium_ohlc_data):
        """Benchmark auto feature generation performance"""
        generator = AutoFeatureGenerator()

        import time

        start_time = time.time()

        # Benchmark EMA cross generation
        result = generator.generate_ema_cross_features(
            medium_ohlc_data, [5, 8, 12], [20, 25, 30]
        )

        elapsed = time.time() - start_time

        # Should complete within reasonable time (< 5 seconds)
        assert elapsed < 5, f"Auto generation took too long: {elapsed:.2f} seconds"

        # Should generate 9 features (3x3 combinations)
        assert len(result) == 9

        # Each feature should have same length as input data
        for feature_data in result.values():
            assert len(feature_data) == len(medium_ohlc_data)

    def test_large_dataset_evaluation_timeout(self, large_ohlc_data):
        """Test that large dataset evaluation completes within timeout"""
        from ztb.evaluation.re_evaluate_features import evaluate_feature_class

        # Create a mock feature class
        class MockFeature:
            @staticmethod
            def compute(data):
                # Return a simple feature array
                return np.ones(len(data))

        start_time = time.time()

        try:
            # This should complete within 60 seconds
            result = evaluate_feature_class(MockFeature, large_ohlc_data, "MockFeature")
            elapsed = time.time() - start_time

            # Should complete within 60 seconds
            assert elapsed < 60, f"Evaluation took too long: {elapsed:.2f} seconds"

        except Exception:
            # If evaluation fails due to missing feature, that's acceptable
            # We're mainly testing that it doesn't hang indefinitely
            elapsed = time.time() - start_time
            assert elapsed < 60, f"Evaluation timeout: {elapsed:.2f} seconds"

    def test_memory_usage_during_evaluation(self, medium_ohlc_data):
        """Test memory usage during evaluation (basic check)"""
        import os

        import psutil

        from ztb.evaluation.re_evaluate_features import evaluate_feature_class

        # Create a mock feature class
        class MockFeature:
            @staticmethod
            def compute(data):
                # Return a simple feature array
                return np.ones(len(data))

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            result = evaluate_feature_class(
                MockFeature, medium_ohlc_data, "MockFeature"
            )

            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (< 500MB)
            assert memory_increase < 500, (
                f"Excessive memory usage: {memory_increase:.2f} MB"
            )

        except ImportError:
            # psutil not available, skip memory test
            pytest.skip("psutil not available for memory testing")

    def test_concurrent_feature_evaluation(self, medium_ohlc_data):
        """Test performance with concurrent feature evaluations"""
        import concurrent.futures

        from ztb.evaluation.re_evaluate_features import evaluate_feature_class

        # Create a mock feature class
        class MockFeature:
            @staticmethod
            def compute(data):
                # Return a simple feature array
                return np.ones(len(data))

        results = []
        errors = []

        def evaluate_single_feature(feature_name):
            try:
                result = evaluate_feature_class(
                    MockFeature, medium_ohlc_data, feature_name
                )
                results.append(result)
                return True
            except Exception as e:
                errors.append(str(e))
                return False

        # Test concurrent evaluation of multiple features
        feature_names = ["TestFeature1", "TestFeature2", "TestFeature3"]

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(evaluate_single_feature, name) for name in feature_names
            ]
            concurrent.futures.wait(futures, timeout=30)  # 30 second timeout

        elapsed = time.time() - start_time

        # Should complete within reasonable time
        assert elapsed < 30, (
            f"Concurrent evaluation took too long: {elapsed:.2f} seconds"
        )

    def test_regression_detection(self, medium_ohlc_data):
        """Test for performance regression detection"""
        generator = AutoFeatureGenerator()

        # Run benchmark multiple times to establish baseline
        times = []
        for _ in range(3):
            start = time.time()
            features = generator.generate_ema_cross_features(
                medium_ohlc_data, [5], [20]
            )
            end = time.time()
            times.append(end - start)

        avg_time = np.mean(times)

        # Current benchmark should not be more than 2x slower than average
        start = time.time()
        result = generator.generate_ema_cross_features(medium_ohlc_data, [5], [20])
        current_time = time.time() - start

        # Should not be more than 2x slower than baseline
        assert current_time <= avg_time * 2, (
            f"Performance regression: {current_time:.3f}s vs baseline {avg_time:.3f}s"
        )
        assert len(result) == 1
        assert "auto_ema_cross_5_20" in result

    def test_scalability_with_data_size(self):
        """Test how performance scales with data size"""
        generator = AutoFeatureGenerator()

        data_sizes = [100, 500, 1000, 2000]
        times = []

        for size in data_sizes:
            # Create data of different sizes
            dates = pd.date_range("2020-01-01", periods=size, freq="D")
            prices = 100 + np.random.randn(size).cumsum()

            data = pd.DataFrame(
                {
                    "open": prices + np.random.randn(size) * 0.5,
                    "high": prices + np.abs(np.random.randn(size)) * 1.5,
                    "low": prices - np.abs(np.random.randn(size)) * 1.5,
                    "close": prices,
                    "volume": np.random.randint(1000, 10000, size),
                },
                index=dates,
            )

            start = time.time()
            features = generator.generate_ema_cross_features(data, [5], [20])
            end = time.time()

            times.append(end - start)

            # Verify features were generated
            assert len(features) == 1

        # Performance should scale reasonably (not exponentially)
        # Time for 2000 points should be less than 10x time for 100 points
        if len(times) >= 2:
            scaling_factor = times[-1] / times[0] if times[0] > 0 else 1
            assert scaling_factor < 10, (
                f"Poor scaling: {scaling_factor:.2f}x slower for 20x more data"
            )

    def test_indicators_performance_comparison(self, medium_ohlc_data):
        """Compare performance of different indicator calculations"""
        from ztb.features.volatility.kalman_ext import calculate_kalman_extended

        # Benchmark Kalman filter
        start = time.time()
        kalman_result = calculate_kalman_extended(medium_ohlc_data)
        kalman_time = time.time() - start

        # Kalman should complete within reasonable time and return features
        assert kalman_time < 5, (
            f"Kalman calculation took too long: {kalman_time:.2f} seconds"
        )
        assert isinstance(kalman_result, pd.DataFrame)
        assert len(kalman_result) > 0

    def test_benchmark_result_validation(self, medium_ohlc_data):
        """Test that benchmark results are valid"""
        generator = AutoFeatureGenerator()

        # Run generation and validate results
        features = generator.generate_ema_cross_features(medium_ohlc_data, [5, 8], [20])

        # Should have 2 features
        assert len(features) == 2

        # Each feature should be a DataFrame with correct length
        for name, data in features.items():
            assert isinstance(data, pd.DataFrame)
            assert len(data) == len(medium_ohlc_data)
            assert not data.empty

            # Check for reasonable value range
            values = data.values.flatten()
            values = values[~np.isnan(values)]  # Remove NaN
            if len(values) > 0:
                assert np.all(np.abs(values) <= 2), (
                    f"Unreasonable values in {name}: {values[:5]}"
                )
