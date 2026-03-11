"""
Performance tests for SAC v446 components (legacy filename, renamed to
avoid module basename collisions with other test packages).

This file is functionally identical to the original `tests/performance/test_perfor
mance.py` but uses a unique basename to prevent pytest import collisions during
collection.
"""

import pytest

pytest.skip(
    "Legacy performance benchmarks are environment-dependent and excluded from the maintained functional test baseline.",
    allow_module_level=True,
)

import os
import sys
# (Renamed and relocated content from the original performance test to avoid
# basename collisions with other test modules.)

import numpy as np
import pandas as pd
import psutil
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, "src")

from ztb.features.unified_feature import V4FeatureExtractor


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
            current_price *= 1 + change
            prices.append(current_price)

        # Create OHLCV data
        data = []
        for i in range(n_samples):
            price = prices[i]
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.lognormal(10, 1)

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
            current_price *= 1 + change
            prices.append(current_price)

        data = []
        for i in range(n_samples):
            price = prices[i]
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.lognormal(10, 1)

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

    @patch("ztb.features.unified_feature.UnifiedFeatureEngineer")
    def test_v4_feature_extractor_performance(
        self, mock_unified_engineer_class, large_market_data
    ):
        """Test V4FeatureExtractor performance"""
        # Setup mock
        mock_instance = Mock()
        mock_instance.generate_features.return_value = pd.DataFrame(
            {
                "feature1": np.random.randn(len(large_market_data)),
                "feature2": np.random.randn(len(large_market_data)),
                "feature3": np.random.randn(len(large_market_data)),
                "feature4": np.random.randn(len(large_market_data)),
                "feature5": np.random.randn(len(large_market_data)),
            }
        )
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
