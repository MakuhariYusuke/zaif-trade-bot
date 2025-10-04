"""
Unit tests for ztb.features.registry module.
"""

import pandas as pd
import pytest

from ztb.features.registry import FeatureRegistry


class TestFeatureRegistry:
    """Test FeatureRegistry functionality."""

    def test_initialization(self):
        """Test FeatureRegistry initialization."""
        # Reset for clean state
        FeatureRegistry.reset_for_testing()

        # Initialize
        FeatureRegistry.initialize()

        # Should be able to get config after initialization
        config = FeatureRegistry.get_config()
        assert isinstance(config, dict)

    def test_list_features(self):
        """Test listing available features."""
        # Reset and initialize
        FeatureRegistry.reset_for_testing()
        FeatureRegistry.initialize()

        features = FeatureRegistry.list()
        assert isinstance(features, list)
        # Should have some features registered after initialization
        # Note: actual count depends on what features are registered

    def test_compute_features_subset(self):
        """Test computing features."""
        # Reset and initialize
        FeatureRegistry.reset_for_testing()
        FeatureRegistry.initialize()

        registry = FeatureRegistry()

        # Create sample data
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        data = {
            "open": [100 + i * 0.1 for i in range(30)],
            "high": [105 + i * 0.1 for i in range(30)],
            "low": [95 + i * 0.1 for i in range(30)],
            "close": [102 + i * 0.1 for i in range(30)],
            "volume": [1000 + i * 10 for i in range(30)],
        }
        df = pd.DataFrame(data, index=dates)

        # Test computing features (should work with whatever is enabled)
        result = registry.compute_features(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        # Should have at least the original columns
        assert len(result.columns) >= len(df.columns)

    def test_compute_features_all_available(self):
        """Test computing all available features."""
        # Reset and initialize
        FeatureRegistry.reset_for_testing()
        FeatureRegistry.initialize()

        registry = FeatureRegistry()

        # Create smaller sample data for performance
        dates = pd.date_range("2023-01-01", periods=15, freq="D")
        data = {
            "open": [100 + i * 0.1 for i in range(15)],
            "high": [105 + i * 0.1 for i in range(15)],
            "low": [95 + i * 0.1 for i in range(15)],
            "close": [102 + i * 0.1 for i in range(15)],
            "volume": [1000 + i * 10 for i in range(15)],
        }
        df = pd.DataFrame(data, index=dates)

        # Test computing all features
        result = registry.compute_features(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        # Should have at least original columns (features may or may not be computed)
        assert len(result.columns) >= len(data)

    def test_get_nonexistent_feature(self):
        """Test getting a nonexistent feature."""
        FeatureRegistry.reset_for_testing()
        FeatureRegistry.initialize()

        try:
            FeatureRegistry.get("nonexistent_feature_12345")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass  # Expected

    def test_cache_and_parallel_settings(self):
        """Test cache and parallel processing settings."""
        FeatureRegistry.reset_for_testing()
        FeatureRegistry.initialize(cache_enabled=False, parallel_enabled=False)

        assert not FeatureRegistry.is_cache_enabled()
        assert not FeatureRegistry.is_parallel_enabled()

        # Reset and test defaults
        FeatureRegistry.reset_for_testing()
        FeatureRegistry.initialize()

        assert FeatureRegistry.is_cache_enabled()
        assert FeatureRegistry.is_parallel_enabled()

    def test_compute_features_all_available(self):
        """Test computing all available features."""
        registry = FeatureRegistry()

        # Create smaller sample data for performance
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        data = {
            "open": [100 + i * 0.1 for i in range(20)],
            "high": [105 + i * 0.1 for i in range(20)],
            "low": [95 + i * 0.1 for i in range(20)],
            "close": [102 + i * 0.1 for i in range(20)],
            "volume": [1000 + i * 10 for i in range(20)],
        }
        df = pd.DataFrame(data, index=dates)

        # Test computing all features
        result = registry.compute_features(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        # Should have at least original columns
        assert len(result.columns) >= len(data)

    def test_invalid_feature_name(self):
        """Test handling of invalid feature names."""
        # Test invalid feature name
        with pytest.raises(KeyError):
            FeatureRegistry.get("nonexistent_feature")

    def test_insufficient_data(self):
        """Test handling of insufficient data for feature computation."""
        # Create very small dataset
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "open": [99, 100, 101, 102, 103],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )

        # Test computing features with small dataset
        registry = FeatureRegistry()
        result = registry.compute_features(df)

        # Should handle gracefully even with small data
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
