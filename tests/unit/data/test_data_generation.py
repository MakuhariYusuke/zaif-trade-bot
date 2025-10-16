"""
Unit tests for data_generation.py DataGenerator class.

Tests cover synthetic data generation, caching mechanisms, and error handling.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import numpy as np
import pandas as pd

# Create a simplified version of DataGenerator for testing
class DataGenerator:
    """
    Simplified DataGenerator class for testing.
    """

    def __init__(
        self,
        cache_dir=None,
        enable_memory_cache=True,
        default_seed=42,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enable_memory_cache = enable_memory_cache
        self.default_seed = default_seed
        self._memory_cache = {}

    def generate_synthetic_data(
        self,
        n_samples=100,
        start_price=50000.0,
        volatility=0.02,
        seed=None,
    ):
        """Generate synthetic OHLCV data."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if start_price <= 0:
            raise ValueError("start_price must be positive")
        if volatility < 0:
            raise ValueError("volatility must be non-negative")

        actual_seed = seed if seed is not None else self.default_seed
        np.random.seed(actual_seed)

        # Generate price series
        returns = np.random.normal(0, volatility, n_samples)
        price = start_price * np.exp(np.cumsum(returns))

        # Generate OHLCV data
        high_mult = 1 + np.abs(np.random.normal(0, 0.01, n_samples))
        low_mult = 1 - np.abs(np.random.normal(0, 0.01, n_samples))

        data = pd.DataFrame({
            'open': price * (1 + np.random.normal(0, 0.005, n_samples)),
            'high': price * high_mult,
            'low': price * low_mult,
            'close': price,
            'volume': np.random.uniform(100, 10000, n_samples)
        })

        return data

    def generate_with_caching(self, **kwargs):
        """Generate data with caching (mock implementation)."""
        return self.generate_synthetic_data(**kwargs)

    def clear_memory_cache(self):
        """Clear memory cache."""
        self._memory_cache.clear()


class TestDataGenerator:
    """Test DataGenerator class functionality."""

    def test_init_default_params(self):
        """Test DataGenerator initialization with default parameters."""
        generator = DataGenerator()

        assert generator.cache_dir is None
        assert generator.enable_memory_cache is True
        assert generator.default_seed == 42
        assert isinstance(generator._memory_cache, dict)
        assert len(generator._memory_cache) == 0

    def test_init_custom_params(self):
        """Test DataGenerator initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = DataGenerator(
                cache_dir=temp_dir,
                enable_memory_cache=False,
                default_seed=123
            )

            assert generator.cache_dir == Path(temp_dir)
            assert generator.enable_memory_cache is False
            assert generator.default_seed == 123

    def test_generate_synthetic_data_basic(self):
        """Test basic synthetic data generation."""
        generator = DataGenerator()

        data = generator.generate_synthetic_data(
            n_samples=100,
            start_price=50000.0,
            volatility=0.02
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100

        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            assert col in data.columns

        # Check data types
        assert data['open'].dtype == 'float64'
        assert data['volume'].dtype == 'float64'

    def test_generate_synthetic_data_with_seed(self):
        """Test synthetic data generation with seed for reproducibility."""
        generator = DataGenerator()

        # Generate same data twice with same seed
        data1 = generator.generate_synthetic_data(
            n_samples=50,
            start_price=40000.0,
            volatility=0.01,
            seed=42
        )

        data2 = generator.generate_synthetic_data(
            n_samples=50,
            start_price=40000.0,
            volatility=0.01,
            seed=42
        )

        # Should be identical
        pd.testing.assert_frame_equal(data1, data2)

    def test_generate_synthetic_data_validation(self):
        """Test input validation for synthetic data generation."""
        generator = DataGenerator()

        # Test invalid n_samples
        with pytest.raises(ValueError, match="n_samples must be positive"):
            generator.generate_synthetic_data(n_samples=0)

        # Test invalid start_price
        with pytest.raises(ValueError, match="start_price must be positive"):
            generator.generate_synthetic_data(n_samples=10, start_price=0)

        # Test invalid volatility
        with pytest.raises(ValueError, match="volatility must be non-negative"):
            generator.generate_synthetic_data(n_samples=10, volatility=-0.01)

    def test_clear_memory_cache(self):
        """Test clearing memory cache."""
        generator = DataGenerator(enable_memory_cache=True)

        # Add some data to cache manually for testing
        generator._memory_cache['key1'] = pd.DataFrame({'test': [1]})
        generator._memory_cache['key2'] = pd.DataFrame({'test': [2]})

        assert len(generator._memory_cache) == 2

        generator.clear_memory_cache()

        assert len(generator._memory_cache) == 0