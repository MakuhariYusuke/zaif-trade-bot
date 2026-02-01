#!/usr/bin/env python3
"""
Unit tests for legacy data loading helpers.

Note: ImprovedDataLoader is kept for backward compatibility; advanced CSV I/O
is centralized in ztb.io.advanced_csv.
"""

import gzip
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ztb.utils.data.data_generation import DataGenerator
from ztb.utils.data.improved_data_loader import ImprovedDataLoader


class TestDataLoadingImprovements:
    """Test data loading optimization features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = DataGenerator(enable_memory_cache=True)
        self.improved_loader = ImprovedDataLoader()

    def test_load_data_with_memory_map(self):
        """Test memory-mapped file loading."""
        # Create temporary numpy file
        test_data = np.random.rand(100, 10)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            np.save(tmp.name, test_data)

            try:
                # Load with memory map (auto-detect shape and dtype)
                mmap_data = self.generator.load_data_with_memory_map(tmp.name)

                # Verify data integrity
                assert mmap_data.shape == test_data.shape
                np.testing.assert_array_equal(mmap_data, test_data)

                # Verify it's memory mapped
                assert hasattr(mmap_data, "filename")
                assert Path(mmap_data.filename) == Path(tmp.name)

            finally:
                try:
                    Path(tmp.name).unlink()
                except PermissionError:
                    pass  # File may be locked by memory map


    def test_load_compressed_data_gzip(self):
        """Test loading gzip compressed data."""
        test_df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})

        with tempfile.NamedTemporaryFile(suffix=".pkl.gz", delete=False) as tmp:
            try:
                # Save compressed data
                with gzip.open(tmp.name, "wb") as f:
                    pickle.dump(test_df, f)

                # Load compressed data
                loaded_df = self.generator.load_compressed_data(
                    tmp.name, compression="gzip"
                )

                # Verify data integrity
                pd.testing.assert_frame_equal(loaded_df, test_df)

            finally:
                try:
                    Path(tmp.name).unlink()
                except PermissionError:
                    pass

    @pytest.mark.asyncio
    async def test_load_data_async(self):
        """Test asynchronous data loading."""
        # Create temporary CSV files
        test_dfs = []
        file_paths = []

        for i in range(3):
            df = pd.DataFrame({"col1": [i, i + 1, i + 2], "col2": ["a", "b", "c"]})
            test_dfs.append(df)

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                df.to_csv(tmp.name, index=False)
                file_paths.append(tmp.name)

        try:
            # Load asynchronously
            loaded_dfs = await self.generator.load_data_async(file_paths, max_workers=2)

            # Verify all files loaded
            assert len(loaded_dfs) == 3

            # Verify data integrity (order may vary due to async)
            loaded_data = {tuple(df["col1"].tolist()) for df in loaded_dfs}
            expected_data = {tuple(df["col1"].tolist()) for df in test_dfs}
            assert loaded_data == expected_data

        finally:
            for path in file_paths:
                Path(path).unlink()

    def test_prefetch_data(self):
        """Test data prefetching."""
        # Create temporary CSV file
        test_df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            test_df.to_csv(tmp.name, index=False)

            try:
                # Prefetch data
                executor = self.generator.prefetch_data([tmp.name], prefetch_size=1)

                # Verify executor returned
                assert executor is not None
                executor.shutdown(wait=True)

                # Verify data was cached (if memory cache enabled)
                if self.generator.enable_memory_cache:
                    # Cache key is based on md5 hash of file path
                    import hashlib

                    cache_key = f"prefetch_{hashlib.md5(tmp.name.encode()).hexdigest()}"
                    assert cache_key in self.generator._memory_cache

            finally:
                try:
                    Path(tmp.name).unlink()
                except PermissionError:
                    pass


class TestImprovedDataLoader:
    """Test legacy ImprovedDataLoader features (compat coverage)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ImprovedDataLoader()
        self.generator = DataGenerator(enable_memory_cache=True)

    def teardown_method(self):
        """Clean up after tests."""
        self.loader.cleanup()

    def test_load_csv_memory_mapped(self):
        """Test memory-mapped CSV loading."""
        # Create test CSV
        test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="1min"),
                "open": np.random.uniform(30000, 40000, 100),
                "high": np.random.uniform(30000, 40000, 100),
                "low": np.random.uniform(30000, 40000, 100),
                "close": np.random.uniform(30000, 40000, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
            test_data.to_csv(tmp.name, index=False)

        try:
            # Load data
            loaded_df = self.loader.load_csv_memory_mapped(
                tmp.name, parse_dates=["timestamp"]
            )

            # Verify data
            assert len(loaded_df) == 100
            assert list(loaded_df.columns) == list(test_data.columns)
            pd.testing.assert_frame_equal(loaded_df, test_data, check_dtype=False)

        finally:
            Path(tmp.name).unlink()

    def test_compute_features_parallel(self):
        """Test parallel feature computation."""
        # Create test data
        data = pd.DataFrame(
            {
                "close": np.random.uniform(30000, 40000, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            }
        )

        # Define feature functions
        def sma_5(series):
            return series.rolling(5).mean()

        def volume_ratio(series):
            return series / series.shift(1)

        feature_functions = {
            "sma_5": lambda df: sma_5(df["close"]),
            "volume_ratio": lambda df: volume_ratio(df["volume"]),
        }

        # Compute features
        result_df = self.loader.compute_features_parallel(data, feature_functions)

        # Verify results
        assert "sma_5" in result_df.columns
        assert "volume_ratio" in result_df.columns
        assert len(result_df) == len(data)

        # Check SMA calculation
        expected_sma = data["close"].rolling(5).mean()
        pd.testing.assert_series_equal(
            result_df["sma_5"], expected_sma, check_names=False
        )

    def test_incremental_feature_computation(self):
        """Test incremental feature computation with caching."""
        # Create test data
        data = pd.DataFrame(
            {
                "close": np.random.uniform(30000, 40000, 50),
                "volume": np.random.uniform(1000, 10000, 50),
            }
        )

        def simple_ma(series):
            return series.rolling(3).mean()

        feature_functions = {"ma_3": lambda df: simple_ma(df["close"])}

        cache_key = "test_incremental"

        # First computation
        result1 = self.loader.incremental_feature_computation(
            data, feature_functions, cache_key
        )

        # Second computation (should use cache)
        result2 = self.loader.incremental_feature_computation(
            data, feature_functions, cache_key
        )

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

        # Check cache file exists
        cache_file = self.loader.cache_dir / f"{cache_key}_features.pkl"
        assert cache_file.exists()

    @pytest.mark.asyncio
    async def test_load_csv_async(self):
        """Test asynchronous CSV loading."""
        # Create test CSV
        test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1min"),
                "close": np.random.uniform(30000, 40000, 10),
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
            test_data.to_csv(tmp.name, index=False)

        try:
            # Load asynchronously
            loaded_df = await self.loader.load_csv_async(
                tmp.name, parse_dates=["timestamp"]
            )

            # Verify data
            assert len(loaded_df) == 10
            pd.testing.assert_frame_equal(loaded_df, test_data, check_dtype=False)

        finally:
            Path(tmp.name).unlink()

    def test_prefetch_data(self):
        """Test data prefetching."""
        pytest.skip("Async prefetch test requires more complex setup")

    def test_load_compressed_data_unsupported_compression(self):
        """Test error handling for unsupported compression."""
        # Create a dummy file first
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            test_df = pd.DataFrame({"A": [1]})
            with open(tmp.name, "wb") as f:
                pickle.dump(test_df, f)

            try:
                with pytest.raises(ValueError, match="Unsupported compression"):
                    self.generator.load_compressed_data(tmp.name, compression="invalid")
            finally:
                try:
                    Path(tmp.name).unlink()
                except PermissionError:
                    pass

    def test_load_data_with_memory_map_file_not_found(self):
        """Test error handling for missing memory map file."""
        with pytest.raises(FileNotFoundError):
            self.generator.load_data_with_memory_map("nonexistent.npy")

    def test_load_compressed_data_file_not_found(self):
        """Test error handling for missing compressed file."""
        with pytest.raises(FileNotFoundError):
            self.generator.load_compressed_data("nonexistent.pkl.gz")
