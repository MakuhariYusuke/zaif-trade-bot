"""
Unit tests for ztb.cache.data_loader module.
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ztb.cache.data_loader import CacheDataLoader


class TestDataLoader:
    """Test cases for CacheDataLoader class."""

    def test_init(self):
        """Test DataLoader initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            loader = CacheDataLoader(cache_dir=tmp_dir)
            assert loader.cache_dir == Path(tmp_dir)
            assert loader.cache_dir.exists()

    def test_init_default_cache_dir(self):
        """Test DataLoader with default cache directory."""
        loader = CacheDataLoader()
        assert loader.cache_dir == Path("data/cache")
        # Don't check exists() as it may not exist in test environment

    def test_load_with_cache_hit(self, tmp_path):
        """Test load_with_cache when cache exists."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create mock cached data
        test_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        cache_file = cache_dir / "test_key.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(test_df, f)

        loader = CacheDataLoader(cache_dir=str(cache_dir))

        def mock_load_func():
            return pd.DataFrame({"new": [5, 6]})

        result = loader.load_with_cache("test_key", mock_load_func)

        # Should return cached data
        pd.testing.assert_frame_equal(result, test_df)

    def test_load_with_cache_miss(self, tmp_path):
        """Test load_with_cache when cache doesn't exist."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        loader = CacheDataLoader(cache_dir=str(cache_dir))

        test_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        def mock_load_func():
            return test_df

        result = loader.load_with_cache("test_key", mock_load_func)

        # Should return new data
        pd.testing.assert_frame_equal(result, test_df)

        # Should have created cache file
        cache_file = cache_dir / "test_key.pkl"
        assert cache_file.exists()

    def test_load_with_cache_corrupted(self, tmp_path):
        """Test load_with_cache with corrupted cache file."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create corrupted cache file
        cache_file = cache_dir / "test_key.pkl"
        with open(cache_file, "wb") as f:
            f.write(b"corrupted data")

        loader = CacheDataLoader(cache_dir=str(cache_dir))

        test_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        def mock_load_func():
            return test_df

        result = loader.load_with_cache("test_key", mock_load_func)

        # Should return new data despite corrupted cache
        pd.testing.assert_frame_equal(result, test_df)

    @patch("ztb.cache.data_loader.safe_operation")
    def test_load_with_cache_safe_operation_failure(self, mock_safe_operation):
        """Test load_with_cache when safe_operation fails."""
        mock_safe_operation.return_value = pd.DataFrame()

        loader = CacheDataLoader()

        def mock_load_func():
            return pd.DataFrame({"col1": [1, 2]})

        result = loader.load_with_cache("test_key", mock_load_func)

        # Should return empty DataFrame on failure
        assert result.empty
        mock_safe_operation.assert_called_once()

    def test_load_multiple(self, tmp_path):
        """Test load_multiple method."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        loader = CacheDataLoader(cache_dir=str(cache_dir))

        loaders = {
            "key1": lambda: pd.DataFrame({"a": [1]}),
            "key2": lambda: pd.DataFrame({"b": [2]}),
        }

        result = loader.load_multiple(loaders)

        assert len(result) == 2
        pd.testing.assert_frame_equal(result["key1"], pd.DataFrame({"a": [1]}))
        pd.testing.assert_frame_equal(result["key2"], pd.DataFrame({"b": [2]}))

    @patch("ztb.cache.data_loader.safe_operation")
    def test_load_multiple_safe_operation_failure(self, mock_safe_operation):
        """Test load_multiple when safe_operation fails."""
        mock_safe_operation.return_value = {}

        loader = CacheDataLoader()

        loaders = {"key1": lambda: pd.DataFrame({"a": [1]})}

        result = loader.load_multiple(loaders)

        # Should return empty dict on failure
        assert result == {}
        mock_safe_operation.assert_called_once()

    def test_clear_cache_specific(self, tmp_path):
        """Test clear_cache for specific key."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create cache files
        cache_file1 = cache_dir / "key1.pkl"
        cache_file2 = cache_dir / "key2.pkl"
        cache_file1.write_text("data1")
        cache_file2.write_text("data2")

        loader = CacheDataLoader(cache_dir=str(cache_dir))

        loader.clear_cache("key1")

        assert not cache_file1.exists()
        assert cache_file2.exists()

    def test_clear_cache_all(self, tmp_path):
        """Test clear_cache for all keys."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create cache files
        cache_file1 = cache_dir / "key1.pkl"
        cache_file2 = cache_dir / "key2.pkl"
        cache_file1.write_text("data1")
        cache_file2.write_text("data2")

        loader = CacheDataLoader(cache_dir=str(cache_dir))

        loader.clear_cache()

        assert not cache_file1.exists()
        assert not cache_file2.exists()

    def test_list_cached(self, tmp_path):
        """Test list_cached method."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create cache files
        (cache_dir / "key1.pkl").write_text("data1")
        (cache_dir / "key2.pkl").write_text("data2")
        (cache_dir / "not_cache.txt").write_text("not cache")  # Non-pkl file

        loader = CacheDataLoader(cache_dir=str(cache_dir))

        cached_keys = loader.list_cached()

        assert set(cached_keys) == {"key1", "key2"}
