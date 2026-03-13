"""
Unit tests for ztb.cache.sqlite_cache module.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ztb.cache.sqlite_cache import SQLiteCache, close_global_cache, get_cache, set_cache


class TestSQLiteCache:
    """Test cases for SQLiteCache class."""

    def test_init_default(self):
        """Test SQLiteCache initialization with defaults."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"
            cache = SQLiteCache(db_path=db_path)

            assert cache.db_path == db_path
            assert cache.max_items == 50000
            assert cache.task_mode == "default"
            assert cache.get_default_ttl() == 3600
            assert cache.conn is not None

            cache.close()

    def test_init_custom(self):
        """Test SQLiteCache initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"
            cache = SQLiteCache(db_path=db_path, max_items=1000, task_mode="training")

            assert cache.db_path == db_path
            assert cache.max_items == 1000
            assert cache.task_mode == "training"
            assert cache.get_default_ttl() == 1800  # training TTL

            cache.close()

    def test_get_default_ttl(self):
        """Test get_default_ttl for different task modes."""
        cache = SQLiteCache()

        assert cache.get_default_ttl() == 3600  # default

        cache.set_task_mode("re_evaluation")
        assert cache.get_default_ttl() == 600

        cache.set_task_mode("training")
        assert cache.get_default_ttl() == 1800

        cache.set_task_mode("testing")
        assert cache.get_default_ttl() == 300

        cache.set_task_mode("unknown")
        assert cache.get_default_ttl() == 300  # unchanged from testing

        cache.close()

    @patch("ztb.cache.sqlite_cache.logging")
    def test_set_task_mode(self, mock_logging):
        """Test set_task_mode."""
        cache = SQLiteCache()

        cache.set_task_mode("training")
        assert cache.task_mode == "training"
        mock_logging.info.assert_called_once()

        # Reset mock
        mock_logging.reset_mock()

        cache.set_task_mode("unknown")
        assert cache.task_mode == "training"  # unchanged
        mock_logging.warning.assert_called_once()

        cache.close()

    def test_generate_key(self):
        """Test _generate_key method."""
        cache = SQLiteCache()
        key1 = cache._generate_key("arg1", "arg2", 123)
        key2 = cache._generate_key("arg1", "arg2", 123)
        key3 = cache._generate_key("different", "args")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 64  # SHA256 hex length

        cache.close()

    def test_set_get(self):
        """Test basic set and get operations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"
            cache = SQLiteCache(db_path=db_path)

            # Test set and get
            test_data = {"key": "value", "number": 42}
            cache.set("test_key", test_data)

            result = cache.get("test_key")
            assert result == test_data

            # Test get non-existent key
            result = cache.get("non_existent")
            assert result is None

            cache.close()

    def test_set_with_ttl(self):
        """Test set with custom TTL."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"
            cache = SQLiteCache(db_path=db_path, enable_memory_cache=False)
            try:
                with patch("ztb.cache.sqlite_cache.time.time", return_value=1000.0):
                    cache.set("ttl_key", "ttl_value", ttl_sec=1)
                    result = cache.get("ttl_key")
                assert result == "ttl_value"

                with patch("ztb.cache.sqlite_cache.time.time", return_value=1001.0):
                    result = cache.get("ttl_key")
                assert result is None
            finally:
                cache.close()

    def test_touch(self):
        """Test touch method."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"
            cache = SQLiteCache(db_path=db_path)

            cache.set("touch_key", "touch_value")

            # Touch should not fail
            cache.touch("touch_key")

            # Value should still be retrievable
            result = cache.get("touch_key")
            assert result == "touch_value"

            cache.close()

    def test_purge(self):
        """Test manual purge."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"
            cache = SQLiteCache(db_path=db_path, max_items=2)

            # Add items
            cache.set("key1", "value1")
            cache.set("key2", "value2")
            cache.set("key3", "value3")  # Should trigger purge

            # Should have only 2 items
            result1 = cache.get("key1")
            result2 = cache.get("key2")
            result3 = cache.get("key3")

            # At least key3 should exist, others may be purged
            assert result3 == "value3"

            cache.close()

    def test_close(self):
        """Test close method."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"
            cache = SQLiteCache(db_path=db_path)

            # Close should not raise exception
            cache.close()

            # Connection should be closed
            with pytest.raises(sqlite3.ProgrammingError):
                cache.conn.execute("SELECT 1")

    @patch("ztb.cache.sqlite_cache._global_cache")
    def test_global_functions(self, mock_global_cache):
        """Test global cache functions."""
        # Test set_cache
        set_cache("global_key", "global_value")
        mock_global_cache.set.assert_called_once_with(
            "global_key", "global_value", None
        )

        # Test get_cache
        mock_global_cache.get.return_value = "retrieved_value"
        result = get_cache("global_key")
        assert result == "retrieved_value"
        mock_global_cache.get.assert_called_once_with("global_key")

    @patch("ztb.cache.sqlite_cache._global_cache")
    def test_close_global_cache(self, mock_global_cache):
        """Test close_global_cache function."""
        close_global_cache()
        mock_global_cache.close.assert_called_once()
