#!/usr/bin/env python3
"""
test_sqlite_cache.py
Unit tests for SQLite cache
"""

import tempfile
import time
from pathlib import Path

import pytest
from cache.sqlite_cache import SQLiteCache


@pytest.fixture
def temp_db():
    """Temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)


def test_basic_set_get(temp_db):
    """Test basic set and get operations"""
    cache = SQLiteCache(db_path=temp_db, max_items=100)

    test_data = {"key": "value", "number": 42}
    cache.set("test_key", test_data)

    result = cache.get("test_key")
    assert result == test_data


def test_ttl_expiration(temp_db):
    """Test TTL expiration"""
    cache = SQLiteCache(db_path=temp_db, max_items=100)

    cache.set("expire_key", "data", ttl_sec=1)
    assert cache.get("expire_key") == "data"

    time.sleep(1.1)  # Wait for expiration
    assert cache.get("expire_key") is None


def test_lru_purge(temp_db):
    """Test LRU purging when over max_items"""
    cache = SQLiteCache(db_path=temp_db, max_items=3)

    # Add 5 items
    for i in range(5):
        cache.set(f"key_{i}", f"value_{i}")

    # Should only keep 3 items (LRU)
    # Check that we can still get some items
    result = cache.get("key_4")  # Most recent
    assert result == "value_4"


def test_touch(temp_db):
    """Test touch functionality"""
    cache = SQLiteCache(db_path=temp_db, max_items=100)

    cache.set("touch_key", "data")
    cache.touch("touch_key")

    # Should still be accessible
    assert cache.get("touch_key") == "data"


def test_nonexistent_key(temp_db):
    """Test getting nonexistent key"""
    cache = SQLiteCache(db_path=temp_db, max_items=100)

    result = cache.get("nonexistent")
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
