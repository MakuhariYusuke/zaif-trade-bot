#!/usr/bin/env python3
"""
sqlite_cache.py
Lightweight SQLite-based cache for Raspberry Pi environments.
"""

import sqlite3
import time
import hashlib
from typing import Optional, Any
from pathlib import Path
import pickle
import logging


class SQLiteCache:
    """SQLite-based LRU cache with TTL support and task-specific optimization"""
    
    def __init__(self, db_path: Optional[Path] = None, max_items: int = 50000, task_mode: str = "default"):
        super().__init__()
        if db_path is None:
            db_path = Path("cache/cache.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.max_items = max_items
        self.task_mode = task_mode

        # Task-specific TTL defaults
        self.default_ttls = {
            "default": 3600,      # 1 hour
            "re_evaluation": 600, # 10 minutes for re-evaluation tasks
            "training": 1800,     # 30 minutes for ML training
            "testing": 300        # 5 minutes for testing
        }

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
    
    def get_default_ttl(self) -> int:
        """Get default TTL based on task mode"""
        return self.default_ttls.get(self.task_mode, self.default_ttls["default"])
    
    def set_task_mode(self, mode: str) -> None:
        """Change task mode for different TTL behavior"""
        if mode in self.default_ttls:
            self.task_mode = mode
            logging.info(f"Cache task mode set to '{mode}' (TTL: {self.get_default_ttl()}s)")
        else:
            logging.warning(f"Unknown task mode '{mode}', keeping '{self.task_mode}'")
    
    def _init_db(self) -> None:
        """Initialize database schema"""
        conn = self.conn
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                created_at INTEGER,
                last_access INTEGER,
                ttl_sec INTEGER
            )
        """)
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.commit()
    
    def _generate_key(self, *args: Any) -> str:
        """Generate SHA256 key from arguments"""
        key_str = str(args)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def set(self, key: str, value: Any, ttl_sec: Optional[int] = None) -> None:
        """Set cache value with optional TTL"""
        if ttl_sec is None:
            ttl_sec = self.get_default_ttl()
        
        value_blob = pickle.dumps(value)
        now = int(time.time())
        
        conn = self.conn
        conn.execute("""
            INSERT OR REPLACE INTO cache (key, value, created_at, last_access, ttl_sec)
            VALUES (?, ?, ?, ?, ?)
        """, (key, value_blob, now, now, ttl_sec))
        conn.commit()
    def get(self, key: str) -> Optional[Any]:
        """Get cache value, returns None if expired or not found"""
        now = int(time.time())
        
        conn = self.conn
        cursor = conn.execute("""
            SELECT value, ttl_sec, created_at FROM cache WHERE key = ?
        """, (key,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        value_blob, ttl_sec, created_at = row
        
        # Check TTL
        if ttl_sec and (now - created_at) > ttl_sec:
            # Expired, remove
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return None
        
        # Update last_access for LRU
        conn.execute("""
            UPDATE cache SET last_access = ? WHERE key = ?
        """, (now, key))
        conn.commit()
        
        return pickle.loads(value_blob)
    def touch(self, key: str) -> None:
        """Update last_access for LRU"""
        now = int(time.time())
        conn = self.conn
        conn.execute("""
            UPDATE cache SET last_access = ? WHERE key = ?
        """, (now, key))
        conn.commit()
    
    def _purge_if_needed(self, conn: sqlite3.Connection) -> None:
        """Purge oldest entries if over max_items (LRU)"""
        cursor = conn.execute("SELECT COUNT(*) FROM cache")
        count = cursor.fetchone()[0]
        
        if count > self.max_items:
            # Remove oldest accessed items
            excess = count - self.max_items
            conn.execute("""
                DELETE FROM cache WHERE key IN (
                    SELECT key FROM cache ORDER BY last_access ASC LIMIT ?
                )
            """, (excess,))
            conn.commit()
    def purge(self, max_items: Optional[int] = None) -> None:
        """Manual purge"""
        if max_items is None:
            max_items = self.max_items
        
        conn = self.conn
        cursor = conn.execute("SELECT COUNT(*) FROM cache")
        count = cursor.fetchone()[0]
        
        if count > max_items:
            excess = count - max_items
            conn.execute("""
                DELETE FROM cache WHERE key IN (
                    SELECT key FROM cache ORDER BY last_access ASC LIMIT ?
                )
            """, (excess,))
            conn.commit()
    
    def close(self) -> None:
        """Close the database connection"""
        if self.conn:
            self.conn.close()
# Global cache instance
_global_cache = SQLiteCache()

def close_global_cache() -> None:
    """Close the global cache connection"""
    _global_cache.close()
_global_cache = SQLiteCache()

# Convenience functions
def set_cache(key: str, value: Any, ttl_sec: Optional[int] = None) -> None:
    """Global cache setter"""
    _global_cache.set(key, value, ttl_sec)


def get_cache(key: str) -> Optional[Any]:
    """Global cache getter"""
    return _global_cache.get(key)
if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    # Simple test
    cache = SQLiteCache()
    
    # Test set/get
    cache.set("test_key", {"data": "value"}, ttl_sec=3600)
    result = cache.get("test_key")
    logging.info(f"Cache test: {result}")
    
    # Test expiration (set TTL=1 second)
    cache.set("expire_key", "expires soon", ttl_sec=1)
    time.sleep(2)
    result = cache.get("expire_key")
    logging.info(f"Expired key: {result}")  # Should be None
    print(f"Expired key: {result}")  # Should be None

    # Close the cache connection
    cache.close()