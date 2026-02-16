"""
Cache management utilities for Zaif Trade Bot.

This module provides various caching strategies including LRU, TTL, and memory-aware caching.
"""

import threading
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class TTLCache:
    """
    Time-To-Live cache with automatic expiration.
    """

    def __init__(self, ttl_seconds: float = 300.0):  # 5 minutes default
        self.ttl = ttl_seconds
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    # Expired, remove it
                    del self.cache[key]
            return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with current timestamp."""
        with self.lock:
            self.cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cached values."""
        with self.lock:
            self.cache.clear()

    def cleanup(self) -> int:
        """Remove expired entries. Returns number of removed entries."""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, (_, timestamp) in self.cache.items()
                if current_time - timestamp >= self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]
            return len(expired_keys)


class MemoryAwareCache:
    """
    Cache that respects memory limits and automatically evicts old entries.
    """

    def __init__(self, max_memory_mb: float = 100.0):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[
            str, tuple[Any, float, int]
        ] = {}  # key -> (value, timestamp, size_bytes)
        self.current_memory = 0
        self.lock = threading.RLock()

    def _get_size(self, obj: Any) -> int:
        """Estimate memory size of an object."""
        # Simple estimation - can be improved with more sophisticated methods
        if isinstance(obj, (int, float)):
            return 28  # Approximate size of Python numeric objects
        elif isinstance(obj, str):
            return 49 + len(obj)  # Base size + string length
        elif isinstance(obj, (list, tuple)):
            return 64 + sum(self._get_size(item) for item in obj)
        elif isinstance(obj, dict):
            return 240 + sum(
                self._get_size(k) + self._get_size(v) for k, v in obj.items()
            )
        else:
            return 1000  # Default estimate for complex objects

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                value, timestamp, size = self.cache[key]
                # Update access time
                self.cache[key] = (value, time.time(), size)
                return value
            return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with memory management."""
        with self.lock:
            size = self._get_size(value)

            # Check if we need to evict entries
            while self.current_memory + size > self.max_memory_bytes and self.cache:
                # Remove oldest accessed entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                _, _, removed_size = self.cache.pop(oldest_key)
                self.current_memory -= removed_size

            # Add new entry
            self.cache[key] = (value, time.time(), size)
            self.current_memory += size

    def clear(self) -> None:
        """Clear all cached values."""
        with self.lock:
            self.cache.clear()
            self.current_memory = 0


def cached_with_ttl(ttl_seconds: float) -> Callable[[F], F]:
    """
    Decorator that caches function results with TTL.

    Args:
        ttl_seconds: Time-to-live in seconds
    """
    cache = TTLCache(ttl_seconds)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create cache key from function arguments
            key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def memory_cached(max_memory_mb: float = 50.0) -> Callable[[F], F]:
    """
    Decorator that caches function results with memory limits.

    Args:
        max_memory_mb: Maximum memory to use for cache in MB
    """
    cache = MemoryAwareCache(max_memory_mb)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            result = cache.get(key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# Global cache instances
model_cache = TTLCache(ttl_seconds=3600)  # 1 hour for models
data_cache = MemoryAwareCache(max_memory_mb=200.0)  # 200MB for data
computation_cache = TTLCache(ttl_seconds=1800)  # 30 minutes for computations
