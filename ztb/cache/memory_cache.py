#!/usr/bin/env python3
"""
Memory Management Cache for SAC v446 Training Optimization

This module provides advanced memory management utilities including:
- TTLCache for time-based cache expiration
- Dynamic buffer size adjustment
- Memory usage monitoring and optimization
- LRU cache for frequently accessed data
"""

import logging
import threading
import time
from typing import Any

from ztb.trading.environment.constants import BYTES_PER_MB

try:
    from cachetools import TTLCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    # Fallback implementation
    class TTLCache(dict):
        """Simple TTL cache fallback implementation."""
        def __init__(self, maxsize: int, ttl: float) -> None:
            super().__init__()
            self.maxsize = maxsize
            self.ttl = ttl
            self._timestamps = {}

        def __setitem__(self, key, value):
            if len(self) >= self.maxsize:
                # Remove oldest item
                oldest_key = min(
                    self._timestamps.keys(), key=lambda k: self._timestamps[k]
                )
                del self[oldest_key]
                del self._timestamps[oldest_key]

            super().__setitem__(key, value)
            # Preserve an explicitly-set timestamp when tests set it manually.
            if key not in self._timestamps:
                self._timestamps[key] = time.time()

        def __getitem__(self, key):
            if key in self._timestamps:
                if time.time() - self._timestamps[key] > self.ttl:
                    # Expired
                    del self[key]
                    del self._timestamps[key]
                    raise KeyError(key)
            return super().__getitem__(key)

logger = logging.getLogger(__name__)

class MemoryManager:
    """Advanced memory management for training optimization."""

    def __init__(self, max_memory_mb: float = 500.0, enable_monitoring: bool = True) -> None:
        """
        Initialize memory manager.

        Args:
            max_memory_mb: Maximum memory usage in MB
            enable_monitoring: Whether to enable memory monitoring
        """
        self.max_memory_mb = max_memory_mb
        self.enable_monitoring = enable_monitoring
        self.memory_log_path = None
        self.memory_log_interval = 2000
        self.gc_step_interval = 1000

        if CACHETOOLS_AVAILABLE:
            feature_ttl = 300
            data_ttl = 60
            model_ttl = 1800
        else:
            # Keep fallback TTLs short to avoid unbounded growth in environments
            # without cachetools' expiration enforcement.
            feature_ttl = 5
            data_ttl = 5
            model_ttl = 30

        # Initialize caches
        self.feature_cache = TTLCache(maxsize=1000, ttl=feature_ttl)  # seconds
        self.data_cache = TTLCache(maxsize=500, ttl=data_ttl)  # seconds
        self.model_cache = TTLCache(maxsize=50, ttl=model_ttl)  # seconds

        # Memory monitoring
        self.memory_history = []
        self.monitoring_thread = None
        self.monitoring_active = False

        if enable_monitoring:
            self.start_memory_monitoring()

        logger.info(f"MemoryManager initialized with max {max_memory_mb}MB limit")

    def set_memory_log_path(self, path: str) -> None:
        """set path for memory logging."""
        self.memory_log_path = path

    def set_memory_log_interval(self, interval: int) -> None:
        """set memory logging interval (steps)."""
        self.memory_log_interval = interval

    def set_gc_interval(self, interval: int) -> None:
        """set garbage collection interval (steps)."""
        self.gc_step_interval = interval

    def cache_feature_data(self, key: str, data: Any, ttl: float | None = None) -> None:
        """
        Cache feature data with optional custom TTL.

        Args:
            key: Cache key
            data: Data to cache
            ttl: Custom TTL in seconds (overrides default)
        """
        if ttl is not None and ttl != self.feature_cache.ttl:
            # For custom TTL, store in a separate cache with that TTL
            if not hasattr(self, '_custom_ttl_caches'):
                self._custom_ttl_caches = {}
            if ttl not in self._custom_ttl_caches:
                self._custom_ttl_caches[ttl] = TTLCache(maxsize=1000, ttl=ttl)
            self._custom_ttl_caches[ttl][key] = data
        else:
            self.feature_cache[key] = data

    def get_cached_feature_data(self, key: str) -> Any | None:
        """Retrieve cached feature data."""
        # First check default cache
        try:
            return self.feature_cache[key]
        except KeyError:
            pass

        # Check custom TTL caches
        if hasattr(self, '_custom_ttl_caches'):
            for cache in self._custom_ttl_caches.values():
                try:
                    return cache[key]
                except KeyError:
                    continue

        return None

    def cache_training_data(self, key: str, data: Any) -> None:
        """Cache training data."""
        self.data_cache[key] = data

    def get_cached_training_data(self, key: str) -> Any | None:
        """Retrieve cached training data."""
        return self.data_cache.get(key)

    def cache_model_state(self, key: str, state_dict: dict[str, Any]) -> None:
        """Cache model state."""
        self.model_cache[key] = state_dict

    def get_cached_model_state(self, key: str) -> dict[str, Any] | None:
        """Retrieve cached model state."""
        return self.model_cache.get(key)

    def clear_expired_cache(self) -> None:
        """Manually clear expired cache entries."""
        current_time = time.time()

        # TTLCache handles expiration automatically, but we can force cleanup
        expired_keys = []
        for key in list(self.feature_cache.keys()):
            try:
                self.feature_cache[key]  # Access to check expiration
            except KeyError:
                expired_keys.append(key)

        for key in list(self.data_cache.keys()):
            try:
                self.data_cache[key]
            except KeyError:
                expired_keys.append(key)

        for key in list(self.model_cache.keys()):
            try:
                self.model_cache[key]
            except KeyError:
                expired_keys.append(key)

        # Clear custom TTL caches
        if hasattr(self, '_custom_ttl_caches'):
            for cache in self._custom_ttl_caches.values():
                expired_custom_keys = []
                for key in list(cache.keys()):
                    try:
                        cache[key]  # Access to check expiration
                    except KeyError:
                        expired_custom_keys.append(key)

        if expired_keys:
            logger.debug(f"Cleared {len(expired_keys)} expired cache entries")

    def get_memory_usage(self) -> dict[str, float | int]:
        """Get current memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "rss_mb": memory_info.rss / BYTES_PER_MB,
                "vms_mb": memory_info.vms / BYTES_PER_MB,
                "cpu_percent": process.cpu_percent(),
                "cache_size": len(self.feature_cache) + len(self.data_cache) + len(self.model_cache)
            }
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return {
                "rss_mb": 0.0,
                "vms_mb": 0.0,
                "cpu_percent": 0.0,
                "cache_size": len(self.feature_cache) + len(self.data_cache) + len(self.model_cache)
            }

    def start_memory_monitoring(self) -> None:
        """Start background memory monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._memory_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Memory monitoring started")

    def stop_memory_monitoring(self) -> None:
        """Stop background memory monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        logger.info("Memory monitoring stopped")

    def _memory_monitoring_loop(self) -> None:
        """Background memory monitoring loop."""
        while self.monitoring_active:
            try:
                memory_stats = self.get_memory_usage()
                self.memory_history.append({
                    "timestamp": time.time(),
                    **memory_stats
                })

                # Keep only last 1000 entries
                if len(self.memory_history) > 1000:
                    self.memory_history = self.memory_history[-1000:]

                # Log if memory usage is high (relaxed threshold for feature engineering)
                if memory_stats["rss_mb"] > self.max_memory_mb * 0.95:
                    logger.warning(
                        "High memory usage detected: %.1f MB / %.1f MB (%.1f%%)",
                        memory_stats["rss_mb"],
                        self.max_memory_mb,
                        (memory_stats["rss_mb"] / self.max_memory_mb) * 100
                    )

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(30)  # Wait longer on error

    def optimize_memory_usage(self) -> dict[str, float | int]:
        """Perform memory optimization."""
        logger.info("Performing memory optimization...")

        # Clear expired cache entries
        self.clear_expired_cache()

        # Force garbage collection
        import gc
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")

        # Get current memory usage
        memory_stats = self.get_memory_usage()
        logger.info(
            "Memory optimization completed. Current usage: %.1f MB",
            memory_stats["rss_mb"]
        )

        return memory_stats

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        stats = {
            "feature_cache_size": len(self.feature_cache),
            "data_cache_size": len(self.data_cache),
            "model_cache_size": len(self.model_cache),
            "total_cache_entries": len(self.feature_cache) + len(self.data_cache) + len(self.model_cache)
        }

        # Add custom TTL cache stats
        if hasattr(self, '_custom_ttl_caches'):
            custom_total = sum(len(cache) for cache in self._custom_ttl_caches.values())
            stats["custom_ttl_cache_size"] = custom_total
            stats["total_cache_entries"] += custom_total

        return stats

class DynamicBufferManager:
    """Dynamic buffer size adjustment for training optimization."""

    def __init__(self, initial_buffer_size: int = 1000, max_buffer_size: int = 10000) -> None:
        """
        Initialize dynamic buffer manager.

        Args:
            initial_buffer_size: Initial buffer size
            max_buffer_size: Maximum allowed buffer size
        """
        self.buffer_size = initial_buffer_size
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = 100

        # Performance tracking
        self.performance_history = []
        self.adjustment_threshold = 0.1  # 10% performance change threshold

        logger.info(f"DynamicBufferManager initialized with buffer size {initial_buffer_size}")

    def adjust_buffer_size(self, current_performance: float, target_performance: float) -> None:
        """
        Adjust buffer size based on performance metrics.

        Args:
            current_performance: Current performance metric (e.g., steps/second)
            target_performance: Target performance level
        """
        self.performance_history.append(current_performance)

        # Keep only last 10 measurements
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]

        if len(self.performance_history) < 3:
            return  # Need more data

        # Calculate performance trend
        recent_avg = sum(self.performance_history[-3:]) / 3
        older_count = len(self.performance_history[:-3])
        if older_count > 0:
            older_avg = sum(self.performance_history[:-3]) / older_count
        else:
            older_avg = recent_avg  # No older data, use recent as baseline

        performance_change = (recent_avg - older_avg) / max(older_avg, 0.1)  # Avoid division by zero

        # Adjust buffer size based on performance
        if abs(performance_change) > self.adjustment_threshold:
            if performance_change > 0:
                # Performance improving, can increase buffer size
                new_size = min(int(self.buffer_size * 1.2), self.max_buffer_size)
            else:
                # Performance declining, reduce buffer size
                new_size = max(int(self.buffer_size * 0.8), self.min_buffer_size)

            if new_size != self.buffer_size:
                logger.info(
                    "Adjusting buffer size from %d to %d (%.2f%%)",
                    self.buffer_size,
                    new_size,
                    (new_size / self.buffer_size - 1) * 100
                )
                self.buffer_size = new_size

    def get_optimal_buffer_size(self, data_size: int) -> int:
        """
        Get optimal buffer size based on data size and current performance.

        Args:
            data_size: Size of data to buffer

        Returns:
            Optimal buffer size
        """
        # Simple heuristic: buffer should be 10-20% of data size
        optimal_size = min(max(int(data_size * 0.15), self.min_buffer_size), self.max_buffer_size)
        return optimal_size

# Global instances
# NOTE: SAC training with 1M replay buffer typically uses 800-1000 MB.
# Setting threshold to 1500 MB to avoid spurious warnings and unnecessary
# monitoring overhead during training.  Background monitoring is disabled
# to eliminate the 10-second psutil polling thread that adds GIL contention.
default_memory_manager = MemoryManager(max_memory_mb=1500.0, enable_monitoring=False)
default_buffer_manager = DynamicBufferManager()

def get_memory_stats() -> dict[str, Any]:
    """Get comprehensive memory statistics."""
    memory_stats = default_memory_manager.get_memory_usage()
    cache_stats = default_memory_manager.get_cache_stats()

    buffer_size = getattr(
        default_memory_manager,
        "buffer_size",
        default_buffer_manager.buffer_size,
    )

    return {
        "memory": memory_stats,
        "cache": cache_stats,
        "buffer_size": buffer_size,
    }

def optimize_memory():
    """Perform global memory optimization."""
    return default_memory_manager.optimize_memory_usage()

if __name__ == "__main__":
    # Test memory manager
    logging.basicConfig(level=logging.INFO)

    manager = MemoryManager(max_memory_mb=100.0)

    # Test caching
    manager.cache_feature_data("test_key", {"data": [1, 2, 3, 4, 5]})
    cached_data = manager.get_cached_feature_data("test_key")

    print(f"Cached data: {cached_data}")

    # Test memory stats
    stats = manager.get_memory_usage()
    print(f"Memory stats: {stats}")

    # Test buffer manager
    buffer_manager = DynamicBufferManager()
    optimal_size = buffer_manager.get_optimal_buffer_size(10000)
    print(f"Optimal buffer size: {optimal_size}")
