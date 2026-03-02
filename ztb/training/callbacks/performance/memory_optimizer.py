#!/usr/bin/env python3
"""
Memory Optimization Module.

This module provides memory-efficient data structures and cleanup mechanisms
to prevent memory leaks and optimize performance in the callback system.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
import weakref
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, cast

import psutil

from ztb.trading.environment.constants import BYTES_PER_MB

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")
RefGetter = Callable[[], object | None]

def _default_pool_object() -> object:
    return {}

@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""

    max_cache_size: int = 1000
    cleanup_interval: float = 300.0  # 5 minutes
    retention_hours: int = 24
    max_series_size: int = 5000  # Reduced from 10000
    enable_weak_refs: bool = True
    memory_threshold_mb: int = 500  # Trigger cleanup at 500MB
    enable_gc_optimization: bool = True

class _ThreadSafeStatsBase:
    """Shared lock/rate helpers for memory utility components."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

class LRUCache(_ThreadSafeStatsBase, Generic[K, V]):
    """
    Thread-safe LRU (Least Recently Used) cache with memory limits.

    Features:
    - Automatic cleanup of old entries
    - Memory usage monitoring
    - Thread-safe operations
    """

    def __init__(self, max_size: int = 1000, enable_weak_refs: bool = True):
        super().__init__()
        self.max_size = max_size
        self.enable_weak_refs = enable_weak_refs
        self._cache: OrderedDict[K, V] = OrderedDict()
        # Public alias expected by tests
        self.cache = self._cache
        self._access_times: dict[K, float] = {}
        self._hit_count = 0
        self._miss_count = 0

    def get(self, key: K) -> V | None:
        """Get an item from the cache."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._access_times[key] = time.time()
                self._hit_count += 1
                return self._cache[key]
            self._miss_count += 1
            return None

    def put(self, key: K, value: V) -> None:
        """Put an item in the cache."""
        with self._lock:
            if key in self._cache:
                # Update existing item
                self._cache.move_to_end(key)
            else:
                # Add new item
                if len(self._cache) >= self.max_size:
                    # Remove least recently used
                    oldest_key, _ = self._cache.popitem(last=False)
                    self._access_times.pop(oldest_key, None)

            self._cache[key] = value
            self._access_times[key] = time.time()

    def remove(self, key: K) -> bool:
        """Remove an item from the cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_times.pop(key, None)
                return True
            return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._hit_count = 0
            self._miss_count = 0

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> dict[str, float | int]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._safe_ratio(float(self._hit_count), float(total_requests))
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "memory_usage_mb": self._estimate_memory_usage(),
            }

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Rough estimation
        return len(self._cache) * 0.001  # Assume ~1KB per entry

class MemoryPool(_ThreadSafeStatsBase, Generic[T]):
    """
    Memory pool for reusing objects to reduce allocation overhead.

    Features:
    - Object pooling to reduce GC pressure
    - Automatic cleanup of unused objects
    - Memory usage monitoring
    """

    def __init__(
        self, pool_size: int = 100, object_factory: Callable[[], T] | None = None
    ):
        super().__init__()
        self.object_factory = object_factory or cast(
            Callable[[], T], _default_pool_object
        )
        self.max_pool_size = pool_size
        self._pool: deque[T] = deque(maxlen=pool_size)
        self._created_count = 0
        self._reused_count = 0

    def acquire(self) -> T:
        """Acquire an object from the pool."""
        with self._lock:
            if self._pool:
                # LIFO reuse to favor recently released objects (better cache locality)
                self._reused_count += 1
                return self._pool.pop()

            self._created_count += 1
            return self.object_factory()

    def release(self, obj: T) -> None:
        """Release an object back to the pool."""
        with self._lock:
            if len(self._pool) < self.max_pool_size:
                # Reset object if it has a reset method
                if hasattr(obj, "reset"):
                    try:
                        reset_fn = getattr(obj, "reset")
                        if callable(reset_fn):
                            reset_fn()
                    except Exception:
                        pass  # Ignore reset errors
                # Use append (right side) and pop() in acquire to implement LIFO
                self._pool.append(obj)

    def get_stats(self) -> dict[str, float | int]:
        """Get pool statistics."""
        with self._lock:
            total = self._created_count + self._reused_count
            reuse_rate = self._safe_ratio(float(self._reused_count), float(total))
            return {
                "pool_size": len(self._pool),
                "max_pool_size": self.max_pool_size,
                "created_count": self._created_count,
                "reused_count": self._reused_count,
                "reuse_rate": reuse_rate,
            }

    @property
    def pool(self) -> list[T]:
        """Public alias to inspect the current pool contents (for tests)."""
        with self._lock:
            return list(self._pool)

class MemoryMonitor(_ThreadSafeStatsBase):
    """
    Memory usage monitor with automatic cleanup triggers.

    Features:
    - Real-time memory monitoring
    - Automatic cleanup when thresholds are exceeded
    - Memory leak detection
    """

    def __init__(self, config: MemoryConfig | None = None):
        super().__init__()
        self.config = config or MemoryConfig()
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._last_memory_mb = 0.0
        self._memory_history: deque[float] = deque(maxlen=100)
        self._cleanup_callbacks: list[Callable[[], None]] = []

    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self._monitoring and self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="memory-monitor", daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Memory monitoring started")

    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        self._monitor_thread = None
        self.logger.info("Memory monitoring stopped")

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback to be called during cleanup."""
        with self._lock:
            self._cleanup_callbacks.append(callback)

    def get_memory_stats(self) -> dict[str, object]:
        """Get current memory statistics."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = float(memory_info.rss) / BYTES_PER_MB

            with self._lock:
                self._memory_history.append(memory_mb)
                self._last_memory_mb = memory_mb
                peak_mb = max(self._memory_history) if self._memory_history else 0.0
                average_mb = (
                    sum(self._memory_history) / len(self._memory_history)
                    if self._memory_history
                    else 0.0
                )

            threshold = float(self.config.memory_threshold_mb)
            exceeded = memory_mb > threshold
            pressure = self._safe_ratio(memory_mb, threshold)

            return {
                "current_mb": memory_mb,
                "peak_mb": peak_mb,
                "average_mb": average_mb,
                "threshold_mb": threshold,
                "exceeded_threshold": exceeded,
                "memory_pressure": pressure,
            }
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {e}")
            return {
                "current_mb": 0.0,
                "peak_mb": 0.0,
                "average_mb": 0.0,
                "threshold_mb": float(self.config.memory_threshold_mb),
                "exceeded_threshold": False,
                "memory_pressure": 0.0,
                "error": str(e),
            }

    def force_cleanup(self) -> None:
        """Force immediate cleanup."""
        self.logger.info("Forcing memory cleanup")

        with self._lock:
            callbacks = list(self._cleanup_callbacks)

        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Error in cleanup callback: {e}")

        # Run garbage collection
        if self.config.enable_gc_optimization:
            collected = gc.collect()
            self.logger.info(f"Garbage collection collected {collected} objects")

    def emergency_cleanup(self) -> None:
        """Compatibility alias used by distributed integration paths."""
        self.force_cleanup()

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                if bool(stats.get("exceeded_threshold", False)):
                    self.logger.warning(
                        "Memory threshold exceeded: "
                        f"{stats.get('current_mb', 0.0):.1f}MB > {stats.get('threshold_mb', 0.0)}MB"
                    )
                    self.force_cleanup()

                time.sleep(max(0.1, self.config.cleanup_interval))

            except Exception as e:
                self.logger.error(f"Error in memory monitor loop: {e}")
                time.sleep(60.0)  # Back off on errors

class WeakRefRegistry(_ThreadSafeStatsBase):
    """
    Registry for weak references to prevent memory leaks.

    Features:
    - Automatic cleanup of dead references
    - Thread-safe operations
    - Memory leak prevention
    """

    def __init__(self):
        super().__init__()
        self._refs: dict[str, RefGetter] = {}

    def register(self, name: object, obj: object | None = None) -> None:
        """Register an object with weak reference.

        Accept both signatures for compatibility with existing tests:
        - register(name: str, obj: object)
        - register(obj: object, name: str)
        """
        with self._lock:
            # Accept either signature: register(name, obj) or register(obj, name)
            if isinstance(name, str) and obj is not None:
                key = name
                value = obj
            elif isinstance(obj, str):
                # Passed (obj, name)
                value = name
                key = obj
            else:
                # Fallback: use string representation as key
                key = str(name)
                value = obj

            try:
                # Store a weakref where possible; otherwise store a callable returning the value
                ref = weakref.ref(
                    value,
                    lambda _ref, ref_name=key: self._cleanup_dead_ref(ref_name),
                )
                self._refs[key] = ref
            except TypeError:
                # Unweakrefable (e.g., list of primitives); store a callable provider
                self._refs[key] = lambda captured=value: captured

    def cleanup(self) -> int:
        """Alias for cleanup_dead_refs used by tests."""
        return self.cleanup_dead_refs()

    def unregister(self, name: str) -> None:
        """Unregister an object."""
        with self._lock:
            self._refs.pop(name, None)

    def get(self, name: str) -> object | None:
        """Get an object by name."""
        with self._lock:
            ref = self._refs.get(name)
            return ref() if ref else None

    def cleanup_dead_refs(self) -> int:
        """Clean up dead references and return count of cleaned refs."""
        with self._lock:
            dead_names: list[str] = []
            for name, ref in self._refs.items():
                if ref() is None:
                    dead_names.append(name)

            for name in dead_names:
                del self._refs[name]

            return len(dead_names)

    def _cleanup_dead_ref(self, name: str) -> None:
        """Callback for when a weak reference dies."""
        with self._lock:
            self._refs.pop(name, None)

    @property
    def registry(self) -> dict[str, RefGetter]:
        """Public snapshot of registered weak references."""
        with self._lock:
            return dict(self._refs)

# Global instances
_global_memory_monitor: MemoryMonitor | None = None
_global_weak_registry: WeakRefRegistry | None = None

def get_global_memory_monitor() -> MemoryMonitor:
    """Get the global memory monitor instance."""
    global _global_memory_monitor
    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor()
    return _global_memory_monitor

def get_global_weak_registry() -> WeakRefRegistry:
    """Get the global weak reference registry instance."""
    global _global_weak_registry
    if _global_weak_registry is None:
        _global_weak_registry = WeakRefRegistry()
    return _global_weak_registry
