"""
Memory management utilities for Zaif Trade Bot.

This module provides context managers and utilities for efficient memory management,
particularly for temporary arrays and large data structures.
Enhanced with TTLCache memory management integration.
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, TypeVar

import numpy as np
import psutil
from numpy.typing import NDArray

from ztb.cache.memory_cache import default_memory_manager
# 117# A-fix: Break circular import (utils → trading.environment → torch)
BYTES_PER_MB = 1024 * 1024
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=np.ndarray)


@contextmanager
def temporary_array(*args: Any, **kwargs: Any) -> Generator[NDArray[Any], None, None]:
    """
    Context manager for temporary numpy arrays.

    Automatically cleans up memory when exiting the context.
    Useful for large temporary arrays that should be freed immediately after use.

    Args:
        *args: Arguments to pass to np.array()
        **kwargs: Keyword arguments to pass to np.array()

    Yields:
        The created numpy array

    Example:
        with temporary_array(data, dtype=np.float32) as arr:
            result = process_array(arr)
        # arr is automatically deleted here
    """
    arr = np.array(*args, **kwargs)
    try:
        yield arr
    finally:
        # Force garbage collection of the array
        del arr


def memory_efficient_processing(
    data: NDArray[Any], chunk_size: Optional[int] = None
) -> Generator[NDArray[Any], None, None]:
    """
    Generator for memory-efficient processing of large arrays.

    Automatically chunks large arrays for processing to avoid memory issues.

    Args:
        data: Input array to process
        chunk_size: Size of chunks to process (auto-determined if None)

    Yields:
        Chunked array segments
    """
    if chunk_size is None:
        # Auto-determine chunk size based on available memory
        chunk_size = min(10000, len(data) // 4 + 1)

    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size]
        try:
            yield chunk
        finally:
            # Clean up chunk
            del chunk


class OperationMemoryTracker:
    """
    Track memory usage of operations with TTLCache integration.

    Useful for debugging memory leaks and optimizing memory usage.
    Enhanced with MemoryManager integration for comprehensive tracking.
    """

    def __init__(self, enable_cache_tracking: bool = True) -> None:
        super().__init__()
        self._initial_memory = 0
        self._peak_memory = 0
        self._cache_initial_size = 0
        self.enable_cache_tracking = enable_cache_tracking
        self.memory_manager = default_memory_manager if enable_cache_tracking else None

    def __enter__(self) -> Any:
        process = psutil.Process()
        self._initial_memory = process.memory_info().rss

        # Track cache size if enabled
        if self.memory_manager:
            cache_stats = self.memory_manager.get_cache_stats()
            self._cache_initial_size = cache_stats["total_cache_entries"]

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        process = psutil.Process()
        final_memory = process.memory_info().rss
        memory_delta = final_memory - self._initial_memory

        # Get cache statistics
        cache_final_size = 0
        if self.memory_manager:
            cache_stats = self.memory_manager.get_cache_stats()
            cache_final_size = cache_stats["total_cache_entries"]

        cache_delta = cache_final_size - self._cache_initial_size

        logger.debug(
            f"Memory usage: initial={self._initial_memory / BYTES_PER_MB:.1f}MB, "
            f"final={final_memory / BYTES_PER_MB:.1f}MB, "
            f"delta={memory_delta / BYTES_PER_MB:+.1f}MB, "
            f"cache_delta={cache_delta:+d} entries"
        )

        if memory_delta > 50 * BYTES_PER_MB:  # 50MB threshold
            logger.warning(
                f"Large memory increase detected: {memory_delta / BYTES_PER_MB:.1f}MB"
            )

        # Trigger memory optimization if memory usage is high
        if self.memory_manager and final_memory > 1500 * BYTES_PER_MB:  # 1500MB threshold
            logger.info("High memory usage detected, triggering optimization...")
            self.memory_manager.optimize_memory_usage()

        # Memory leak prevention: only run GC on large memory increases
        # Unconditional gc.collect() per-call was causing severe performance degradation
        if memory_delta > 50 * BYTES_PER_MB:
            import gc

            collected = gc.collect()
            if collected > 0:
                logger.debug(f"Garbage collection freed {collected} objects")


def optimize_array_dtype(arr: NDArray[Any]) -> NDArray[Any]:
    """
    Optimize array dtype for memory efficiency.

    Args:
        arr: Input array

    Returns:
        Array with optimized dtype
    """
    if arr.dtype == np.float64 and arr.max() < 1e6 and arr.min() > -1e6:
        # Convert float64 to float32 if values are in reasonable range
        return arr.astype(np.float32)
    elif arr.dtype == np.int64 and arr.max() < 2**31 and arr.min() >= -(2**31):
        # Convert int64 to int32 if values fit
        return arr.astype(np.int32)

    return arr


def cleanup_training_memory(
    env: Optional[Any] = None,
    model: Optional[Any] = None,
    data_cache: Optional[dict] = None,
    force_gc: bool = True,
    optimize_cache: bool = True,
) -> None:
    """
    Perform comprehensive memory cleanup after training operations.

    Enhanced with MemoryManager integration for cache optimization.

    Args:
        env: Training environment to close
        model: Model object (not deleted if might be saved)
        data_cache: Data cache dictionary to clear
        force_gc: Whether to force garbage collection
        optimize_cache: Whether to optimize memory cache
    """
    import gc

    try:
        # Clear data cache
        if data_cache is not None:
            data_cache.clear()
            logger.debug("Cleared data cache")

        # Close environment
        if env is not None and hasattr(env, "close"):
            env.close()
            logger.debug("Closed training environment")

        # Clear model references (but don't delete if it might be saved)
        if model is not None:
            # Just log, don't delete
            logger.debug("Model cleanup skipped (may be saved)")

        # Optimize memory cache
        if optimize_cache:
            default_memory_manager.optimize_memory_usage()
            logger.debug("Optimized memory cache")

        # Force garbage collection
        if force_gc:
            collected = gc.collect()
            logger.debug(f"Garbage collection completed: {collected} objects collected")

        logger.info("Training memory cleanup completed")

    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.

    Enhanced with MemoryManager integration for comprehensive statistics.

    Returns:
        Dictionary with memory usage information in MB and cache stats
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()

        base_stats = {
            "rss": memory_info.rss / BYTES_PER_MB,  # Resident Set Size
            "vms": memory_info.vms / BYTES_PER_MB,  # Virtual Memory Size
            "percent": process.memory_percent(),
        }

        # Add cache statistics from MemoryManager
        cache_stats = default_memory_manager.get_cache_stats()
        base_stats.update(
            {
                "cache_feature_entries": cache_stats["feature_cache_size"],
                "cache_data_entries": cache_stats["data_cache_size"],
                "cache_model_entries": cache_stats["model_cache_size"],
                "cache_total_entries": cache_stats["total_cache_entries"],
            }
        )

        return base_stats

    except ImportError:
        logger.warning("psutil not available for memory monitoring")
        return {
            "rss": 0.0,
            "vms": 0.0,
            "percent": 0.0,
            "cache_feature_entries": 0,
            "cache_data_entries": 0,
            "cache_model_entries": 0,
            "cache_total_entries": 0,
        }
    except Exception as e:
        logger.warning(f"Failed to get memory usage: {e}")
        return {
            "rss": 0.0,
            "vms": 0.0,
            "percent": 0.0,
            "cache_feature_entries": 0,
            "cache_data_entries": 0,
            "cache_model_entries": 0,
            "cache_total_entries": 0,
        }


def check_memory_pressure(threshold_mb: float = 1000.0) -> bool:
    """
    Check if memory usage is above threshold.

    Enhanced with cache-aware memory pressure detection.

    Args:
        threshold_mb: Memory threshold in MB

    Returns:
        True if memory usage is above threshold
    """
    memory = get_memory_usage()

    # Check RSS memory
    memory_pressure = memory["rss"] > threshold_mb

    # Also check cache size as additional pressure indicator
    cache_pressure = (
        memory.get("cache_total_entries", 0) > 1000
    )  # Arbitrary cache size threshold

    if memory_pressure or cache_pressure:
        logger.warning(
            f"Memory pressure detected: RSS={memory['rss']:.1f}MB/{threshold_mb:.1f}MB, "
            f"Cache={memory.get('cache_total_entries', 0)} entries"
        )
        return True

    return False


def cleanup_memory(
    caches: Optional[Dict[str, Any]] = None,
    managers: Optional[Dict[str, Any]] = None,
    force_gc: bool = True,
    optimize_cache: bool = True,
) -> None:
    """
    Perform comprehensive memory cleanup for general use.

    Args:
        caches: Dictionary of cache objects to clear (name -> cache_object)
        managers: Dictionary of manager objects to cleanup (name -> manager_object)
        force_gc: Whether to force garbage collection
        optimize_cache: Whether to optimize memory cache
    """
    import gc

    try:
        # Clear caches
        if caches is not None:
            for name, cache in caches.items():
                if hasattr(cache, "clear"):
                    cache.clear()
                    logger.debug(f"Cleared cache: {name}")
                elif hasattr(cache, "cleanup"):
                    cache.cleanup()
                    logger.debug(f"Cleaned up cache: {name}")

        # Cleanup managers
        if managers is not None:
            for name, manager in managers.items():
                if hasattr(manager, "cleanup"):
                    manager.cleanup()
                    logger.debug(f"Cleaned up manager: {name}")
                elif hasattr(manager, "clear_cache"):
                    manager.clear_cache()
                    logger.debug(f"Cleared cache for manager: {name}")

        # Optimize memory cache
        if optimize_cache:
            default_memory_manager.optimize_memory_usage()
            logger.debug("Optimized memory cache")

        # Force garbage collection
        if force_gc:
            collected = gc.collect()
            logger.debug(f"Garbage collection completed: {collected} objects collected")

        logger.info("Memory cleanup completed")

    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")
