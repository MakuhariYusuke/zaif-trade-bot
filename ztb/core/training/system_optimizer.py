"""
System-level optimizations for SAC v421.

This module provides comprehensive system optimizations including:
- Memory management and leak prevention
- CPU performance optimization
- I/O improvements with caching
- Integration with existing utilities (MemoryTracker, PerformanceProfiler, TTLCache)
"""

import gc
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np
import psutil
import torch
from torch import nn

from ztb.trading.environment.constants import BYTES_PER_MB
from ztb.utils.cache_utils import TTLCache
from ztb.utils.memory_utils import MemoryTracker
from ztb.utils.performance_profiler import PerformanceProfiler

logger = logging.getLogger(__name__)


class SystemOptimizer:
    """
    Comprehensive system-level optimizer for training pipelines.

    Integrates memory management, performance profiling, and I/O optimizations.
    """

    def __init__(
        self,
        enable_memory_tracking: bool = True,
        enable_performance_profiling: bool = True,
        enable_io_caching: bool = True,
        memory_threshold_mb: float = 100.0,
        cache_ttl_seconds: int = 300,
        gc_interval_steps: int = 100,
    ):
        """
        Initialize system optimizer.

        Args:
            enable_memory_tracking: Enable memory usage tracking
            enable_performance_profiling: Enable CPU performance profiling
            enable_io_caching: Enable I/O caching with TTL
            memory_threshold_mb: Memory usage threshold for warnings (MB)
            cache_ttl_seconds: TTL for cache entries in seconds
            gc_interval_steps: Interval for garbage collection (training steps)
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_performance_profiling = enable_performance_profiling
        self.enable_io_caching = enable_io_caching
        self.memory_threshold_mb = memory_threshold_mb
        self.cache_ttl_seconds = cache_ttl_seconds
        self.gc_interval_steps = gc_interval_steps

        # Initialize components
        self.memory_tracker = MemoryTracker() if enable_memory_tracking else None
        self.performance_profiler = (
            PerformanceProfiler() if enable_performance_profiling else None
        )
        self.io_cache = (
            TTLCache(ttl_seconds=cache_ttl_seconds) if enable_io_caching else None
        )

        # Tracking state
        self.step_counter = 0
        self.memory_history: List[float] = []
        self.performance_history: List[Dict[str, float]] = []
        self.cache_hits = 0
        self.cache_misses = 0

        # Threading lock for thread safety
        self._lock = threading.RLock()

    @contextmanager
    def optimize_training_step(self, step_name: str = "training_step"):
        """
        Context manager for optimized training step execution.

        Args:
            step_name: Name of the training step for profiling
        """
        start_time = time.time()

        # Memory tracking
        memory_ctx = self.memory_tracker if self.memory_tracker else None
        if memory_ctx:
            memory_ctx.__enter__()

        # Performance profiling
        perf_enabled = self.performance_profiler is not None
        if perf_enabled:
            perf_start = time.time()

        try:
            yield
        finally:
            # Performance profiling cleanup
            if perf_enabled:
                perf_time = time.time() - perf_start
                # Record performance data manually since PerformanceProfiler doesn't have context manager
                perf_stats = {
                    "step_time": perf_time,
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "step_name": step_name,
                    "timestamp": time.time(),
                }
                with self._lock:
                    self.performance_history.append(perf_stats)

            # Memory tracking cleanup
            if memory_ctx:
                memory_ctx.__exit__(None, None, None)

            # Record metrics
            with self._lock:
                self.step_counter += 1
                step_time = time.time() - start_time

                # Memory metrics
                if self.memory_tracker:
                    current_memory = psutil.Process().memory_info().rss / BYTES_PER_MB
                    self.memory_history.append(current_memory)

                    if current_memory > self.memory_threshold_mb:
                        logger.warning(
                            f"High memory usage in {step_name}: {current_memory:.1f}MB "
                            f"(threshold: {self.memory_threshold_mb}MB)"
                        )

                # Performance metrics
                if self.performance_profiler:
                    perf_stats = {
                        "step_time": step_time,
                        "cpu_percent": psutil.cpu_percent(interval=None),
                        "step_name": step_name,
                        "timestamp": time.time(),
                    }
                    self.performance_history.append(perf_stats)

                # Periodic garbage collection
                if self.step_counter % self.gc_interval_steps == 0:
                    self._perform_garbage_collection()

    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """
        Optimize model memory usage.

        Args:
            model: PyTorch model to optimize

        Returns:
            Optimized model
        """
        if not isinstance(model, nn.Module):
            return model

        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        # Use mixed precision if available
        if torch.cuda.is_available():
            model = model.to(dtype=torch.float16, memory_format=torch.contiguous_format)

        # Pin memory for faster GPU transfers
        if torch.cuda.is_available():
            for param in model.parameters():
                param.data = (
                    param.data.pin_memory() if param.data.is_cuda else param.data
                )

        logger.info("Applied model memory optimizations")
        return model

    def optimize_dataloader(self, dataloader: Any) -> Any:
        """
        Optimize dataloader for memory efficiency and performance.

        Args:
            dataloader: DataLoader to optimize

        Returns:
            Optimized dataloader
        """
        # Configure optimal settings for memory efficiency (only if not already initialized)
        if hasattr(dataloader, "pin_memory") and not hasattr(dataloader, "_iterator"):
            dataloader.pin_memory = torch.cuda.is_available()

        # Note: persistent_workers and prefetch_factor can only be set during DataLoader creation
        # We can't modify them on an existing DataLoader

        logger.info("Applied dataloader optimizations")
        return dataloader

    def cache_io_operation(self, key: str, operation: callable, *args, **kwargs) -> Any:
        """
        Cache I/O operation results using TTL cache.

        Args:
            key: Cache key
            operation: Function to execute if not cached
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Cached or computed result
        """
        if not self.io_cache:
            return operation(*args, **kwargs)

        with self._lock:
            if self.io_cache.get(key) is not None:
                self.cache_hits += 1
                logger.debug(f"Cache hit for key: {key}")
                return self.io_cache.get(key)
            else:
                self.cache_misses += 1
                result = operation(*args, **kwargs)
                self.io_cache.set(key, result)
                logger.debug(f"Cache miss for key: {key}, stored result")
                return result

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            Dictionary with system statistics
        """
        with self._lock:
            stats = {
                "step_counter": self.step_counter,
                "memory_tracking_enabled": self.enable_memory_tracking,
                "performance_profiling_enabled": self.enable_performance_profiling,
                "io_caching_enabled": self.enable_io_caching,
            }

            if self.memory_history:
                stats.update(
                    {
                        "current_memory_mb": self.memory_history[-1]
                        if self.memory_history
                        else 0,
                        "peak_memory_mb": max(self.memory_history)
                        if self.memory_history
                        else 0,
                        "avg_memory_mb": np.mean(self.memory_history)
                        if self.memory_history
                        else 0,
                    }
                )

            if self.performance_history:
                recent_perf = self.performance_history[-10:]  # Last 10 steps
                stats.update(
                    {
                        "avg_step_time": np.mean([p["step_time"] for p in recent_perf]),
                        "avg_cpu_percent": np.mean(
                            [p["cpu_percent"] for p in recent_perf]
                        ),
                    }
                )

            if self.io_cache:
                stats.update(
                    {
                        "cache_size": len(self.io_cache.cache),
                        "cache_hits": self.cache_hits,
                        "cache_misses": self.cache_misses,
                        "cache_hit_rate": self.cache_hits
                        / (self.cache_hits + self.cache_misses)
                        if (self.cache_hits + self.cache_misses) > 0
                        else 0,
                    }
                )

            return stats

    def _perform_garbage_collection(self) -> None:
        """Perform garbage collection to free memory."""
        logger.debug("Performing garbage collection")
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_stats(self) -> None:
        """Reset all collected statistics."""
        with self._lock:
            self.step_counter = 0
            self.memory_history.clear()
            self.performance_history.clear()
            self.cache_hits = 0
            self.cache_misses = 0

            if self.io_cache:
                self.io_cache.clear()

        logger.info("System optimizer statistics reset")


class MemoryOptimizer:
    """
    Specialized memory optimization utilities.
    """

    @staticmethod
    def optimize_tensor_memory(tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize tensor memory usage.

        Args:
            tensor: Input tensor

        Returns:
            Memory-optimized tensor
        """
        # Use contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Optimize dtype if possible
        if tensor.dtype == torch.float64:
            # Check if float32 is sufficient
            if tensor.abs().max() < 1e6:
                tensor = tensor.to(dtype=torch.float32)

        return tensor

    @staticmethod
    def clear_gpu_cache() -> None:
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        process = psutil.Process()
        memory_info = process.memory_info()

        stats = {
            "rss_mb": memory_info.rss / BYTES_PER_MB,
            "vms_mb": memory_info.vms / BYTES_PER_MB,
        }

        if torch.cuda.is_available():
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / BYTES_PER_MB
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / BYTES_PER_MB

        return stats


class PerformanceOptimizer:
    """
    CPU performance optimization utilities.
    """

    @staticmethod
    def optimize_numpy_operations() -> None:
        """Optimize NumPy operations for performance."""
        # Set optimal thread count for NumPy
        import os

        os.environ["OMP_NUM_THREADS"] = str(max(1, psutil.cpu_count() // 2))
        os.environ["MKL_NUM_THREADS"] = str(max(1, psutil.cpu_count() // 2))

    @staticmethod
    def enable_torch_optimizations() -> None:
        """Enable PyTorch performance optimizations."""
        torch.set_num_threads(max(1, psutil.cpu_count() // 2))

        if torch.cuda.is_available():
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    @staticmethod
    def get_cpu_stats() -> Dict[str, float]:
        """
        Get CPU performance statistics.

        Returns:
            Dictionary with CPU statistics
        """
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_current": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        }
