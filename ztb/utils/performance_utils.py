"""
Performance monitoring utilities for Zaif Trade Bot.

This module provides decorators and utilities for monitoring function performance,
memory usage, and execution times.
"""

import logging
import time
import types as _types
from functools import wraps
from typing import Any, Callable, TypeVar

import psutil

# 117# A-fix: Break circular import (utils → trading.environment → torch)
BYTES_PER_MB = 1024 * 1024
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# GPU monitoring (optional) - import torch lazily to avoid ABI mismatch crashes at test-time
# TORCH_AVAILABLE will be set dynamically when needed.
TORCH_AVAILABLE = False
torch = None

def timed(func: F) -> F:
    """
    Decorator that logs function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.debug(f"{func.__name__} took {duration:.4f}s")

    return wrapper  # type: ignore[return-value]

def timed_with_memory(func: F) -> F:
    """
    Decorator that logs function execution time and memory usage.

    Args:
        func: Function to monitor

    Returns:
        Wrapped function that logs execution time and memory delta
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        process = psutil.Process()
        start_time = time.perf_counter()
        start_memory = process.memory_info().rss

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            logger.debug(
                f"{func.__name__}: {duration:.4f}s, "
                f"memory delta: {memory_delta / BYTES_PER_MB:+.1f}MB"
            )

    return wrapper  # type: ignore[return-value]

class CodePerformanceMonitor:
    """
    Context manager for monitoring code block performance.
    """

    def __init__(self, name: str, log_level: int = logging.DEBUG):
        self.name = name
        self.log_level = log_level
        self.start_time: float | None = None
        self.start_memory: int | None = None
        self.start_gpu_memory: int | None = None

    def __enter__(self) -> None:
        self.start_time = time.perf_counter()
        process = psutil.Process()
        self.start_memory = process.memory_info().rss

        # GPU memory monitoring - import torch lazily to avoid module import failures
        global TORCH_AVAILABLE, torch
        try:
            if not TORCH_AVAILABLE:
                import importlib

                torch_mod = importlib.import_module("torch")
                torch = torch_mod
                TORCH_AVAILABLE = True
        except Exception:
            TORCH_AVAILABLE = False
            torch = None

        if TORCH_AVAILABLE and torch is not None:
            try:
                if torch.cuda.is_available():
                    self.start_gpu_memory = torch.cuda.memory_allocated()
            except Exception:
                self.start_gpu_memory = None

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: _types.TracebackType | None,
    ) -> None:
        if self.start_time is None:
            return

        end_time = time.perf_counter()
        process = psutil.Process()
        end_memory = process.memory_info().rss

        duration = end_time - self.start_time
        memory_delta = end_memory - self.start_memory

        # GPU memory monitoring - import torch lazily in case it was not loaded yet
        gpu_memory_info = ""
        try:
            if not TORCH_AVAILABLE:
                import importlib

                torch_mod = importlib.import_module("torch")
                torch = torch_mod
                TORCH_AVAILABLE = True
        except Exception:
            TORCH_AVAILABLE = False
            torch = None

        if (
            TORCH_AVAILABLE
            and torch is not None
            and hasattr(torch, "cuda")
            and torch.cuda.is_available()
            and self.start_gpu_memory is not None
        ):
            try:
                end_gpu_memory = torch.cuda.memory_allocated()
                gpu_memory_delta = end_gpu_memory - self.start_gpu_memory
                gpu_memory_info = f", GPU: {gpu_memory_delta / BYTES_PER_MB:+.1f}MB"
            except Exception:
                pass

        logger.log(
            self.log_level,
            f"{self.name}: {duration:.4f}s, "
            f"CPU memory: {memory_delta / BYTES_PER_MB:+.1f}MB{gpu_memory_info}",
        )

# Backwards compatibility alias
PerformanceMonitor = CodePerformanceMonitor

def profile_function(
    func: F, sample_rate: float = 1.0, log_threshold: float = 0.1
) -> F:
    """
    Decorator that profiles function execution with sampling.

    Args:
        func: Function to profile
        sample_rate: Fraction of calls to profile (0.0 to 1.0)
        log_threshold: Minimum execution time to log (seconds)

    Returns:
        Wrapped function with profiling
    """
    import random

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if random.random() > sample_rate:
            return func(*args, **kwargs)

        with CodePerformanceMonitor(f"sampled_{func.__name__}"):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time

            if duration >= log_threshold:
                logger.info(f"{func.__name__} execution time: {duration:.4f}s")

        return result

    return wrapper  # type: ignore[return-value]
