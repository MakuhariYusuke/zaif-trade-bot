"""
Memory management utilities for Zaif Trade Bot.

This module provides context managers and utilities for efficient memory management,
particularly for temporary arrays and large data structures.
"""

from contextlib import contextmanager
from typing import Any, Generator, Optional, TypeVar
import numpy as np
from numpy.typing import NDArray

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

T = TypeVar('T', bound=np.ndarray)


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


@contextmanager
def memory_efficient_processing(
    data: NDArray[Any],
    chunk_size: Optional[int] = None
) -> Generator[NDArray[Any], None, None]:
    """
    Context manager for memory-efficient processing of large arrays.

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
        chunk = data[i:i + chunk_size]
        try:
            yield chunk
        finally:
            # Clean up chunk
            del chunk


class MemoryTracker:
    """
    Track memory usage of operations.

    Useful for debugging memory leaks and optimizing memory usage.
    """

    def __init__(self) -> None:
        super().__init__()
        self._initial_memory = 0
        self._peak_memory = 0

    def __enter__(self):
        import psutil
        process = psutil.Process()
        self._initial_memory = process.memory_info().rss
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        import psutil
        process = psutil.Process()
        final_memory = process.memory_info().rss
        memory_delta = final_memory - self._initial_memory

        logger.debug(
            f"Memory usage: initial={self._initial_memory / 1024 / 1024:.1f}MB, "
            f"final={final_memory / 1024 / 1024:.1f}MB, "
            f"delta={memory_delta / 1024 / 1024:+.1f}MB"
        )

        if memory_delta > 50 * 1024 * 1024:  # 50MB threshold
            logger.warning(f"Large memory increase detected: {memory_delta / 1024 / 1024:.1f}MB")


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
    elif arr.dtype == np.int64 and arr.max() < 2**31 and arr.min() >= -2**31:
        # Convert int64 to int32 if values fit
        return arr.astype(np.int32)

    return arr