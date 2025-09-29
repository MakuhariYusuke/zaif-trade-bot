#!/usr/bin/env python3
"""
test_utils.py
Test utilities and decorators
"""

import time
from typing import Any, Callable


def timed_test(func: Callable) -> Callable:
    """
    Decorator to add execution time monitoring to test functions.

    Usage:
        @timed_test
        def test_something():
            pass
    """

    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            if duration > 5.0:  # Log slow tests
                print(f"SLOW TEST: {func.__name__} took {duration:.2f}s")
            return result
        except Exception:
            duration = time.time() - start
            print(f"FAILED TEST: {func.__name__} failed after {duration:.2f}s")
            raise

    return wrapper
