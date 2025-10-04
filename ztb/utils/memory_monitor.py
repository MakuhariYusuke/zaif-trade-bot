#!/usr/bin/env python3
"""
memory_monitor.py
Memory monitoring utilities for development and testing
"""

from typing import cast

import psutil

from ztb.utils.config import ZTBConfig


def check_memory_usage(threshold_mb: int = 1000) -> None:
    """
    Check current memory usage and warn if above threshold.

    Args:
        threshold_mb: Memory usage threshold in MB
    """
    config = ZTBConfig()
    if config.get_bool("ZTB_DEV_MEMORY_WARN"):
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > threshold_mb:
            print(
                f"WARNING: High memory usage: {memory_mb:.1f}MB (threshold: {threshold_mb}MB)"
            )


def get_memory_usage() -> float:
    """
    Get current memory usage in MB.

    Returns:
        Memory usage in MB
    """
    process = psutil.Process()
    return cast(float, process.memory_info().rss / 1024 / 1024)


def log_memory_usage(label: str = "") -> None:
    """
    Log current memory usage with optional label.

    Args:
        label: Optional label for the log message
    """
    memory_mb = get_memory_usage()
    label_str = f" [{label}]" if label else ""
    print(f"Memory usage{label_str}: {memory_mb:.1f}MB")
