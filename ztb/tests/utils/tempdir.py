#!/usr/bin/env python3
"""
tempdir.py
Temporary directory utilities for test isolation
"""

import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


def get_unique_temp_dir(prefix: str = "test") -> str:
    """
    Generate a unique temporary directory path with timestamp and UUID for test isolation.

    Args:
        prefix: Directory name prefix

    Returns:
        Absolute path to unique temporary directory
    """
    unique_id = uuid.uuid4().hex[:8]
    timestamp = int(time.time() * 1000000)  # Microsecond precision
    temp_base = Path(tempfile.gettempdir())
    temp_dir = temp_base / f"{prefix}_{timestamp}_{unique_id}"
    return str(temp_dir)


@contextmanager
def TempDirManager(prefix: str = "test") -> Generator[str, None, None]:
    """
    Context manager for creating and cleaning up unique temporary directories.

    Usage:
        with TempDirManager("my_test") as temp_dir:
            # Use temp_dir
            pass
        # Automatically cleaned up
    """
    temp_dir = get_unique_temp_dir(prefix)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        yield temp_dir
    finally:
        # Clean up
        try:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors in tests


def cleanup_temp_dir(temp_dir: str) -> None:
    """
    Manually clean up a temporary directory.

    Args:
        temp_dir: Path to directory to remove
    """
    try:
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass  # Ignore cleanup errors
