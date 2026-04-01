"""Generic temporary-file and atomic replacement helpers.

These helpers are intentionally domain-agnostic so they can be reused by
training, checkpointing, and scheduler code without importing v460-specific
modules.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Callable


def atomic_replace_with_tmp(
    *,
    target_path: Path,
    prefix: str,
    suffix: str,
    writer: Callable[[str], None],
) -> None:
    """Write a file via tmp + replace to avoid partial artifacts."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd_tmp, tmp_path = tempfile.mkstemp(
        dir=str(target_path.parent),
        prefix=prefix,
        suffix=suffix,
    )
    os.close(fd_tmp)
    try:
        writer(tmp_path)
        os.replace(tmp_path, str(target_path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def capture_bytes_via_tmpfile(
    *,
    suffix: str,
    writer: Callable[[str], None],
    prefix: str = ".capture_",
) -> bytes:
    """Capture writer output by pointing it at a temporary file."""
    fd_tmp, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd_tmp)
    try:
        writer(tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def restore_bytes_via_tmpfile(
    *,
    data: bytes,
    suffix: str,
    reader: Callable[[str], None],
    prefix: str = ".restore_",
) -> None:
    """Restore bytes into an API that expects a file path."""
    fd_tmp, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd_tmp)
    try:
        Path(tmp_path).write_bytes(data)
        reader(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
