"""
Text read/write helpers.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from ztb.io.common import PathLike, _to_path, atomic_write_text


def read_text(path: PathLike, encoding: str = "utf-8") -> str:
    target = _to_path(path)
    return target.read_text(encoding=encoding)


def write_text(path: PathLike, content: str, encoding: str = "utf-8") -> Path:
    return atomic_write_text(path, content, encoding=encoding)


def read_last_lines(
    path: PathLike,
    *,
    count: int = 100,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> list[str]:
    """Read up to `count` trailing lines without loading full file into memory."""
    if count <= 0:
        return []

    target = _to_path(path)
    with open(target, "r", encoding=encoding, errors=errors) as f:
        return list(deque(f, maxlen=count))
