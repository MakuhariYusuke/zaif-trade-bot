"""
Text read/write helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ztb.io.common import PathLike, _to_path, atomic_write_text


def read_text(path: PathLike, encoding: str = "utf-8") -> str:
    target = _to_path(path)
    return target.read_text(encoding=encoding)


def write_text(path: PathLike, content: str, encoding: str = "utf-8") -> Path:
    return atomic_write_text(path, content, encoding=encoding)
