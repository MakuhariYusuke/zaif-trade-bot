"""
Shared helpers for IO utilities.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def _to_path(path: PathLike) -> Path:
    return path if isinstance(path, Path) else Path(path)


def ensure_parent_dir(path: PathLike) -> Path:
    target = _to_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def atomic_write_text(path: PathLike, content: str, encoding: str = "utf-8") -> Path:
    target = ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding=encoding, delete=False, dir=str(target.parent)
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, target)
    return target


def atomic_write_bytes(path: PathLike, content: bytes) -> Path:
    target = ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=str(target.parent)
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, target)
    return target
