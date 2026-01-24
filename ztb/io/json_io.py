"""
JSON read/write helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from ztb.io.common import PathLike, _to_path, atomic_write_text


def read_json(path: PathLike, encoding: str = "utf-8") -> Any:
    target = _to_path(path)
    return json.loads(target.read_text(encoding=encoding))


def write_json(
    path: PathLike,
    data: Any,
    indent: int = 2,
    ensure_ascii: bool = False,
    default: Optional[Any] = str,
    encoding: str = "utf-8",
) -> Path:
    content = json.dumps(
        data, indent=indent, ensure_ascii=ensure_ascii, default=default
    )
    return atomic_write_text(path, content, encoding=encoding)
