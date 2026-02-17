"""
JSON read/write helpers.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from ztb.io.common import PathLike, _to_path, atomic_write_text


JSONObject = dict[str, object]
JSONArray = list[object]
JSONDefaultEncoder = Callable[[object], object]


def read_json(path: PathLike, encoding: str = "utf-8") -> object:
    target = _to_path(path)
    return json.loads(target.read_text(encoding=encoding))


def read_json_object(path: PathLike, encoding: str = "utf-8") -> JSONObject:
    data = read_json(path, encoding=encoding)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def read_json_array(path: PathLike, encoding: str = "utf-8") -> JSONArray:
    data = read_json(path, encoding=encoding)
    if not isinstance(data, list):
        raise TypeError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return data


def write_json(
    path: PathLike,
    data: object,
    indent: int = 2,
    ensure_ascii: bool = False,
    default: Optional[JSONDefaultEncoder] = str,
    encoding: str = "utf-8",
) -> Path:
    content = json.dumps(
        data, indent=indent, ensure_ascii=ensure_ascii, default=default
    )
    return atomic_write_text(path, content, encoding=encoding)
