"""
YAML read/write helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ztb.io.common import PathLike, _to_path, atomic_write_text


def read_yaml(path: PathLike, encoding: str = "utf-8") -> Any:
    target = _to_path(path)
    return yaml.safe_load(target.read_text(encoding=encoding))


def write_yaml(
    path: PathLike,
    data: Any,
    default_flow_style: bool = False,
    sort_keys: bool = False,
    allow_unicode: bool = False,
    encoding: str = "utf-8",
) -> Path:
    content = yaml.safe_dump(
        data,
        default_flow_style=default_flow_style,
        sort_keys=sort_keys,
        allow_unicode=allow_unicode,
    )
    return atomic_write_text(path, content, encoding=encoding)
