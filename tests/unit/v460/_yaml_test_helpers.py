from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def parse_yaml_mapping(yaml_text: str) -> dict[str, object]:
    """Parse YAML text and require a top-level mapping."""
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise TypeError("expected YAML mapping")
    return data


@lru_cache(maxsize=None)
def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load YAML from a file and require a top-level mapping."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"expected YAML mapping in {path}")
    return data
