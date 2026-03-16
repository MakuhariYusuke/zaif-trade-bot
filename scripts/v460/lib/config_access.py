"""Small typed helpers for v460 config access.

Centralizes repetitive config extraction/casting used across v460 tasks.

Why ``value: object``?
    YAML の ``cfg.get(key)`` は ``str | int | float | bool | list | dict | None``
    のいずれかを返す。これらの *最も狭い共通基底型* は ``object`` であり、
    ``Any`` を使わずに型安全を維持するために意図的に ``object`` を採用している。
    各 ``as_*`` 関数内で ``isinstance`` ガードを行い、安全にキャストする。

    259# P3-3: この設計意図の明文化。
"""

from __future__ import annotations

from typing import cast

from ztb.types.common import ConfigSection, is_config_dict


def section(cfg: ConfigSection, key: str) -> ConfigSection:
    value = cfg.get(key)
    if is_config_dict(value):
        return cast(ConfigSection, value)
    return {}


def as_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def as_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default
