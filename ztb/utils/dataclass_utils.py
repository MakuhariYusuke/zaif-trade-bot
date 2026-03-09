"""Helpers for dataclass-backed mapping sanitization."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

@lru_cache(maxsize=None)
def get_dataclass_field_names(dataclass_type: type[object]) -> frozenset[str]:
    """Return cached field names for a dataclass type."""
    field_map = getattr(dataclass_type, "__dataclass_fields__", None)
    if not isinstance(field_map, Mapping):
        raise TypeError(f"{dataclass_type!r} is not a dataclass type")
    return frozenset(str(name) for name in field_map.keys())

def filter_known_dataclass_fields(
    dataclass_type: type[object],
    values: Mapping[str, object],
) -> dict[str, object]:
    """Keep only keys that match dataclass field names."""
    known_fields = get_dataclass_field_names(dataclass_type)
    return {
        key: value
        for key, value in values.items()
        if key in known_fields
    }
