"""
Shared JSON state persistence helpers for production components.
"""

from __future__ import annotations

from ztb.io.common import PathLike
from ztb.io.json_io import read_json_object, write_json
from ztb.types.common import ObjectMap


def write_state_payload(filepath: PathLike, state: object) -> None:
    """Write component state payload as JSON."""
    write_json(filepath, state, indent=2, ensure_ascii=False)


def read_state_payload(filepath: PathLike) -> ObjectMap:
    """Read component state payload from JSON object."""
    return read_json_object(filepath)
