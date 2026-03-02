"""
Shared JSON state persistence helpers.
"""

from __future__ import annotations

from ztb.io.common import PathLike
from ztb.io.json_io import read_json_object, write_json

def write_state_payload(filepath: PathLike, state: object) -> None:
    """Write state payload as JSON."""
    write_json(filepath, state, indent=2, ensure_ascii=False)

def read_state_payload(filepath: PathLike) -> dict[str, object]:
    """Read state payload as JSON object."""
    return read_json_object(filepath)
