"""
Shared JSON state persistence helpers for production components.
"""

from __future__ import annotations

from ztb.io.common import PathLike
from ztb.io.state_persistence import (
    read_state_payload as _read_state_payload,
    write_state_payload as _write_state_payload,
)


def write_state_payload(filepath: PathLike, state: object) -> None:
    """Write component state payload as JSON."""
    _write_state_payload(filepath, state)


def read_state_payload(filepath: PathLike) -> dict[str, object]:
    """Read component state payload from JSON object."""
    return _read_state_payload(filepath)
