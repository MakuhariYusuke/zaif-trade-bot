#!/usr/bin/env python3
"""Shared JSON compatibility helpers for v459 scripts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ztb.io.json_io import write_json


def to_json_compatible(value: object) -> object:
    """Convert nested objects into JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return [to_json_compatible(item) for item in value.tolist()]

    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()

    if isinstance(value, Mapping):
        return {str(key): to_json_compatible(item) for key, item in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_compatible(item) for item in value]

    # Handle pandas scalar NA-like values.
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # type: ignore[no-any-return]
        except Exception:
            pass

    return str(value)


def write_json_compatible(path: str | Path, payload: object) -> Path:
    """Write payload after converting it into JSON-compatible object graph."""
    serializable = to_json_compatible(payload)
    return write_json(path, serializable, indent=2, ensure_ascii=False)


__all__ = ["to_json_compatible", "write_json_compatible"]
