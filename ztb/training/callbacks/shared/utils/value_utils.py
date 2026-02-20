#!/usr/bin/env python3
"""Shared value conversion/history helpers for callback implementations."""

from __future__ import annotations

from typing import Optional

import numpy as np


def as_optional_float(value: object) -> Optional[float]:
    """Best-effort conversion for scalar numeric values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def append_bounded(history: list[float], value: float, max_len: int) -> None:
    """Append to history while keeping bounded memory usage."""
    history.append(value)
    overflow = len(history) - max_len
    if overflow > 0:
        del history[:overflow]


def as_optional_array(value: object) -> Optional[np.ndarray]:
    """Best-effort conversion to ndarray with empty-array guard."""
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr
