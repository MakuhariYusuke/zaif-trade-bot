"""
Datetime utilities for multi-timeframe feature generation.

Provides a safe, Python-level fallback for timestamp parsing to avoid
Windows SIGINT issues in pandas' C extensions.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

_DEFAULT_SAFE = "1" if os.name == "nt" else "0"
_SAFE_DATETIME = os.getenv("ZTB_SAFE_DATETIME", _DEFAULT_SAFE) == "1"

def safe_to_datetime_series(
    series: pd.Series,
    errors: str = "coerce",
    utc: bool | None = None,
) -> pd.Series:
    """Convert a Series to datetime with a safe Python fallback.

    When ZTB_SAFE_DATETIME=1, parse strings using datetime.fromisoformat
    to avoid pandas' C-level array_strptime path.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    if not _SAFE_DATETIME:
        return pd.to_datetime(series, errors=errors, utc=utc)

    parsed = _parse_datetime_values(series, errors=errors, utc=utc)
    dt_index = pd.DatetimeIndex(parsed)

    if utc:
        if dt_index.tz is None:
            dt_index = dt_index.tz_localize("UTC")
        else:
            dt_index = dt_index.tz_convert("UTC")

    return pd.Series(dt_index, index=series.index, name=series.name)

def _parse_datetime_values(
    values: Iterable[object],
    errors: str = "coerce",
    utc: bool | None = None,
) -> list[object]:
    parsed: list[object] = []
    has_tz = False

    for value in values:
        if value is None or pd.isna(value):
            parsed.append(pd.NaT)
            continue

        dt = _parse_single_value(value, errors=errors, utc=utc)
        if isinstance(dt, datetime) and dt.tzinfo is not None:
            has_tz = True
        parsed.append(dt)

    # If any timezone-aware values are present, normalize everything to UTC.
    if utc is None and has_tz:
        utc = True
        normalized = []
        for value in parsed:
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    normalized.append(value.replace(tzinfo=timezone.utc))
                else:
                    normalized.append(value.astimezone(timezone.utc))
            else:
                normalized.append(value)
        parsed = normalized

    return parsed

def _parse_single_value(
    value: object,
    errors: str = "coerce",
    utc: bool | None = None,
) -> object:
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    elif isinstance(value, (np.integer, np.floating, int, float)):
        dt = _parse_epoch_number(value)
        if dt is None:
            return _handle_parse_error(value, errors)
    else:
        text = str(value).strip()
        if not text:
            return _handle_parse_error(value, errors)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except Exception:
            return _handle_parse_error(value, errors)

    if isinstance(dt, datetime) and utc:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

    return dt

def _parse_epoch_number(value: object) -> datetime | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None

    abs_val = abs(val)
    if abs_val >= 1e15:
        # Likely nanoseconds
        return datetime.fromtimestamp(val / 1_000_000_000, tz=timezone.utc)
    if abs_val >= 1e12:
        # Likely milliseconds
        return datetime.fromtimestamp(val / 1_000, tz=timezone.utc)
    if abs_val >= 1e9:
        # Likely seconds
        return datetime.fromtimestamp(val, tz=timezone.utc)
    return None

def _handle_parse_error(value: object, errors: str) -> object:
    if errors == "raise":
        raise ValueError(f"Failed to parse datetime value: {value!r}")
    return pd.NaT
