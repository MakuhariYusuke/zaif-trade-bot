"""Hour-based rule helpers.

time-of-day 依存の分岐を 1 箇所に集約する。
skip_gate hour offsets、sell-hour boost、hard skip 時刻判定で再利用する。
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import datetime, timezone


def current_utc_hour() -> int:
    """現在の UTC hour を返す."""
    return datetime.now(timezone.utc).hour


def utc_hour_from_timestamp(timestamp: float) -> int:
    """UNIX timestamp から UTC hour を返す."""
    return time.gmtime(timestamp).tm_hour


def resolve_hour_float(
    hour_values: Mapping[int, float] | None,
    utc_hour: int,
    *,
    default: float = 0.0,
) -> float:
    """hour->float マップから値を解決する."""
    if not hour_values:
        return default
    return float(hour_values.get(utc_hour, default))


def resolve_optional_hour_float(
    hour_values: Mapping[int, float] | None,
    utc_hour: int,
) -> float | None:
    """hour->float マップから値を解決し、未設定時は None を返す."""
    if not hour_values:
        return None
    value = hour_values.get(utc_hour)
    if value is None:
        return None
    return float(value)
