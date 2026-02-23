"""
Shared JSONL alert loading utilities for ops notifiers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from ztb.io.jsonl import iter_jsonl_objects

LEVEL_ORDER: dict[str, int] = {"INFO": 0, "WARN": 1, "ERROR": 2, "CRITICAL": 3, "FAIL": 4}


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_alerts_from_jsonl(
    jsonl_path: Path,
    since_seconds: int,
    min_level: str,
) -> list[dict[str, object]]:
    """Load alerts from JSONL with tolerant parsing."""
    since_time = datetime.now(timezone.utc) - timedelta(seconds=max(since_seconds, 0))
    min_rank = LEVEL_ORDER.get(min_level.upper(), LEVEL_ORDER["WARN"])
    alerts: list[dict[str, object]] = []

    for alert in iter_jsonl_objects(jsonl_path, warn_malformed=False):
        level = str(alert.get("level", "INFO")).upper()
        if LEVEL_ORDER.get(level, LEVEL_ORDER["INFO"]) < min_rank:
            continue

        alert_time = _parse_timestamp(alert.get("timestamp"))
        if alert_time is None or alert_time < since_time:
            continue

        alerts.append(alert)

    return alerts
