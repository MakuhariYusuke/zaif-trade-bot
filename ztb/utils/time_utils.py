"""Shared timestamp helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def current_iso_timestamp(*, utc: bool = False) -> str:
    """Return an ISO timestamp for lightweight metadata and logs."""
    now = datetime.now(timezone.utc) if utc else datetime.now()
    return now.isoformat()
