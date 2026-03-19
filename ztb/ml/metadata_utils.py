"""Shared metadata timestamp helpers for model/report artifacts."""

from __future__ import annotations

from datetime import datetime, timezone


def current_iso_timestamp(*, utc: bool = False) -> str:
    """Return an ISO timestamp for lightweight artifact metadata.

    Default keeps the existing local-naive contract used across legacy
    training scripts. Callers can opt into explicit UTC when needed.
    """
    now = datetime.now(timezone.utc) if utc else datetime.now()
    return now.isoformat()
