"""
Lightweight notifications interface used by monitoring module.
This file intentionally provides a minimal, typed implementation so that
static type checking (mypy) does not raise import-not-found errors.

At runtime the project may use ztb.ops.alerts.notifications or other
pluggable notification systems; this module is a safe fallback.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def send_notification(title: str, message: str, priority: str = "normal", fields: Optional[Dict[str, Any]] = None) -> bool:
    """Send a lightweight notification. Returns True on success.

    This implementation logs the notification and returns True so callers
    can assume notifications are available at runtime. Replace with a
    real implementation (Discord/Slack) if needed.
    """
    try:
        logger.info("Notification: %s | %s | %s", title, priority, message)
        return True
    except Exception:
        logger.exception("Failed to send notification")
        return False
