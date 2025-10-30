#!/usr/bin/env python3
"""
Notification utilities for training events.
"""

from typing import Optional

try:
    from ztb.utils.notify.discord import DiscordNotifier
except ImportError:
    # Fallback if discord notifier is not available
    class DiscordNotifier:
        def __init__(self, webhook_url=None):
            self.webhook_url = webhook_url

        def send_notification(self, *args, **kwargs):
            pass  # No-op


def get_notifier(webhook_url: Optional[str] = None, offline_mode: bool = False):
    """
    Get appropriate notifier based on configuration.

    Args:
        webhook_url: Discord webhook URL
        offline_mode: Whether to disable notifications

    Returns:
        Notifier instance
    """
    if offline_mode or webhook_url is None:
        return DiscordNotifier(webhook_url=None)  # Explicitly disable

    return DiscordNotifier(webhook_url=webhook_url)
