"""
Notification system — re-export from canonical ztb.utils.notify.

重複排除: 実装は ztb.utils.notify.discord に一元化。
このモジュールは後方互換のための re-export のみ提供する。
"""

from ztb.utils.notify.discord import (  # noqa: F401
    DiscordNotifier,
    MockNotifier,
    get_notifier,
    send_notification,
)
