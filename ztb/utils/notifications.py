#!/usr/bin/env python3
"""
Notification utilities — re-export from canonical ztb.utils.notify.

重複排除: 実装は ztb.utils.notify.discord に一元化。
このモジュールは後方互換のための re-export のみ提供する。
"""

from ztb.utils.notify.discord import DiscordNotifier  # noqa: F401
from ztb.utils.notify.discord import get_notifier  # noqa: F401
