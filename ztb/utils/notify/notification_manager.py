"""
NotificationManager: Unified notification and error handling.

Provides a common interface for logging, error handling, and notifications.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ztb.utils.core.logger import LoggerManager


class NotificationManager:
    """Unified notification manager for errors, logs, and alerts"""

    def __init__(self, logger_manager: 'LoggerManager') -> None:
        super().__init__()
        self.logger_manager = logger_manager

    def handle_error(self, error: Exception, context: str = "") -> None:
        """Handle error with logging and notification"""
        msg = f"Error in {context}: {error}" if context else f"Error: {error}"
        self.logger_manager.send_custom_notification("🚨 Error", msg, color=0xFF0000)

    def send_log(self, level: str, msg: str, color: Optional[int] = None) -> None:
        """Send log notification with appropriate color"""
        if level == "error":
            color = color or 0xFF0000
        elif level == "warning":
            color = color or 0xFFA500
        elif level == "info":
            color = color or 0x00AAFF
        else:
            color = color or 0x808080

        self.logger_manager.send_custom_notification(f"📝 {level.capitalize()}", msg, color=color)

    def send_alert(self, title: str, message: str, color: int = 0xFFFF00) -> None:
        """Send alert notification"""
        self.logger_manager.send_custom_notification(f"⚠️ {title}", message, color=color)