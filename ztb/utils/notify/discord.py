"""
Notification system for trading bot alerts and monitoring.

Supports Discord webhooks for real-time notifications.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    import requests
    from requests.exceptions import RequestException, Timeout
except ImportError:
    # Fallback for test environments
    requests = None
    RequestException = Exception
    Timeout = Exception

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """
    Discord webhook notifier for trading alerts.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 10.0,
    ):
        self.webhook_url = webhook_url
        self.enabled = webhook_url is not None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    def send_notification(
        self,
        title: str,
        message: str,
        color: str = "info",
        fields: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send notification to Discord.

        Args:
            title: Notification title
            message: Main message content
            color: Color theme ('success', 'error', 'warning', 'info')
            fields: Additional fields to include

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Discord notification skipped (disabled): {title}")
            return True

        # Map color names to Discord embed colors
        color_map = {
            "success": 0x00FF00,  # Green
            "error": 0xFF0000,  # Red
            "warning": 0xFFFF00,  # Yellow
            "info": 0x0099FF,  # Blue
        }
        embed_color = color_map.get(color, color_map["info"])

        # Build embed
        embed: Dict[str, Any] = {
            "title": title,
            "description": message,
            "color": embed_color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if fields:
            embed["fields"] = [
                {"name": key, "value": str(value), "inline": True}
                for key, value in fields.items()
            ]

        payload = {"embeds": [embed]}

        try:
            if not self.webhook_url:
                logger.error("Webhook URL not configured")
                return False

            payload = {"embeds": [embed]}

            # Retry logic with exponential backoff
            for attempt in range(self.max_retries + 1):
                try:
                    response = requests.post(
                        self.webhook_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    logger.info(f"Discord notification sent: {title}")
                    return True

                except Timeout as e:
                    error_msg = f"Timeout sending Discord notification (attempt {attempt + 1}/{self.max_retries + 1})"
                    logger.warning(f"{error_msg}: {e}")
                    if attempt < self.max_retries:
                        time.sleep(
                            self.retry_delay * (2**attempt)
                        )  # Exponential backoff
                        continue
                    else:
                        logger.error(
                            f"Failed to send Discord notification after {self.max_retries + 1} attempts: {e}"
                        )
                        return False

                except RequestException as e:
                    error_msg = f"Request error sending Discord notification (attempt {attempt + 1}/{self.max_retries + 1})"
                    logger.warning(f"{error_msg}: {e}")
                    if attempt < self.max_retries:
                        time.sleep(
                            self.retry_delay * (2**attempt)
                        )  # Exponential backoff
                        continue
                    else:
                        logger.error(
                            f"Failed to send Discord notification after {self.max_retries + 1} attempts: {e}"
                        )
                        return False

                except Exception as e:
                    # For unexpected errors, don't retry
                    logger.error(f"Unexpected error sending Discord notification: {e}")
                    return False

            return False

        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False


    def notify_job_completion(
        self, job_id: str, success: bool, metrics: Dict[str, Any]
    ) -> None:
        """Notify about job completion"""
        status = "✅ Success" if success else "❌ Failed"
        title = f"🔬 ML Job {status}"
        message = f"Job {job_id} completed"
    def notify_trading_signal(
        self, symbol: str, signal: str, confidence: float
    ) -> None:
        """Notify about trading signals"""
        title = f"📈 Trading Signal: {symbol}"
        message = f"Signal: {signal.upper()} (Confidence: {confidence:.2%})"
        color = (
            "success" if signal == "buy" else "error" if signal == "sell" else "info"
        )
        fields = {"Symbol": symbol, "Signal": signal, "Confidence": f"{confidence:.2%}"}

    def notify_drift_alert(
        self, drift_type: str, severity: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Notify about data or model drift detection"""
        title = f"🔄 Drift Alert: {drift_type.title()}"
        message = f"Drift detected with severity: {severity.upper()}"

        # Set color based on severity
        color_map = {
            "low": "warning",
            "medium": "warning",
            "high": "error",
            "critical": "error",
        }
        color = color_map.get(severity.lower(), "warning")

        self.send_notification(title, message, color, details)

    def notify_quality_gate_failure(
        self, gate_type: str, reason: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Notify about quality gate failures"""
        title = f"🚫 Quality Gate Failed: {gate_type}"
        message = f"Reason: {reason}"
        self.send_notification(title, message, "error", details)


# Backward compatibility
MockNotifier = DiscordNotifier

# Global notifier instance (initialized from environment)
_default_notifier: Optional[DiscordNotifier] = None


def send_notification(
    title: str,
    message: str,
    priority: str = "normal",
    fields: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Send a notification with the given parameters.

    Args:
        title: Notification title
        message: Notification message
        priority: Priority level (low, normal, high)
        fields: Additional fields

    Returns:
        True if sent successfully, False otherwise
    """
    color_map = {"low": "info", "normal": "info", "high": "warning"}
    color = color_map.get(priority, "info")
    return get_notifier().send_notification(title, message, color, fields)


def get_notifier() -> DiscordNotifier:
    """Get the global Discord notifier instance."""
    global _default_notifier
    if _default_notifier is None:
        _default_notifier = DiscordNotifier()
    return _default_notifier
