"""
Notification system for trading bot alerts and monitoring.

Supports Discord webhooks for real-time notifications.
"""
import logging
from typing import Dict, Any, Optional
import requests
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class DiscordNotifier:
    """
    Discord webhook notifier for trading alerts.
    """

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.enabled = webhook_url is not None

    def send_notification(self, title: str, message: str, color: str = "info",
                         fields: Optional[Dict[str, Any]] = None) -> bool:
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
            'success': 0x00ff00,  # Green
            'error': 0xff0000,    # Red
            'warning': 0xffff00,  # Yellow
            'info': 0x0099ff      # Blue
        }
        embed_color = color_map.get(color, color_map['info'])

        # Build embed
        embed: Dict[str, Any] = {
            "title": title,
            "description": message,
            "color": embed_color,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if fields:
            embed["fields"] = [
                {"name": key, "value": str(value), "inline": True}
                for key, value in fields.items()
            ]

        payload = {
            "embeds": [embed]
        }

        try:
            if not self.webhook_url:
                logger.error("Webhook URL not configured")
                return False
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"Discord notification sent: {title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    def notify_data_pipeline_status(self, status: str, details: Dict[str, Any]) -> None:
        """Notify about data pipeline status"""
        title = f"📊 Data Pipeline {status.title()}"
        message = f"Data acquisition and integrity check completed"
        self.send_notification(title, message, "success" if status == "success" else "error", details)

    def notify_job_completion(self, job_id: str, success: bool, metrics: Dict[str, Any]):
        """Notify about job completion"""
        status = "✅ Success" if success else "❌ Failed"
        title = f"🔬 ML Job {status}"
        message = f"Job {job_id} completed"
        color = "success" if success else "error"
        self.send_notification(title, message, color, metrics)

    def notify_risk_alert(self, alert_type: str, details: Dict[str, Any]):
        """Notify about risk management alerts"""
        title = f"⚠️ Risk Alert: {alert_type}"
        message = "Risk management threshold exceeded"
        self.send_notification(title, message, "warning", details)

    def notify_trading_signal(self, symbol: str, signal: str, confidence: float):
        """Notify about trading signals"""
        title = f"📈 Trading Signal: {symbol}"
        message = f"Signal: {signal.upper()} (Confidence: {confidence:.2%})"
        color = "success" if signal == "buy" else "error" if signal == "sell" else "info"
        fields = {"Symbol": symbol, "Signal": signal, "Confidence": f"{confidence:.2%}"}
        self.send_notification(title, message, color, fields)

    def notify_drift_alert(self, drift_type: str, severity: str, details: Dict[str, Any]):
        """Notify about data or model drift detection"""
        title = f"🔄 Drift Alert: {drift_type.title()}"
        message = f"Drift detected with severity: {severity.upper()}"

        # Set color based on severity
        color_map = {
            'low': 'info',
            'medium': 'warning',
            'high': 'error',
            'critical': 'error'
        }
        color = color_map.get(severity.lower(), 'warning')

        self.send_notification(title, message, color, details)

    def notify_quality_gate_failure(self, gate_type: str, reason: str, details: Dict[str, Any]):
        """Notify about quality gate failures"""
        title = f"🚫 Quality Gate Failed: {gate_type}"
        message = f"Reason: {reason}"
        self.send_notification(title, message, "error", details)

# Backward compatibility
MockNotifier = DiscordNotifier

# Global notifier instance (initialized from environment)
_default_notifier: Optional[DiscordNotifier] = None

def get_notifier() -> DiscordNotifier:
    """Get global notifier instance"""
    global _default_notifier
    if _default_notifier is None:
        import os
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        _default_notifier = DiscordNotifier(webhook_url)
    return _default_notifier

def send_notification(title: str, message: str, priority: str = "normal",
                     fields: Optional[Dict[str, Any]] = None) -> bool:
    """
    Send notification using global notifier.

    Args:
        title: Notification title
        message: Main message content
        priority: Priority level ('low', 'normal', 'high')
        fields: Additional fields

    Returns:
        True if sent successfully, False otherwise
    """
    color_map = {
        'low': 'info',
        'normal': 'info',
        'high': 'warning'
    }
    color = color_map.get(priority, 'info')
    return get_notifier().send_notification(title, message, color, fields)