"""
Alerting system for health monitoring and critical system notifications.
"""

import json
import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AlertPriority(Enum):
    """Alert priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HealthAlert:
    """Represents a health monitoring alert."""

    title: str
    message: str
    priority: AlertPriority
    source: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if self.timestamp is None:
            import time

            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "title": self.title,
            "message": self.message,
            "priority": self.priority.value,
            "source": self.source,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class AlertConfig:
    """Configuration for alert notifications."""

    # Email configuration
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: Optional[List[str]] = None

    # Slack configuration
    slack_webhook_url: Optional[str] = None

    # Alert thresholds
    alert_on_critical: bool = True
    alert_on_warning: bool = False
    alert_on_healthy: bool = False

    # Cooldown settings (seconds)
    critical_cooldown: int = 3600  # 1 hour
    warning_cooldown: int = 86400  # 24 hours

    def is_email_configured(self) -> bool:
        """Check if email alerting is configured."""
        return all(
            [
                self.smtp_server,
                self.smtp_username,
                self.smtp_password,
                self.email_from,
                self.email_to,
            ]
        )

    def is_slack_configured(self) -> bool:
        """Check if Slack alerting is configured."""
        return self.slack_webhook_url is not None


class AlertManager:
    """
    Manages health alerts and notifications.

    Supports multiple notification channels: email, Slack, and logging.
    """

    def __init__(self, config: AlertConfig):
        self.config = config
        self.last_alert_times: Dict[str, float] = {}
        self.alert_history: List[HealthAlert] = []

    def should_alert(self, alert: HealthAlert) -> bool:
        """Determine if an alert should be sent based on configuration and cooldown."""
        # Check priority thresholds
        if (
            alert.priority == AlertPriority.CRITICAL
            and not self.config.alert_on_critical
        ):
            return False
        if alert.priority == AlertPriority.MEDIUM and not self.config.alert_on_warning:
            return False
        if alert.priority == AlertPriority.LOW and not self.config.alert_on_healthy:
            return False

        # Check cooldown
        alert_key = f"{alert.source}:{alert.priority.value}"
        last_time = self.last_alert_times.get(alert_key, 0)

        import time

        current_time = time.time()

        cooldown_period = (
            self.config.critical_cooldown
            if alert.priority == AlertPriority.CRITICAL
            else self.config.warning_cooldown
        )

        if current_time - last_time < cooldown_period:
            logger.debug(f"Alert {alert_key} is in cooldown period")
            return False

        return True

    def send_alert(self, alert: HealthAlert) -> bool:
        """
        Send an alert through configured channels.

        Returns:
            True if alert was sent successfully
        """
        if not self.should_alert(alert):
            return False

        success = False

        # Try email notification
        if self.config.is_email_configured():
            try:
                self._send_email_alert(alert)
                success = True
                logger.info(f"Email alert sent for: {alert.title}")
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")

        # Try Slack notification
        if self.config.is_slack_configured():
            try:
                self._send_slack_alert(alert)
                success = True
                logger.info(f"Slack alert sent for: {alert.title}")
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")

        # Always log the alert
        self._log_alert(alert)

        if success:
            # Update last alert time
            alert_key = f"{alert.source}:{alert.priority.value}"
            self.last_alert_times[alert_key] = alert.timestamp or 0
            self.alert_history.append(alert)

        return success

    def _send_email_alert(self, alert: HealthAlert) -> None:
        """Send alert via email."""
        if not self.config.is_email_configured():
            return

        msg = MIMEMultipart()
        msg["From"] = self.config.email_from
        msg["To"] = ", ".join(self.config.email_to or [])
        msg["Subject"] = f"[{alert.priority.value.upper()}] {alert.title}"

        # Create HTML body
        html_body = f"""
        <html>
        <body>
            <h2>{alert.title}</h2>
            <p><strong>Priority:</strong> {alert.priority.value.upper()}</p>
            <p><strong>Source:</strong> {alert.source}</p>
            <p><strong>Message:</strong></p>
            <pre>{alert.message}</pre>
        """

        if alert.details:
            html_body += (
                "<h3>Details:</h3><pre>"
                + json.dumps(alert.details, indent=2)
                + "</pre>"
            )

        html_body += """
        </body>
        </html>
        """

        msg.attach(MIMEText(html_body, "html"))

        # Send email
        server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
        server.starttls()
        server.login(self.config.smtp_username or "", self.config.smtp_password or "")
        text = msg.as_string()
        server.sendmail(self.config.email_from or "", self.config.email_to or [], text)
        server.quit()

    def _send_slack_alert(self, alert: HealthAlert) -> None:
        """Send alert via Slack webhook."""
        if not self.config.is_slack_configured():
            return

        import requests

        # Create Slack message
        color = {
            AlertPriority.LOW: "good",
            AlertPriority.MEDIUM: "warning",
            AlertPriority.HIGH: "danger",
            AlertPriority.CRITICAL: "danger",
        }.get(alert.priority, "warning")

        slack_message = {
            "attachments": [
                {
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Priority",
                            "value": alert.priority.value.upper(),
                            "short": True,
                        },
                        {"title": "Source", "value": alert.source, "short": True},
                    ],
                }
            ]
        }

        if alert.details:
            slack_message["attachments"][0]["fields"].append(
                {
                    "title": "Details",
                    "value": json.dumps(alert.details, indent=2),
                    "short": False,
                }
            )

        response = requests.post(
            self.config.slack_webhook_url or "", json=slack_message, timeout=10
        )
        response.raise_for_status()

    def _log_alert(self, alert: HealthAlert) -> None:
        """Log alert to application logs."""
        log_message = (
            f"ALERT [{alert.priority.value.upper()}] {alert.title}: {alert.message}"
        )

        if alert.priority == AlertPriority.CRITICAL:
            logger.critical(log_message)
        elif alert.priority == AlertPriority.HIGH:
            logger.error(log_message)
        elif alert.priority == AlertPriority.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def get_alert_history(self, limit: int = 50) -> List[HealthAlert]:
        """Get recent alert history."""
        return self.alert_history[-limit:]

    def clear_history(self) -> None:
        """Clear alert history."""
        self.alert_history.clear()


def create_alert_manager(config_path: Optional[str] = None) -> AlertManager:
    """
    Create an AlertManager instance from configuration.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Configured AlertManager instance
    """
    config = AlertConfig()

    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            # Update config from file
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        except Exception as e:
            logger.error(f"Failed to load alert config from {config_path}: {e}")

    return AlertManager(config)


def create_health_alert_from_result(result: Dict[str, Any]) -> Optional[HealthAlert]:
    """
    Create a health alert from health check result.

    Args:
        result: Health check result dictionary

    Returns:
        HealthAlert if critical issues found, None otherwise
    """
    status = result.get("status", "unknown")

    if status == "critical":
        return HealthAlert(
            title="Critical System Health Issues",
            message="System health check detected critical issues that require immediate attention.",
            priority=AlertPriority.CRITICAL,
            source="health_check",
            details=result,
        )
    elif status == "warning":
        return HealthAlert(
            title="System Health Warnings",
            message="System health check detected warnings that should be reviewed.",
            priority=AlertPriority.HIGH,
            source="health_check",
            details=result,
        )

    return None
