#!/usr/bin/env python3
"""
Test script for the alert system.

This script tests email and Slack notification functionality.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.ops.alerts.alert_system import (
    AlertConfig,
    AlertManager,
    AlertPriority,
    HealthAlert,
)


def test_email_alert():
    """Test email alert functionality."""
    print("Testing email alert functionality...")

    # Create test alert
    alert = HealthAlert(
        title="Test Email Alert",
        message="This is a test email alert from the health monitoring system.",
        priority=AlertPriority.MEDIUM,
        source="test_system",
        details={"test": True, "component": "email"},
    )

    # Create alert config (you'll need to set these environment variables)
    config = AlertConfig(
        smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        smtp_username=os.getenv("EMAIL_USERNAME"),
        smtp_password=os.getenv("EMAIL_PASSWORD"),
        email_from=os.getenv("EMAIL_FROM"),
        email_to=os.getenv("EMAIL_TO", "").split(",")
        if os.getenv("EMAIL_TO")
        else None,
        slack_webhook_url=None,  # Disable Slack for this test
    )

    # Create alert manager
    manager = AlertManager(config)

    try:
        # Send test alert
        success = manager.send_alert(alert)
        if success:
            print("✅ Email alert sent successfully!")
        else:
            print("❌ Email alert failed to send")
    except Exception as e:
        print(f"❌ Email alert error: {e}")


def test_slack_alert():
    """Test Slack alert functionality."""
    print("Testing Slack alert functionality...")

    # Create test alert
    alert = HealthAlert(
        title="Test Slack Alert",
        message="This is a test Slack alert from the health monitoring system.",
        priority=AlertPriority.HIGH,
        source="test_system",
        details={"test": True, "component": "slack"},
    )

    # Create alert config
    config = AlertConfig(
        smtp_server=None,
        slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),  # Disable email for this test
    )

    # Create alert manager
    manager = AlertManager(config)

    try:
        # Send test alert
        success = manager.send_alert(alert)
        if success:
            print("✅ Slack alert sent successfully!")
        else:
            print("❌ Slack alert failed to send")
    except Exception as e:
        print(f"❌ Slack alert error: {e}")


def test_alert_priority_filtering():
    """Test alert priority filtering."""
    print("Testing alert priority filtering...")

    config = AlertConfig(
        alert_on_critical=True,
        alert_on_warning=True,
        alert_on_healthy=False,  # Only send critical and warning alerts
    )

    manager = AlertManager(config)

    # Test different priority alerts
    alerts = [
        HealthAlert("Low Priority", "Low test", AlertPriority.LOW, "test"),
        HealthAlert("Medium Priority", "Medium test", AlertPriority.MEDIUM, "test"),
        HealthAlert("High Priority", "High test", AlertPriority.HIGH, "test"),
        HealthAlert(
            "Critical Priority", "Critical test", AlertPriority.CRITICAL, "test"
        ),
    ]

    for alert in alerts:
        should_send = manager.should_alert(alert)
        status = "✅ Would send" if should_send else "⏭️  Would skip"
        print(f"{status} {alert.priority.value} priority alert")


def main():
    """Run all alert system tests."""
    print("🚨 Alert System Test Suite")
    print("=" * 50)

    # Test priority filtering (doesn't require external services)
    test_alert_priority_filtering()
    print()

    # Test email if credentials are available
    if os.getenv("EMAIL_USERNAME") and os.getenv("EMAIL_PASSWORD"):
        test_email_alert()
    else:
        print(
            "⏭️  Skipping email test (no EMAIL_USERNAME/EMAIL_PASSWORD environment variables)"
        )
    print()

    # Test Slack if webhook URL is available
    if os.getenv("SLACK_WEBHOOK_URL"):
        test_slack_alert()
    else:
        print("⏭️  Skipping Slack test (no SLACK_WEBHOOK_URL environment variable)")
    print()

    print("Test suite completed!")
    print("\nTo test email notifications, set these environment variables:")
    print("  EMAIL_USERNAME=your_email@gmail.com")
    print("  EMAIL_PASSWORD=your_app_password")
    print("  EMAIL_FROM=your_email@gmail.com")
    print("  EMAIL_TO=recipient1@example.com,recipient2@example.com")
    print("  SMTP_SERVER=smtp.gmail.com")
    print("  SMTP_PORT=587")
    print()
    print("To test Slack notifications, set this environment variable:")
    print("  SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...")

    print(
        "\nNote: For Gmail, you'll need an 'App Password' instead of your regular password."
    )
    print(
        "Enable 2FA on your Google account and generate an App Password in security settings."
    )


if __name__ == "__main__":
    main()
