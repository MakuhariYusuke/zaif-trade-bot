#!/usr/bin/env python3
"""
Alert notifier for Zaif Trade Bot.

Reads watcher JSONL and forwards WARN/FAIL alerts to a generic webhook.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def load_alerts(
    jsonl_path: Path, since_seconds: int, min_level: str
) -> List[Dict[str, Any]]:
    """Load alerts from JSONL file."""
    if not jsonl_path.exists():
        print(f"JSONL file not found: {jsonl_path}", file=sys.stderr)
        return []

    since_time = datetime.now() - timedelta(seconds=since_seconds)
    alerts = []
    level_order = {"INFO": 0, "WARN": 1, "ERROR": 2, "CRITICAL": 3}

    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    alert = json.loads(line.strip())
                    alert_time = datetime.fromisoformat(alert["timestamp"])
                    if alert_time >= since_time and level_order.get(
                        alert.get("level", "INFO"), 0
                    ) >= level_order.get(min_level, 1):
                        alerts.append(alert)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading JSONL: {e}", file=sys.stderr)

    return alerts


def send_webhook(
    webhook_url: str,
    title: str,
    correlation_id: str,
    alerts: List[Dict[str, Any]],
    platform: str = "slack",
) -> bool:
    """Send alerts to webhook. Supports Slack and Discord."""
    if not HAS_REQUESTS:
        print("requests not available, cannot send webhook", file=sys.stderr)
        return False

    if platform.lower() == "discord":
        # Discord webhook format
        content_lines = [f"**{title}**", f"Correlation ID: `{correlation_id}`", ""]
        for alert in alerts[:10]:  # Limit to 10 alerts to avoid message length limits
            level = alert.get("level", "INFO")
            message = alert.get("message", "No message")
            timestamp = alert.get("timestamp", "Unknown time")
            content_lines.append(f"• **{level}**: {message} ({timestamp})")

        if len(alerts) > 10:
            content_lines.append(f"... and {len(alerts) - 10} more alerts")

        payload = {
            "content": "\n".join(content_lines),
            "username": os.getenv("ZTB_DISCORD_USERNAME", "Zaif Trade Bot"),
            "avatar_url": os.getenv("ZTB_DISCORD_AVATAR_URL", ""),
        }
    else:
        # Slack webhook format (default)
        payload = {"title": title, "correlation_id": correlation_id, "alerts": alerts}  # type: ignore[dict-item]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 429:
                # Rate limited, retry with exponential backoff
                retry_after = response.headers.get("Retry-After", "1")
                try:
                    wait_seconds = int(retry_after)
                except ValueError:
                    wait_seconds = 1
                wait_seconds = min(wait_seconds, 60)  # Cap at 60 seconds
                print(
                    f"Rate limited, retrying in {wait_seconds} seconds...",
                    file=sys.stderr,
                )
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(
                    f"Webhook send failed after {max_retries} attempts: {e}",
                    file=sys.stderr,
                )
                return False
            else:
                wait_seconds = 2**attempt  # Exponential backoff
                print(
                    f"Webhook send failed, retrying in {wait_seconds} seconds: {e}",
                    file=sys.stderr,
                )
                time.sleep(wait_seconds)

    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Notify alerts via webhook")
    parser.add_argument("--correlation-id", required=True, help="Correlation ID")
    parser.add_argument("--jsonl", type=Path, help="Path to JSONL file")
    parser.add_argument(
        "--since-seconds", type=int, default=600, help="Scan recent seconds"
    )
    parser.add_argument(
        "--level",
        choices=["WARN", "ERROR", "CRITICAL"],
        default="WARN",
        help="Minimum level",
    )
    parser.add_argument(
        "--platform",
        choices=["slack", "discord"],
        default="slack",
        help="Webhook platform",
    )

    args = parser.parse_args()

    jsonl_path = (
        args.jsonl
        or Path("artifacts") / args.correlation_id / "logs" / "watch_log.jsonl"
    )
    alerts = load_alerts(jsonl_path, args.since_seconds, args.level)

    if not alerts:
        print("No alerts to send")
        return 0

    webhook_url = os.getenv("ZTB_ALERT_WEBHOOK")
    title = os.getenv(
        "ZTB_ALERT_TITLE", f"Zaif Trade Bot Alert - {args.correlation_id}"
    )
    platform = args.platform

    # Auto-detect platform if not specified
    if platform == "slack" and webhook_url and "discord.com" in webhook_url:
        platform = "discord"

    if webhook_url:
        success = send_webhook(
            webhook_url, title, args.correlation_id, alerts, platform
        )
        if success:
            print(f"Sent {len(alerts)} alerts to {platform} webhook")
        else:
            return 1
    else:
        # Dry-run
        payload = {
            "title": title,
            "correlation_id": args.correlation_id,
            "alerts": alerts,
        }
        print("DRY-RUN PAYLOAD:")
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
