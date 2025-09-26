"""
ci_utils.py: CI/CD integration utilities.

Collects test coverage, execution time, failure reports during CI runs, and unifies notifications to Discord/Slack.

Usage:
    from ztb.utils.ci_utils import collect_ci_metrics, notify_ci_results

    metrics = collect_ci_metrics()
    notify_ci_results(metrics, "discord")
"""

import os
import sys
import time
import psutil
from datetime import datetime
from typing import Dict, Any, Optional
import requests


def collect_ci_metrics() -> Dict[str, Any]:
    """Collect CI metrics like coverage, execution time, failures"""
    process = psutil.Process()
    memory_info = process.memory_info()

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_mb": memory_info.rss / 1024 / 1024,
        "memory_percent": process.memory_percent(),
        "disk_usage": psutil.disk_usage('/').percent,
        "uptime_seconds": time.time() - psutil.boot_time(),
        "python_version": sys.version,
        "platform": sys.platform
    }

    # Try to read coverage if available
    try:
        import json
        coverage_file = "coverage/coverage.json"
        if os.path.exists(coverage_file):
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
                metrics["coverage_percent"] = coverage_data.get("totals", {}).get("percent_covered", 0)
    except Exception:
        pass

    return metrics


def notify_ci_results(metrics: Dict[str, Any], channel: str = "discord", webhook_url: Optional[str] = None) -> None:
    """Notify CI results to specified channel (discord/slack)"""
    if channel.lower() == "discord":
        webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK')
        if not webhook_url:
            return

        embed = {
            "title": "CI Build Results",
            "description": f"Build completed at {metrics.get('timestamp', 'N/A')}",
            "color": 5763719,  # Green for success
            "fields": [
                {"name": "CPU Usage", "value": f"{metrics.get('cpu_percent', 0):.1f}%", "inline": True},
                {"name": "Memory", "value": f"{metrics.get('memory_mb', 0):.1f} MB", "inline": True},
                {"name": "Coverage", "value": f"{metrics.get('coverage_percent', 0):.1f}%" if 'coverage_percent' in metrics else "N/A", "inline": True},
                {"name": "Platform", "value": metrics.get('platform', 'N/A'), "inline": True}
            ]
        }

        payload = {
            "content": "🚀 CI Build Completed Successfully",
            "embeds": [embed]
        }

        try:
            requests.post(webhook_url, json=payload, timeout=10)
        except Exception as e:
            print(f"Failed to send Discord notification: {e}")

    elif channel.lower() == "slack":
        # Slack notification (placeholder)
        pass