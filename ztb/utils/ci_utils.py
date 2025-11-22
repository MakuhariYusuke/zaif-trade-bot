"""
ci_utils.py: CI/CD integration utilities.

Collects test coverage, execution time, failure reports during CI runs, and unifies notifications to Discord/Slack.

Usage:
    from ztb.utils.ci_utils import collect_ci_metrics, notify_ci_results

    metrics = collect_ci_metrics()
    notify_ci_results(metrics, "discord")
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import requests

try:
    from ztb.trading.environment.constants import BYTES_PER_MB
except (ImportError, OSError):
    BYTES_PER_MB = 1024 * 1024
from ztb.utils.errors import safe_operation
from ztb.utils.file_utils import safe_json_load
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

logger = logging.getLogger(__name__)


def collect_ci_metrics() -> Dict[str, Any]:
    """Collect CI metrics like coverage, execution time, failures"""
    process = psutil.Process()
    memory_info = process.memory_info()

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_mb": memory_info.rss / BYTES_PER_MB,
        "memory_percent": process.memory_percent(),
        "disk_usage": psutil.disk_usage("/").percent,
        "uptime_seconds": time.time() - psutil.boot_time(),
        "python_version": sys.version,
        "platform": sys.platform,
    }

    # Try to read coverage if available
    def collect_coverage():
        coverage_file = "coverage/coverage.json"
        if os.path.exists(coverage_file):
            coverage_data = safe_json_load(Path(coverage_file))
            metrics["coverage_percent"] = coverage_data.get("totals", {}).get(
                "percent_covered", 0
            )

    safe_operation(
        collect_coverage,
        default_result=None,
        logger=logger,
        context="Collecting coverage metrics",
    )

    return metrics


def notify_ci_results(
    metrics: Dict[str, Any], channel: str = "discord", webhook_url: Optional[str] = None
) -> None:
    """Notify CI results to specified channel (discord/slack)"""
    if channel.lower() == "discord":
        webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK")
        if not webhook_url:
            return

        embed = {
            "title": "CI Build Results",
            "description": f"Build completed at {metrics.get('timestamp', 'N/A')}",
            "color": 5763719,  # Green for success
            "fields": [
                {
                    "name": "CPU Usage",
                    "value": f"{metrics.get('cpu_percent', 0):.1f}%",
                    "inline": True,
                },
                {
                    "name": "Memory",
                    "value": f"{metrics.get('memory_mb', 0):.1f} MB",
                    "inline": True,
                },
                {
                    "name": "Coverage",
                    "value": (
                        f"{metrics.get('coverage_percent', 0):.1f}%"
                        if "coverage_percent" in metrics
                        else "N/A"
                    ),
                    "inline": True,
                },
                {
                    "name": "Platform",
                    "value": metrics.get("platform", "N/A"),
                    "inline": True,
                },
            ],
        }

        payload = {"content": "🚀 CI Build Completed Successfully", "embeds": [embed]}

        try:
            requests.post(webhook_url, json=payload, timeout=10)
        except Exception as e:
            logger.error("Failed to send Discord notification: %s", e)

    elif channel.lower() == "slack":
        # Slack notification (placeholder)
        pass
