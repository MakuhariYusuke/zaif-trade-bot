#!/usr/bin/env python3
"""
Disk health monitor for Zaif Trade Bot.

Checks disk space, inode usage, and optional I/O latency, emits alerts in JSON format.
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def get_disk_usage(path: Path) -> Dict[str, Any]:
    """Get disk usage stats."""
    try:
        stat = shutil.disk_usage(path)
        return {
            "total": stat.total,
            "used": stat.used,
            "free": stat.free,
            "free_gb": stat.free / (1024**3),
        }
    except Exception as e:
        return {"error": str(e)}


def get_inode_usage(path: Path) -> Optional[Dict[str, Any]]:
    """Get inode usage if available."""
    try:
        statvfs = os.statvfs(path)
        total_inodes = statvfs.f_files
        free_inodes = statvfs.f_favail
        used_inodes = total_inodes - free_inodes
        usage_pct = (used_inodes / total_inodes) * 100 if total_inodes > 0 else 0
        return {
            "total": total_inodes,
            "used": used_inodes,
            "free": free_inodes,
            "usage_pct": usage_pct,
        }
    except AttributeError:
        # Windows doesn't have statvfs
        return None
    except Exception as e:
        return {"error": str(e)}


def measure_io_latency(path: Path, test_size: int = 1024) -> Optional[float]:
    """Measure I/O latency with small read/write test."""
    try:
        test_file = path / ".disk_health_test"
        data = b"x" * test_size

        # Write test
        start = time.time()
        with open(test_file, "wb") as f:
            f.write(data)
        os.fsync(f.fileno())  # Force write to disk

        # Read test
        with open(test_file, "rb") as f:
            _ = f.read()

        end = time.time()
        test_file.unlink(missing_ok=True)

        return (end - start) * 1000  # ms
    except Exception as e:
        print(f"I/O test failed: {e}", file=sys.stderr)
        return None


def check_health(
    path: Path, min_free_gb: float, max_inode_usage: float, check_io: bool = False
) -> list:
    """Check disk health and return alerts."""
    alerts = []

    # Disk space
    disk = get_disk_usage(path)
    if "error" in disk:
        alerts.append(
            {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "alert_type": "disk_check_failed",
                "message": f"Failed to check disk usage: {disk['error']}",
                "path": str(path),
            }
        )
    elif disk["free_gb"] < min_free_gb:
        alerts.append(
            {
                "timestamp": datetime.now().isoformat(),
                "level": "FAIL" if disk["free_gb"] < min_free_gb / 2 else "WARN",
                "alert_type": "low_disk_space",
                "message": f"Free disk space: {disk['free_gb']:.1f}GB < {min_free_gb}GB",
                "path": str(path),
                "free_gb": disk["free_gb"],
            }
        )

    # Inode usage
    inodes = get_inode_usage(path)
    if inodes and "error" not in inodes:
        if inodes["usage_pct"] > max_inode_usage:
            alerts.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "FAIL" if inodes["usage_pct"] > 95 else "WARN",
                    "alert_type": "high_inode_usage",
                    "message": f"Inode usage: {inodes['usage_pct']:.1f}% > {max_inode_usage}%",
                    "path": str(path),
                    "inode_usage_pct": inodes["usage_pct"],
                }
            )

    # I/O latency (optional)
    if check_io:
        latency = measure_io_latency(path)
        if latency is not None and latency > 100:  # >100ms is slow
            alerts.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "WARN",
                    "alert_type": "slow_io",
                    "message": f"I/O latency: {latency:.1f}ms > 100ms",
                    "path": str(path),
                    "latency_ms": latency,
                }
            )

    return alerts


def write_alerts(alerts: list, log_path: Path):
    """Write alerts to JSONL file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        for alert in alerts:
            f.write(json.dumps(alert) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Check disk health for Zaif Trade Bot")
    parser.add_argument(
        "--correlation-id", required=True, help="Session correlation ID"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("artifacts"),
        help="Path to check (default: artifacts)",
    )
    parser.add_argument(
        "--min-free-gb", type=float, default=10.0, help="Minimum free disk space in GB"
    )
    parser.add_argument(
        "--max-inode-usage",
        type=float,
        default=90.0,
        help="Maximum inode usage percentage",
    )
    parser.add_argument(
        "--check-io", action="store_true", help="Also check I/O latency"
    )

    args = parser.parse_args()

    alerts = check_health(
        args.path, args.min_free_gb, args.max_inode_usage, args.check_io
    )

    if alerts:
        # Print to stdout
        for alert in alerts:
            print(json.dumps(alert))

        # Write to log
        log_path = Path("artifacts") / args.correlation_id / "logs" / "ops_alerts.jsonl"
        write_alerts(alerts, log_path)

        # Exit with error if any FAIL
        if any(a["level"] == "FAIL" for a in alerts):
            sys.exit(1)
    else:
        print("Disk health OK")


if __name__ == "__main__":
    main()
