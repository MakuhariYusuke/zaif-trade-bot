#!/usr/bin/env python3
"""
Thin wrapper for starting monitoring processes.
Usage: python monitoring_start.py [--correlation-id <id>] [--dry-run]
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Start monitoring processes")
    parser.add_argument(
        "--correlation-id", required=True, help="Correlation ID for the session"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without executing"
    )
    args = parser.parse_args()

    corr_id = args.correlation_id
    print(f"Starting monitoring for session: {corr_id}")

    # Command to run monitoring launcher
    cmd = [
        sys.executable,
        "ztb/ztb/ztb/scripts/launch_monitoring.py",
        "--correlation-id",
        corr_id,
    ]

    if args.dry_run:
        print(f"Dry run: {' '.join(cmd)}")
        return

    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))

    if result.returncode == 0:
        print("Monitoring started successfully")
    else:
        print(f"Failed to start monitoring (exit code: {result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
