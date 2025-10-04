#!/usr/bin/env python3
"""
Thin wrapper for starting training sessions.
Usage: python training_start.py [--correlation-id <id>] [--dry-run]
"""

import argparse
import os
import sys
from datetime import datetime, timezone

from ztb.utils.compat_wrapper import run_command_safely


def main() -> None:
    parser = argparse.ArgumentParser(description="Start training session")
    parser.add_argument("--correlation-id", help="Correlation ID for the session")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without executing"
    )
    args = parser.parse_args()

    # Generate correlation ID if not provided
    corr_id = args.correlation_id
    if not corr_id:
        corr_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"Starting training session with correlation ID: {corr_id}")

    # Set environment variable
    env = os.environ.copy()
    env["CORR"] = corr_id

    # Command to run
    cmd = ["make", "1m-start", f"CORR={corr_id}"]

    if args.dry_run:
        print(f"Dry run: {' '.join(cmd)}")
        return

    print(f"Executing: {' '.join(cmd)}")
    result = run_command_safely(cmd, env=env, cwd=os.path.dirname(__file__))

    if result["success"]:
        print("Training started successfully")
        print(f"Monitor with: make 1m-watch CORR={corr_id}")
    else:
        print(f"Failed to start training (exit code: {result['returncode']})")
        sys.exit(result["returncode"])


if __name__ == "__main__":
    main()
