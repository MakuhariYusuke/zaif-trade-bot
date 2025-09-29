#!/usr/bin/env python3
"""
Thin wrapper for checking system status.
Usage: python status_check.py [--correlation-id <id>] [--output <file>]
"""

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Check system status")
    parser.add_argument(
        "--correlation-id", required=True, help="Correlation ID for the session"
    )
    parser.add_argument("--output", help="Output file for status report")
    args = parser.parse_args()

    corr_id = args.correlation_id
    print(f"Checking status for session: {corr_id}")

    # Command to run status snapshot
    cmd = [
        sys.executable,
        "ztb/ztb/ztb/scripts/status_snapshot.py",
        "--correlation-id",
        corr_id,
    ]
    if args.output:
        cmd.extend(["--output", args.output])

    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))

    if result.returncode == 0:
        print("Status check completed successfully")
    else:
        print(f"Status check failed (exit code: {result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
