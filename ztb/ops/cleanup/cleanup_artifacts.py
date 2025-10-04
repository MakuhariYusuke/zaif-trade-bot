#!/usr/bin/env python3
"""
Thin wrapper for cleaning up old artifacts.
Usage: python cleanup_artifacts.py [--dry-run] [--days <n>]
"""

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up old artifacts")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without doing it",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Keep artifacts for this many days (default: 30)",
    )
    args = parser.parse_args()

    print(f"Cleaning up artifacts older than {args.days} days")

    # Command to run artifacts janitor
    cmd = [
        sys.executable,
        "ztb/ztb/ztb/scripts/artifacts_janitor.py",
        "--days",
        str(args.days),
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))

    if result.returncode == 0:
        action = "previewed" if args.dry_run else "completed"
        print(f"Artifact cleanup {action} successfully")
    else:
        print(f"Artifact cleanup failed (exit code: {result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
