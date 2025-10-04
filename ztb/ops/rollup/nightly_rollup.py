#!/usr/bin/env python3
"""
Nightly rollup wrapper for Zaif Trade Bot.

Runs index_sessions, aggregate_trends, make_trends_md, and budget_rollup in sequence.
"""

import argparse
import sys

from ztb.utils.compat_wrapper import run_command_safely


def run_command(command: list[str]) -> int:
    """Run command and return exit code."""
    try:
        cmd_str = " ".join(command)
        print(f"Running: {cmd_str}")
        result = run_command_safely(cmd_str)
        if result["stdout"]:
            print(result["stdout"].rstrip())
        if result["stderr"]:
            print(result["stderr"].rstrip(), file=sys.stderr)
        return int(result["returncode"])
    except Exception as e:
        print(f"Command failed: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run nightly rollup operations")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without executing"
    )
    args = parser.parse_args()

    commands = [
        ["python", "ztb/scripts/index_sessions.py"],
        ["python", "ztb/scripts/aggregate_trends.py"],
        ["python", "ztb/scripts/make_trends_md.py"],
        ["python", "ztb/scripts/budget_rollup.py"],
    ]

    if args.dry_run:
        print("Dry run - would execute:")
        for cmd in commands:
            print(f"  {' '.join(cmd)}")
        return 0

    for cmd in commands:
        exit_code = run_command(cmd)
        if exit_code != 0:
            print(f"Command failed with exit code {exit_code}, stopping rollup")
            return exit_code

    print("Nightly rollup completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
