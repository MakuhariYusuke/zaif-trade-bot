#!/usr/bin/env python3
"""
Cron-ish command runner for Zaif Trade Bot.

Runs commands at fixed intervals with jitter, stops on kill-file.
"""

import argparse
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Optional


def run_command(command: str) -> int:
    """Run the command and return exit code."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        return result.returncode
    except Exception as e:
        print(f"Command failed: {e}", file=sys.stderr)
        return 1


def calculate_next_run(interval_sec: int, jitter_sec: int) -> float:
    """Calculate next run time with jitter."""
    jitter = random.uniform(-jitter_sec, jitter_sec)
    return time.time() + interval_sec + jitter


def format_eta(seconds: float) -> str:
    """Format ETA as human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def perform_catchup(args: argparse.Namespace, kill_file: Path) -> None:
    """Perform catchup runs for missed intervals."""
    # Assume we track last run time in a file
    last_run_file = Path("cronish_last_run.txt")
    now = time.time()

    if last_run_file.exists():
        try:
            with open(last_run_file, "r") as f:
                last_run = float(f.read().strip())
        except (ValueError, OSError):
            last_run = now  # If can't read, assume just started
    else:
        last_run = now  # First run, no catchup needed

    # Calculate how many intervals have passed
    elapsed = now - last_run
    intervals_missed = int(elapsed // args.interval_sec)

    if intervals_missed > 0:
        catchup_count = min(intervals_missed, args.max_catchup)
        print(f"Performing catchup: {catchup_count} missed runs")

        for i in range(catchup_count):
            if kill_file.exists():
                print("Kill file detected during catchup, stopping...")
                sys.exit(0)

            print(f"Catchup {i + 1}/{catchup_count}: Running command...")
            exit_code = run_command(args.command)

            if exit_code != 0 and args.fail_fast:
                print(
                    f"Command failed during catchup with exit code {exit_code}, exiting..."
                )
                sys.exit(exit_code)

            if i < catchup_count - 1:  # Don't sleep after last catchup
                print(f"Catchup cooldown: {args.catchup_cooldown_sec}s")
                time.sleep(args.catchup_cooldown_sec)

    # Update last run time
    try:
        with open(last_run_file, "w") as f:
            f.write(str(now))
    except OSError:
        print("Warning: Could not update last run time", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run commands periodically with jitter"
    )
    parser.add_argument(
        "--interval-sec",
        type=int,
        required=True,
        help="Interval between runs in seconds",
    )
    parser.add_argument(
        "--jitter-sec", type=int, default=0, help="Random jitter in seconds"
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Exit on first command failure"
    )
    parser.add_argument(
        "--max-catchup",
        type=int,
        default=0,
        help="Maximum number of catchup runs on startup (0 = disabled)",
    )
    parser.add_argument(
        "--catchup-cooldown-sec",
        type=int,
        default=1,
        help="Cooldown between catchup runs in seconds",
    )
    parser.add_argument("command", help="Command to run (use quotes for multi-arg)")

    args = parser.parse_args()

    kill_file = Path("ztb.stop")
    cycle_count = 0

    def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
        print("\nStopping cronish...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Perform catchup on startup
    if args.max_catchup > 0:
        perform_catchup(args, kill_file)

    print(
        f"Starting cronish: '{args.command}' every {args.interval_sec}s (±{args.jitter_sec}s)"
    )

    while True:
        if kill_file.exists():
            print("Kill file detected, stopping...")
            sys.exit(0)

        cycle_count += 1
        next_run = calculate_next_run(args.interval_sec, args.jitter_sec)

        print(f"Cycle {cycle_count}: Running command...")
        exit_code = run_command(args.command)

        if exit_code != 0 and args.fail_fast:
            print(f"Command failed with exit code {exit_code}, exiting...")
            sys.exit(exit_code)

        now = time.time()
        if next_run > now:
            eta = next_run - now
            print(f"Next run in {format_eta(eta)}")
            time.sleep(eta)
        else:
            print("Running immediately (overdue)")


if __name__ == "__main__":
    main()
