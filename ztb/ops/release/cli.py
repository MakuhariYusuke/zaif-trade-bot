#!/usr/bin/env python3
"""
Ops command wrapper for Zaif Trade Bot.

Provides unified interface to monitoring and utility scripts.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, args: list[str]) -> int:
    """Run a script with given arguments."""
    script_path = Path("scripts") / script_name
    if not script_path.exists():
        print(f"Script not found: {script_path}", file=sys.stderr)
        return 1

    # Build command
    cmd = [sys.executable, str(script_path)] + args

    # Execute
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except Exception as e:
        print(f"Failed to run {script_name}: {e}", file=sys.stderr)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zaif Trade Bot operations wrapper",
        usage="%(prog)s <command> [args...]\n\nAvailable commands:\n"
        "  watch     Run watch_1m.py\n"
        "  rollup    Run rollup_artifacts.py\n"
        "  tb        Run tb_scrape.py\n"
        "  eta       Run progress_eta.py\n"
        "  errors    Run collect_last_errors.py\n"
        "  notify    Run alert_notifier.py\n"
        "  index     Run index_sessions.py\n"
        "  health    Run disk_health.py\n"
        "  cron      Run cronish.py\n"
        "  validate  Run validate_artifacts.py\n"
        "  bundle    Run bundle_artifacts.py\n"
        "  launch    Run launch_monitoring.py",
    )

    parser.add_argument("command", help="Command to run")
    parser.add_argument("args", nargs="*", help="Arguments for the command")

    args = parser.parse_args()

    # Map commands to scripts
    command_map = {
        "watch": "watch_1m.py",
        "rollup": "rollup_artifacts.py",
        "tb": "tb_scrape.py",
        "eta": "progress_eta.py",
        "errors": "collect_last_errors.py",
        "notify": "alert_notifier.py",
        "index": "index_sessions.py",
        "health": "disk_health.py",
        "cron": "cronish.py",
        "validate": "validate_artifacts.py",
        "bundle": "bundle_artifacts.py",
        "launch": "launch_monitoring.py",
    }

    if args.command not in command_map:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        print("Use --help for available commands", file=sys.stderr)
        sys.exit(1)

    script_name = command_map[args.command]
    exit_code = run_script(script_name, args.args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
