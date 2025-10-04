#!/usr/bin/env python3
"""
Monitoring launcher for Zaif Trade Bot.

Launches watch, tb_scrape, and alert_notifier processes with management.
"""

import argparse
import signal
import subprocess
import sys
from types import FrameType
from typing import List, Optional


class MonitoringLauncher:
    def __init__(
        self,
        correlation_id: str,
        webhook_url: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.correlation_id = correlation_id
        self.webhook_url = webhook_url
        self.dry_run = dry_run
        self.processes: List[subprocess.Popen[str]] = []

    def get_commands(self) -> List[List[str]]:
        """Get list of commands to run."""
        commands = [
            [
                "python",
                "ztb/ztb/ztb/scripts/watch.py",
                "--correlation-id",
                self.correlation_id,
            ],
            [
                "python",
                "ztb/ztb/ztb/scripts/tb_scrape.py",
                "--correlation-id",
                self.correlation_id,
            ],
        ]

        if self.webhook_url:
            commands.append(
                [
                    "python",
                    "ztb/ztb/ztb/scripts/alert_notifier.py",
                    "--correlation-id",
                    self.correlation_id,
                ]
            )

        return commands

    def launch_processes(self) -> None:
        """Launch all monitoring processes."""
        commands = self.get_commands()

        for cmd in commands:
            if self.dry_run:
                print(f"Would run: {' '.join(cmd)}")
            else:
                try:
                    proc = subprocess.Popen(cmd, text=True)
                    self.processes.append(proc)
                    print(f"Launched: {' '.join(cmd)} (PID: {proc.pid})")
                except Exception as e:
                    print(f"Failed to launch {' '.join(cmd)}: {e}", file=sys.stderr)

    def wait_and_cleanup(self) -> None:
        """Wait for processes and clean up on signal."""

        def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
            print("\nReceived signal, terminating processes...")
            self.terminate_all()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if not self.dry_run:
            try:
                for proc in self.processes:
                    proc.wait()
            except KeyboardInterrupt:
                self.terminate_all()
        else:
            # In dry run, just wait for interrupt
            try:
                while True:
                    pass
            except KeyboardInterrupt:
                print("\nDry run interrupted.")

    def terminate_all(self) -> None:
        """Terminate all running processes."""
        for proc in self.processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"Terminated PID: {proc.pid}")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"Killed PID: {proc.pid}")
            except Exception as e:
                print(f"Error terminating PID {proc.pid}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch monitoring processes for Zaif Trade Bot session"
    )
    parser.add_argument(
        "--correlation-id", required=True, help="Session correlation ID"
    )
    parser.add_argument(
        "--webhook", help="Webhook URL for alerts (enables alert_notifier)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without executing"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the processes (default if not dry-run)",
    )

    args = parser.parse_args()

    if args.dry_run and args.execute:
        print("Cannot use both --dry-run and --execute", file=sys.stderr)
        sys.exit(1)

    dry_run = args.dry_run or not args.execute

    launcher = MonitoringLauncher(args.correlation_id, args.webhook, dry_run)
    launcher.launch_processes()
    launcher.wait_and_cleanup()


if __name__ == "__main__":
    main()
