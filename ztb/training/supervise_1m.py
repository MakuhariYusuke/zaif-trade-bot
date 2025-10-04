#!/usr/bin/env python3
"""
Auto-resume supervisor for Zaif Trade Bot training sessions.

Runs training in a loop, resuming on non-zero exit.
Preserves correlation_id across restarts.
Rotates artifacts by timestamp under the same correlation root.
Writes artifacts/{corr}/logs/supervise_log.txt.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from ztb.utils.compat_wrapper import run_command_safely


def acquire_lock(lock_file: Path) -> bool:
    """Acquire exclusive lock."""
    try:
        if lock_file.exists():
            return False
        lock_file.touch()
        return True
    except Exception:
        return False


def check_kill_file() -> bool:
    """Check for kill file."""
    return Path("ztb.stop").exists()


def get_training_command(correlation_id: str, ppo_args: str = "") -> list[str]:
    """Get the training command to run."""
    run_1m_path = Path("ztb/ztb/ztb/scripts/run_1m.py")
    if run_1m_path.exists():
        cmd = [sys.executable, str(run_1m_path), "--correlation-id", correlation_id]
    else:
        # Canonical CLI
        cmd = [
            sys.executable,
            "-m",
            "ztb.training.ppo_trainer",
            "--resume-from",
            "latest",
            "--total-timesteps",
            "1000000",
            "--n-envs",
            "4",
            "--seed",
            "42",
            "--eval-interval",
            "10000",
            "--log-interval",
            "1000",
            "--ckpt-async",
            "--ckpt-compress",
            "zstd",
            "--ckpt-max-pending",
            "1",
        ]
        if ppo_args:
            cmd.extend(ppo_args.split())

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervise training with auto-resume")
    parser.add_argument(
        "--correlation-id", required=True, help="Correlation ID for this session"
    )
    parser.add_argument(
        "--ppo-cli-args", default="", help="Additional PPO CLI args (quoted string)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode for CI smoke test (no child process)",
    )

    args = parser.parse_args()

    correlation_id = args.correlation_id
    artifacts_root = Path("artifacts") / correlation_id
    logs_dir = artifacts_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    lock_file = artifacts_root / ".supervise.lock"
    if not acquire_lock(lock_file):
        print(f"Another supervisor is running for {correlation_id}", file=sys.stderr)
        sys.exit(1)

    log_file = logs_dir / "supervise_log.txt"

    backoff_times = [60, 120, 240, 600]  # Cap at 10 min
    attempt = 0

    with open(log_file, "a") as log:
        log.write(
            f"{datetime.now().isoformat()} - Starting supervision for {correlation_id}\n"
        )

        while True:
            if check_kill_file():
                log.write(
                    f"{datetime.now().isoformat()} - Kill file detected, stopping supervision\n"
                )
                break

            cmd = get_training_command(correlation_id, args.ppo_cli_args)

            log.write(f"{datetime.now().isoformat()} - Running: {' '.join(cmd)}\n")
            log.flush()

            if args.dry_run:
                # Simulate run
                time.sleep(1)
                exit_code = 0
                log.write(
                    f"{datetime.now().isoformat()} - Dry run completed with exit code {exit_code}\n"
                )
            else:
                try:
                    result = run_command_safely(cmd, timeout=3600, cwd=str(Path.cwd()))
                    exit_code = result["returncode"]
                    log.write(
                        f"{datetime.now().isoformat()} - Exit code: {exit_code}\n"
                    )
                    if result["stdout"]:
                        log.write(f"STDOUT: {result['stdout']}\n")
                    if result["stderr"]:
                        log.write(f"STDERR: {result['stderr']}\n")
                except Exception as e:
                    exit_code = 1
                    log.write(
                        f"{datetime.now().isoformat()} - Exception: {e}, exit code: {exit_code}\n"
                    )

            if exit_code == 0:
                log.write(
                    f"{datetime.now().isoformat()} - Training completed successfully\n"
                )
                break
            elif exit_code >= 3:
                log.write(
                    f"{datetime.now().isoformat()} - Non-retriable exit code {exit_code}, stopping\n"
                )
                break
            else:
                # Retriable
                backoff = backoff_times[min(attempt, len(backoff_times) - 1)]
                log.write(
                    f"{datetime.now().isoformat()} - Retriable failure, sleeping {backoff}s then resuming\n"
                )
                time.sleep(backoff)
                attempt += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
