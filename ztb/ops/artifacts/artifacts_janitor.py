#!/usr/bin/env python3
"""
Artifacts janitor for Zaif Trade Bot.

Housekeeps artifacts: delete old runs, rotate oversized logs.
"""

import argparse
import gzip
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path


def rotate_log(log_path: Path, max_mb: float):
    """Rotate log if oversized."""
    if not log_path.exists():
        return

    size_mb = log_path.stat().st_size / 1024 / 1024
    if size_mb > max_mb:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_path = log_path.with_suffix(f".{timestamp}.gz")
        with open(log_path, "rb") as f_in:
            with gzip.open(rotated_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        log_path.unlink()
        print(f"Rotated {log_path} to {rotated_path}")


def delete_old_runs(root: Path, retention_days: int, dry_run: bool):
    """Delete old correlation directories."""
    cutoff = datetime.now() - timedelta(days=retention_days)

    for item in root.iterdir():
        if item.is_dir() and item.name.startswith(
            ("20", "canary_")
        ):  # Correlation IDs or canary
            try:
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if mtime < cutoff:
                    if dry_run:
                        print(f"Would delete: {item}")
                    else:
                        shutil.rmtree(item)
                        print(f"Deleted: {item}")
            except Exception as e:
                print(f"Error checking {item}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Janitor for artifacts")
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts"), help="Root directory"
    )
    parser.add_argument("--retention-days", type=int, default=30, help="Retention days")
    parser.add_argument("--max-log-mb", type=float, default=10, help="Max log size MB")
    parser.add_argument(
        "--dry-run", action="store_true", default=True, help="Dry run (default)"
    )
    parser.add_argument("--apply", action="store_true", help="Actually apply changes")

    args = parser.parse_args()

    dry_run = args.dry_run and not args.apply

    # Rotate logs
    for corr_dir in args.root.iterdir():
        if corr_dir.is_dir():
            logs_dir = corr_dir / "logs"
            if logs_dir.exists():
                rotate_log(logs_dir / "watch_log.jsonl", args.max_log_mb)
                rotate_log(logs_dir / "supervise_log.txt", args.max_log_mb)

    # Delete old runs
    delete_old_runs(args.root, args.retention_days, dry_run)

    print(f"Janitor complete (dry_run={dry_run})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
