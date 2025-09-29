#!/usr/bin/env python3
"""
Last errors collector for Zaif Trade Bot.

Scans logs and watch_log.jsonl for recent ERROR/FAIL entries and generates a summary.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List


def extract_errors_from_logs(log_dir: Path) -> List[str]:
    """Extract ERROR/FAIL lines from log files."""
    errors: List[str] = []
    log_files = list(log_dir.glob("*.log"))

    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()

            # Look for ERROR or FAIL in lines
            for line in lines[-1000:]:  # Last 1000 lines
                if "ERROR" in line.upper() or "FAIL" in line.upper():
                    errors.append(f"{log_file.name}: {line.strip()}")
        except Exception as e:
            print(f"Error reading {log_file}: {e}", file=sys.stderr)

    return errors


def extract_errors_from_watch_log(watch_log_path: Path) -> List[str]:
    """Extract ERROR/FAIL alerts from watch_log.jsonl."""
    errors = []

    if not watch_log_path.exists():
        return errors

    try:
        with open(watch_log_path, "r") as f:
            lines = f.readlines()

        for line in lines[-1000:]:  # Last 1000 entries
            try:
                alert = json.loads(line.strip())
                level = alert.get("level", "").upper()
                if level in ("ERROR", "CRITICAL", "FAIL"):
                    timestamp = alert.get("timestamp", "unknown")
                    message = alert.get("message", str(alert))
                    errors.append(f"watch_log: {timestamp} {level}: {message}")
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"Error reading {watch_log_path}: {e}", file=sys.stderr)

    return errors


def collect_last_errors(correlation_id: str) -> List[str]:
    """Collect last errors from all sources."""
    log_dir = Path("logs")
    watch_log_path = Path("watch_log.jsonl")

    errors = []
    errors.extend(extract_errors_from_logs(log_dir))
    errors.extend(extract_errors_from_watch_log(watch_log_path))

    # Sort by recency (assuming logs are appended, so later in file = more recent)
    # For simplicity, reverse the list to get most recent first
    errors.reverse()

    return errors[:50]  # Limit to last 50 errors


def write_errors_report(correlation_id: str, errors: List[str]) -> None:
    """Write errors to reports file."""
    reports_dir = Path("artifacts") / correlation_id / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "last_errors.txt"

    with open(report_path, "w") as f:
        if errors:
            f.write("Last errors collected:\n\n")
            for error in errors:
                f.write(f"{error}\n")
        else:
            f.write("no recent errors\n")

    print(f"Errors report written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect last errors for Zaif Trade Bot session"
    )
    parser.add_argument(
        "--correlation-id", required=True, help="Session correlation ID"
    )

    args = parser.parse_args()

    errors = collect_last_errors(args.correlation_id)

    if not errors:
        print("no recent errors")
        sys.exit(0)

    write_errors_report(args.correlation_id, errors)


if __name__ == "__main__":
    main()
