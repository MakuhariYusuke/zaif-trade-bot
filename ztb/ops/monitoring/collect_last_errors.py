#!/usr/bin/env python3
"""
Last errors collector for Zaif Trade Bot.

Scans logs and watch_log.jsonl for recent ERROR/FAIL entries and generates a summary.
"""

import argparse
import json
import sys
from pathlib import Path

from ztb.io.text_io import read_last_lines, write_text

def extract_errors_from_logs(log_dir: Path) -> list[str]:
    """Extract ERROR/FAIL lines from log files."""
    errors: list[str] = []
    log_files = list(log_dir.glob("*.log"))

    for log_file in log_files:
        try:
            lines = read_last_lines(log_file, count=1000)

            # Look for ERROR or FAIL in lines
            for line in lines:
                if "ERROR" in line.upper() or "FAIL" in line.upper():
                    errors.append(f"{log_file.name}: {line.strip()}")
        except Exception as e:
            print(f"Error reading {log_file}: {e}", file=sys.stderr)

    return errors

def extract_errors_from_watch_log(watch_log_path: Path) -> list[str]:
    """Extract ERROR/FAIL alerts from watch_log.jsonl."""
    errors: list[str] = []

    if not watch_log_path.exists():
        return errors

    try:
        lines = read_last_lines(watch_log_path, count=1000)

        for line in lines:
            try:
                payload = json.loads(line.strip())
                if not isinstance(payload, dict):
                    continue
                alert = payload
                level = str(alert.get("level", "")).upper()
                if level in ("ERROR", "CRITICAL", "FAIL"):
                    timestamp = alert.get("timestamp", "unknown")
                    message = alert.get("message", str(alert))
                    errors.append(f"watch_log: {timestamp} {level}: {message}")
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"Error reading {watch_log_path}: {e}", file=sys.stderr)

    return errors

def _resolve_watch_log_path(correlation_id: str) -> Path:
    """Resolve watch_log path from canonical artifacts location, with legacy fallback."""
    canonical = Path("artifacts") / correlation_id / "logs" / "watch_log.jsonl"
    if canonical.exists():
        return canonical
    return Path("watch_log.jsonl")

def collect_last_errors(correlation_id: str) -> list[str]:
    """Collect last errors from all sources."""
    log_dir = Path("logs")
    watch_log_path = _resolve_watch_log_path(correlation_id)

    errors = []
    errors.extend(extract_errors_from_logs(log_dir))
    errors.extend(extract_errors_from_watch_log(watch_log_path))

    # Sort by recency (assuming logs are appended, so later in file = more recent)
    # For simplicity, reverse the list to get most recent first
    errors.reverse()

    return errors[:50]  # Limit to last 50 errors

def write_errors_report(correlation_id: str, errors: list[str]) -> None:
    """Write errors to reports file."""
    reports_dir = Path("artifacts") / correlation_id / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "last_errors.txt"

    if errors:
        lines = ["Last errors collected:", ""]
        lines.extend(errors)
        content = "\n".join(lines) + "\n"
    else:
        content = "no recent errors\n"

    write_text(report_path, content)

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
