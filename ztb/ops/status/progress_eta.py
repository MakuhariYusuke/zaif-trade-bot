#!/usr/bin/env python3
"""
Progress ETA estimator for Zaif Trade Bot.

Estimates steps/sec from metrics/logs, computes ETA and completion prediction.
Optionally updates summary.json with progress info.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple


def estimate_steps_per_sec_from_metrics(metrics_path: Path) -> Optional[float]:
    """Estimate steps/sec from metrics.json."""
    if not metrics_path.exists():
        return None

    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)

        steps = data.get("steps", 0)
        elapsed = data.get("elapsed_time", 0)  # in seconds
        if elapsed > 0:
            return float(steps) / float(elapsed)
    except Exception as e:
        print(f"Error reading metrics: {e}", file=sys.stderr)

    return None


def estimate_steps_per_sec_from_logs(log_dir: Path) -> Optional[float]:
    """Estimate steps/sec from log files."""
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        return None

    # Look for patterns like "Step 1000/1000000" or similar
    step_pattern = re.compile(r"Step\s+(\d+)/(\d+)")
    time_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")

    steps_data = []

    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()[-100:]  # Last 100 lines for recent progress

            for line in lines:
                step_match = step_pattern.search(line)
                time_match = time_pattern.search(line)
                if step_match and time_match:
                    current_step = int(step_match.group(1))
                    total_steps = int(step_match.group(2))
                    timestamp = datetime.fromisoformat(time_match.group(1))
                    steps_data.append((current_step, timestamp))
        except Exception as e:
            print(f"Error reading log {log_file}: {e}", file=sys.stderr)

    if len(steps_data) < 2:
        return None

    # Sort by timestamp and calculate rate
    steps_data.sort(key=lambda x: x[1])
    first = steps_data[0]
    last = steps_data[-1]
    delta_steps = last[0] - first[0]
    delta_time = (last[1] - first[1]).total_seconds()

    if delta_time > 0:
        return delta_steps / delta_time

    return None


def compute_eta(
    current_steps: int, total_steps: int, steps_per_sec: float
) -> Tuple[str, str, float]:
    """Compute ETA, completion time, and percentage."""
    if steps_per_sec <= 0:
        return "unknown", "unknown", 0.0

    remaining_steps = total_steps - current_steps
    eta_seconds = remaining_steps / steps_per_sec
    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
    completion_time = eta_time.isoformat()
    pct = (current_steps / total_steps) * 100 if total_steps > 0 else 0.0

    return f"{eta_seconds:.0f}s", completion_time, pct


def update_summary(
    summary_path: Path, steps_per_sec: float, eta: str, completion_time: str, pct: float
) -> None:
    """Update summary.json with progress info."""
    if not summary_path.exists():
        return

    try:
        with open(summary_path, "r") as f:
            data = json.load(f)

        data["progress"] = {
            "steps_per_sec": steps_per_sec,
            "eta": eta,
            "completion_time": completion_time,
            "pct": pct,
        }

        with open(summary_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Updated {summary_path} with progress info")
    except Exception as e:
        print(f"Error updating summary: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate progress ETA for Zaif Trade Bot session"
    )
    parser.add_argument(
        "--correlation-id", required=True, help="Session correlation ID"
    )
    parser.add_argument(
        "--update-summary",
        action="store_true",
        help="Update summary.json with progress info",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=1000000,
        help="Total training steps (default: 1M)",
    )

    args = parser.parse_args()

    artifacts_dir = Path("artifacts") / args.correlation_id
    metrics_path = artifacts_dir / "metrics.json"
    summary_path = artifacts_dir / "summary.json"
    logs_dir = Path("logs")

    # Estimate steps/sec
    steps_per_sec = estimate_steps_per_sec_from_metrics(metrics_path)
    if steps_per_sec is None:
        steps_per_sec = estimate_steps_per_sec_from_logs(logs_dir)

    if steps_per_sec is None:
        print("Could not estimate steps/sec from metrics or logs", file=sys.stderr)
        sys.exit(1)

    # Get current steps from metrics
    current_steps = 0
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
            current_steps = data.get("steps", 0)
        except Exception:
            pass

    eta, completion_time, pct = compute_eta(
        current_steps, args.total_steps, steps_per_sec
    )

    print(f"Current steps: {current_steps}")
    print(f"Steps/sec: {steps_per_sec:.2f}")
    print(f"ETA: {eta}")
    print(f"Completion time: {completion_time}")
    print(f"Progress: {pct:.1f}%")

    if args.update_summary:
        update_summary(summary_path, steps_per_sec, eta, completion_time, pct)


if __name__ == "__main__":
    main()
