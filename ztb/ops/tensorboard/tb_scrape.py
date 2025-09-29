#!/usr/bin/env python3
"""
TensorBoard scalar scraper for Zaif Trade Bot.

Converts TB event scalars to CSV and optionally merges latest into metrics.json.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def scrape_scalars(tb_dir: Path, out_csv: Path) -> Dict[str, float]:
    """Scrape scalars from TB events."""
    if not HAS_TENSORBOARD:
        print("TensorBoard not available, skipping")
        return {}

    if not tb_dir.exists():
        print(f"TB directory not found: {tb_dir}")
        return {}

    latest_scalars = {}

    for event_file in tb_dir.glob("events.out.tfevents.*"):
        try:
            ea = EventAccumulator(str(event_file))
            ea.Reload()
            for tag in ea.Tags()["scalars"]:
                values = ea.Scalars(tag)
                if values:
                    latest_scalars[tag] = values[-1].value
        except Exception as e:
            print(f"Error reading {event_file}: {e}")

    if latest_scalars:
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tag", "value"])
            for tag, value in latest_scalars.items():
                writer.writerow([tag, value])

    return latest_scalars


def merge_to_metrics(correlation_id: str, scalars: Dict[str, float]) -> None:
    """Merge scalars into metrics.json."""
    metrics_path = Path("artifacts") / correlation_id / "metrics.json"
    if not metrics_path.exists():
        return

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        metrics["tb_latest"] = scalars
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"Error merging to metrics: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape TensorBoard scalars")
    parser.add_argument("--correlation-id", required=True, help="Correlation ID")
    parser.add_argument("--tb-dir", type=Path, help="TB directory")
    parser.add_argument("--out-csv", type=Path, help="Output CSV")
    parser.add_argument(
        "--merge-metrics", action="store_true", help="Merge to metrics.json"
    )

    args = parser.parse_args()

    tb_dir = args.tb_dir or Path("artifacts") / args.correlation_id / "tb"
    out_csv = (
        args.out_csv
        or Path("artifacts") / args.correlation_id / "reports" / "tb_scalars.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    scalars = scrape_scalars(tb_dir, out_csv)

    if args.merge_metrics:
        merge_to_metrics(args.correlation_id, scalars)

    print(f"Scraped {len(scalars)} scalars to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
