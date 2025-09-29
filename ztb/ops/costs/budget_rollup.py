#!/usr/bin/env python3
"""
budget_rollup.py - Aggregate cost estimates into daily budget markdown report

This script aggregates cost estimates from run_metadata.json and cost_estimator
results into a daily budget report in markdown format.

Usage:
    python budget_rollup.py --output reports/budget_daily.md
    python budget_rollup.py --date 2024-01-15
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


class DailyTotal(TypedDict):
    total_cost_jpy: float
    gpu_hours: float
    runs: List[Dict[str, Any]]
    run_count: int


def load_run_metadata(runs_dir: Path) -> List[Dict[str, Any]]:
    """Load run metadata from runs directory."""
    metadata_list: List[Dict[str, Any]] = []
    if not runs_dir.exists():
        return metadata_list

    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            metadata_file = run_dir / "run_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                        metadata["run_dir"] = str(run_dir)
                        metadata_list.append(metadata)
                except (json.JSONDecodeError, IOError):
                    continue

    return metadata_list


def load_cost_estimates(runs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load cost estimates from runs directory."""
    cost_estimates: Dict[str, Dict[str, Any]] = {}
    if not runs_dir.exists():
        return cost_estimates

    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            cost_file = run_dir / "cost_estimate.json"
            if cost_file.exists():
                try:
                    with open(cost_file, "r", encoding="utf-8") as f:
                        cost_data = json.load(f)
                        cost_estimates[str(run_dir)] = cost_data
                except (json.JSONDecodeError, IOError):
                    continue

    return cost_estimates


def aggregate_by_date(
    metadata_list: List[Dict[str, Any]],
    cost_estimates: Dict[str, Dict[str, Any]],
    target_date: Optional[date] = None,
) -> Dict[str, DailyTotal]:
    """Aggregate costs by date."""
    daily_totals: Dict[str, DailyTotal] = defaultdict(
        lambda: {"total_cost_jpy": 0.0, "gpu_hours": 0.0, "runs": [], "run_count": 0}
    )

    for metadata in metadata_list:
        run_dir = metadata["run_dir"]
        start_time = metadata.get("start_time")
        if not start_time:
            continue

        try:
            run_date = datetime.fromisoformat(start_time).date()
            if target_date and run_date != target_date:
                continue
        except (ValueError, TypeError):
            continue

        date_key = run_date.isoformat()
        cost_data = cost_estimates.get(run_dir, {})

        total_cost = cost_data.get("total_cost_jpy", 0.0)
        gpu_hours = cost_data.get("gpu_hours", 0.0)

        daily_totals[date_key]["total_cost_jpy"] += total_cost
        daily_totals[date_key]["gpu_hours"] += gpu_hours
        daily_totals[date_key]["run_count"] += 1
        daily_totals[date_key]["runs"].append(
            {
                "run_dir": run_dir,
                "cost_jpy": total_cost,
                "gpu_hours": gpu_hours,
                "start_time": start_time,
            }
        )

    return dict(daily_totals)


def generate_markdown_report(daily_totals: Dict[str, DailyTotal]) -> str:
    """Generate markdown report from daily totals."""
    lines = ["# Daily Budget Report\n"]

    if not daily_totals:
        lines.append("No cost data found.\n")
        return "\n".join(lines)

    # Sort dates
    sorted_dates = sorted(daily_totals.keys(), reverse=True)

    total_all_cost = 0.0
    total_all_gpu_hours = 0.0

    for date_key in sorted_dates:
        data = daily_totals[date_key]
        lines.append(f"## {date_key}\n")
        lines.append(f"- **Total Cost**: ¥{data['total_cost_jpy']:,.0f}")
        lines.append(f"- **GPU Hours**: {data['gpu_hours']:.1f}")
        lines.append(f"- **Runs**: {data['run_count']}\n")

        if data["runs"]:
            lines.append("### Runs\n")
            for run in sorted(
                data["runs"], key=lambda x: x["start_time"], reverse=True
            ):
                lines.append(
                    f"- `{run['run_dir']}`: ¥{run['cost_jpy']:,.0f} ({run['gpu_hours']:.1f}h)"
                )
            lines.append("")

        total_all_cost += data["total_cost_jpy"]
        total_all_gpu_hours += data["gpu_hours"]

    lines.append("## Summary\n")
    lines.append(f"- **Total Cost (All Dates)**: ¥{total_all_cost:,.0f}")
    lines.append(f"- **Total GPU Hours (All Dates)**: {total_all_gpu_hours:.1f}")
    lines.append(f"- **Total Days**: {len(daily_totals)}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate cost estimates into budget report"
    )
    parser.add_argument(
        "--runs-dir",
        "-d",
        type=Path,
        default=Path("runs"),
        help="Runs directory (default: runs/)",
    )
    parser.add_argument("--output", "-o", type=Path, help="Output markdown file path")
    parser.add_argument(
        "--date",
        type=lambda x: datetime.fromisoformat(x).date(),
        help="Filter by specific date (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    try:
        metadata_list = load_run_metadata(args.runs_dir)
        cost_estimates = load_cost_estimates(args.runs_dir)

        daily_totals = aggregate_by_date(metadata_list, cost_estimates, args.date)

        markdown_report = generate_markdown_report(daily_totals)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(markdown_report)
            print(f"Budget report written to {args.output}")
        else:
            print(markdown_report)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
