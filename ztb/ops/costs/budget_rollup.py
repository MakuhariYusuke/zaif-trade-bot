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
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional, TypedDict

from ztb.io.json_io import read_json_object
from ztb.io.text_io import write_text


class RunSummary(TypedDict):
    run_dir: str
    cost_jpy: float
    gpu_hours: float
    start_time: str


class DailyTotal(TypedDict):
    total_cost_jpy: float
    gpu_hours: float
    runs: list[RunSummary]
    run_count: int


def _iter_run_dirs(runs_dir: Path) -> list[Path]:
    """Return run subdirectories under runs_dir."""
    if not runs_dir.exists():
        return []
    return [run_dir for run_dir in runs_dir.iterdir() if run_dir.is_dir()]


def _load_run_json_file(run_dir: Path, filename: str) -> dict[str, object] | None:
    """Load one JSON object under a run directory."""
    target_file = run_dir / filename
    if not target_file.exists():
        return None
    try:
        return read_json_object(target_file)
    except (ValueError, TypeError, OSError):
        return None


def _to_float(value: object, default: float = 0.0) -> float:
    """Convert arbitrary value to float with safe fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_run_metadata(runs_dir: Path) -> list[dict[str, object]]:
    """Load run metadata from runs directory."""
    metadata_list: list[dict[str, object]] = []
    for run_dir in _iter_run_dirs(runs_dir):
        metadata = _load_run_json_file(run_dir, "run_metadata.json")
        if metadata is None:
            continue
        metadata["run_dir"] = str(run_dir)
        metadata_list.append(metadata)
    return metadata_list


def load_cost_estimates(runs_dir: Path) -> dict[str, dict[str, object]]:
    """Load cost estimates from runs directory."""
    cost_estimates: dict[str, dict[str, object]] = {}
    for run_dir in _iter_run_dirs(runs_dir):
        cost_data = _load_run_json_file(run_dir, "cost_estimate.json")
        if cost_data is None:
            continue
        cost_estimates[str(run_dir)] = cost_data
    return cost_estimates


def aggregate_by_date(
    metadata_list: list[dict[str, object]],
    cost_estimates: dict[str, dict[str, object]],
    target_date: Optional[date] = None,
) -> dict[str, DailyTotal]:
    """Aggregate costs by date."""
    daily_totals: dict[str, DailyTotal] = defaultdict(
        lambda: {"total_cost_jpy": 0.0, "gpu_hours": 0.0, "runs": [], "run_count": 0}
    )

    for metadata in metadata_list:
        run_dir_value = metadata.get("run_dir")
        start_time_value = metadata.get("start_time")
        if not isinstance(run_dir_value, str) or not isinstance(start_time_value, str):
            continue

        try:
            run_date = datetime.fromisoformat(start_time_value).date()
            if target_date and run_date != target_date:
                continue
        except (ValueError, TypeError):
            continue

        run_dir = run_dir_value
        date_key = run_date.isoformat()
        cost_data = cost_estimates.get(run_dir, {})

        total_cost = _to_float(cost_data.get("total_cost_jpy"), 0.0)
        gpu_hours = _to_float(cost_data.get("gpu_hours"), 0.0)

        daily_totals[date_key]["total_cost_jpy"] += total_cost
        daily_totals[date_key]["gpu_hours"] += gpu_hours
        daily_totals[date_key]["run_count"] += 1
        daily_totals[date_key]["runs"].append(
            {
                "run_dir": run_dir,
                "cost_jpy": total_cost,
                "gpu_hours": gpu_hours,
                "start_time": start_time_value,
            }
        )

    return dict(daily_totals)


def generate_markdown_report(daily_totals: dict[str, DailyTotal]) -> str:
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
            write_text(args.output, markdown_report, encoding="utf-8")
            print(f"Budget report written to {args.output}")
        else:
            print(markdown_report)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
