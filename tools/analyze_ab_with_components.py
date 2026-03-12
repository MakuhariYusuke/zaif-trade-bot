#!/usr/bin/env python3
"""
Enhanced AB Test Results Analyzer with reward_components visualization.

Analyzes AB test results and visualizes reward_components to understand
which reward shaping strategies are most effective.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

from ztb.io.json_io import write_json
from ztb.reporting.services.catalog import (
    extract_action_distribution_from_payload,
    extract_reward_components_from_payload,
    get_recent_training_reports,
    list_training_reports,
    load_training_report,
)
from ztb.trading.environment.components.rewards.utils import RewardUtils
from ztb.utils.safety import ensure_dict, safe_to_float

DEFAULT_PATTERN = "training_report_*.json"


class LoadedTrainingReport(TypedDict):
    report_file: str
    mtime: float
    payload: dict[str, object]


class BalanceMetrics(TypedDict):
    balance_score: float
    buy_ratio: float
    sell_ratio: float
    hold_ratio: float


class CorrelationDataPoint(TypedDict):
    report_file: str
    reward_components: dict[str, float]
    balance_metrics: BalanceMetrics
    config: dict[str, object]


class ComponentStatistics(TypedDict):
    mean: float
    min: float
    max: float
    count: int


def _list_report_paths(
    reports_dir: Path,
    pattern: str,
    recent_limit: int | None,
) -> list[Path]:
    if pattern == DEFAULT_PATTERN:
        if recent_limit is not None and recent_limit > 0:
            return get_recent_training_reports(limit=recent_limit, reports_dir=reports_dir)
        return list_training_reports(reports_dir=reports_dir)

    report_paths = list(reports_dir.glob(pattern))
    if recent_limit is None or recent_limit <= 0:
        return report_paths

    def mtime_or_zero(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(report_paths, key=mtime_or_zero, reverse=True)[:recent_limit]


def load_training_reports(
    reports_dir: Path,
    pattern: str = DEFAULT_PATTERN,
    filter_recent: int | None = None,
) -> list[LoadedTrainingReport]:
    """Load all training reports matching pattern."""
    reports: list[LoadedTrainingReport] = []
    for report_path in _list_report_paths(reports_dir, pattern, filter_recent):
        payload = load_training_report(report_path)
        if payload is None:
            print(f"Warning: Could not load {report_path.name}: invalid JSON")
            continue
        try:
            mtime = report_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        reports.append(
            {
                "report_file": report_path.name,
                "mtime": mtime,
                "payload": payload,
            }
        )
    return reports


def analyze_action_balance(report_payload: dict[str, object]) -> BalanceMetrics:
    """Calculate action balance metrics."""
    action_dist = extract_action_distribution_from_payload(report_payload)

    buy = safe_to_float(action_dist.get("BUY"), 0.0)
    sell = safe_to_float(action_dist.get("SELL"), 0.0)
    hold = safe_to_float(action_dist.get("HOLD"), 0.0)
    total = buy + sell + hold

    if total <= 0:
        return {
            "balance_score": float("inf"),
            "buy_ratio": 0.0,
            "sell_ratio": 0.0,
            "hold_ratio": 0.0,
        }

    buy_ratio = buy / total
    sell_ratio = sell / total
    hold_ratio = hold / total

    balance_score = RewardUtils.calculate_balance_deviation_from_ratios(
        [buy_ratio, sell_ratio, hold_ratio], [0.333, 0.333, 0.333]
    )
    return {
        "balance_score": balance_score,
        "buy_ratio": buy_ratio,
        "sell_ratio": sell_ratio,
        "hold_ratio": hold_ratio,
    }


def _aggregate_components(
    data_points: list[CorrelationDataPoint],
) -> dict[str, ComponentStatistics]:
    components: defaultdict[str, list[float]] = defaultdict(list)
    for data_point in data_points:
        for key, value in data_point["reward_components"].items():
            components[key].append(safe_to_float(value, 0.0))

    stats: dict[str, ComponentStatistics] = {}
    for key, values in components.items():
        if not values:
            continue
        stats[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }
    return stats


def correlate_components_with_balance(
    reports: list[LoadedTrainingReport],
) -> dict[str, object]:
    """Correlate reward_components with action balance."""
    data_points: list[CorrelationDataPoint] = []

    for report in reports:
        payload = report["payload"]
        components = extract_reward_components_from_payload(payload)
        if not components:
            continue
        balance_metrics = analyze_action_balance(payload)
        data_points.append(
            {
                "report_file": report["report_file"],
                "reward_components": components,
                "balance_metrics": balance_metrics,
                "config": ensure_dict(payload.get("configuration")),
            }
        )

    if not data_points:
        return {"error": "No reports with reward_components found"}

    best_config = min(data_points, key=lambda item: item["balance_metrics"]["balance_score"])
    good_balance = [
        item for item in data_points if item["balance_metrics"]["balance_score"] < 0.2
    ]
    poor_balance = [
        item for item in data_points if item["balance_metrics"]["balance_score"] > 0.4
    ]

    return {
        "total_reports": len(data_points),
        "best_balanced": {
            "report_file": best_config["report_file"],
            "balance_score": best_config["balance_metrics"]["balance_score"],
            "action_distribution": {
                "buy": best_config["balance_metrics"]["buy_ratio"],
                "sell": best_config["balance_metrics"]["sell_ratio"],
                "hold": best_config["balance_metrics"]["hold_ratio"],
            },
            "reward_components": best_config["reward_components"],
        },
        "good_balance_stats": {
            "count": len(good_balance),
            "components": _aggregate_components(good_balance),
        },
        "poor_balance_stats": {
            "count": len(poor_balance),
            "components": _aggregate_components(poor_balance),
        },
    }


def print_analysis_report(
    analysis: dict[str, object],
    output_file: Path | None = None,
) -> None:
    """Print formatted analysis report."""
    if "error" in analysis:
        print(f"\n❌ {analysis['error']}")
        return

    total_reports = safe_to_float(analysis.get("total_reports"), 0.0)
    best = ensure_dict(analysis.get("best_balanced"))
    best_dist = ensure_dict(best.get("action_distribution"))
    best_components = ensure_dict(best.get("reward_components"))
    good = ensure_dict(analysis.get("good_balance_stats"))
    poor = ensure_dict(analysis.get("poor_balance_stats"))
    good_components = ensure_dict(good.get("components"))
    poor_components = ensure_dict(poor.get("components"))

    print("\n" + "=" * 80)
    print("AB Test Results Analysis with reward_components")
    print("=" * 80)
    print(f"\nTotal reports analyzed: {int(total_reports)}")

    print("\n" + "-" * 80)
    print("Best Balanced Configuration:")
    print("-" * 80)
    print(f"  Report: {best.get('report_file', 'unknown')}")
    print(f"  Balance Score: {safe_to_float(best.get('balance_score'), 0.0):.4f}")
    print("  Action Distribution:")
    print(f"    BUY:  {safe_to_float(best_dist.get('buy'), 0.0):.2%}")
    print(f"    SELL: {safe_to_float(best_dist.get('sell'), 0.0):.2%}")
    print(f"    HOLD: {safe_to_float(best_dist.get('hold'), 0.0):.2%}")
    print("\n  Reward Components:")
    for key, value in best_components.items():
        print(f"    {key:20s}: {safe_to_float(value, 0.0):8.6f}")

    print("\n" + "-" * 80)
    print(f"Good Balance (score < 0.2): {int(safe_to_float(good.get('count'), 0.0))} reports")
    print("-" * 80)
    if good_components:
        for key, stats_obj in good_components.items():
            stats = ensure_dict(stats_obj)
            print(
                f"  {key:20s}: mean={safe_to_float(stats.get('mean'), 0.0):8.6f}, "
                f"min={safe_to_float(stats.get('min'), 0.0):8.6f}, "
                f"max={safe_to_float(stats.get('max'), 0.0):8.6f}"
            )

    print("\n" + "-" * 80)
    print(f"Poor Balance (score > 0.4): {int(safe_to_float(poor.get('count'), 0.0))} reports")
    print("-" * 80)
    if poor_components:
        for key, stats_obj in poor_components.items():
            stats = ensure_dict(stats_obj)
            print(
                f"  {key:20s}: mean={safe_to_float(stats.get('mean'), 0.0):8.6f}, "
                f"min={safe_to_float(stats.get('min'), 0.0):8.6f}, "
                f"max={safe_to_float(stats.get('max'), 0.0):8.6f}"
            )

    print("\n" + "-" * 80)
    print("Insights:")
    print("-" * 80)
    if good_components and poor_components:
        common_keys = set(good_components.keys()) & set(poor_components.keys())
        for key in sorted(common_keys):
            good_stats = ensure_dict(good_components.get(key))
            poor_stats = ensure_dict(poor_components.get(key))
            good_mean = safe_to_float(good_stats.get("mean"), 0.0)
            poor_mean = safe_to_float(poor_stats.get("mean"), 0.0)
            diff = good_mean - poor_mean
            if abs(diff) > 0.001:
                direction = "higher" if diff > 0 else "lower"
                print(f"  • Good balance has {direction} {key}: {diff:+.6f} difference")

    if output_file:
        write_json(output_file, analysis, indent=2, ensure_ascii=False)
        print(f"\n✓ Analysis saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze AB test results with reward_components"
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory containing training reports",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="File pattern to match",
    )
    parser.add_argument("--output", help="Output JSON file for analysis results")
    parser.add_argument("--filter-recent", type=int, help="Only analyze N most recent reports")

    args = parser.parse_args()
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        print(f"Error: Directory not found: {reports_dir}")
        return

    print(f"Loading reports from: {reports_dir}")
    reports = load_training_reports(
        reports_dir=reports_dir,
        pattern=args.pattern,
        filter_recent=args.filter_recent,
    )
    print(f"Loaded {len(reports)} reports")

    print("\nAnalyzing reward_components and action balance...")
    analysis = correlate_components_with_balance(reports)

    output_path = Path(args.output) if args.output else None
    print_analysis_report(analysis, output_path)


if __name__ == "__main__":
    main()
