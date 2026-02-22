#!/usr/bin/env python3
"""
Simple script to aggregate training_report_*.json produced by AB tests/training runs
and write a summary JSON that CI can upload. Reports are grouped by
`training.model_name` and the output contains `report_count`, `mean_sharpe`,
`mean_total_return`, and a list of `files` for the run artifacts. The output
is compatible with `tools/ci/check_optimizer_gates.py` used for gating.

Usage:
    python tools/ci/evaluate_training_runs.py --out reports/mtf_optimizer_summary.json
"""
import argparse
from pathlib import Path
from typing import TypedDict

from ztb.io.json_io import write_json
from ztb.reporting.services.catalog import list_training_reports, load_training_report
from ztb.utils.safety import ensure_dict, safe_to_float


class ReportMetrics(TypedDict):
    file: str
    model_name: str
    sharpe_ratio: float
    total_return: float


class ModelSummary(TypedDict):
    model_name: str
    files: list[str]
    report_count: int
    mean_sharpe: float
    mean_total_return: float


def find_training_reports() -> list[Path]:
    return list_training_reports(reports_dir=Path("reports"))


def extract_metrics_from_report(path: Path) -> ReportMetrics | None:
    obj = load_training_report(path)
    if obj is None:
        return None
    # Report may store metrics under 'training_stats' or 'metrics'
    metrics = ensure_dict(obj.get("training_stats"))
    if not metrics:
        metrics = ensure_dict(obj.get("metrics"))

    configuration = ensure_dict(obj.get("configuration"))
    training = ensure_dict(configuration.get("training"))
    raw_model_name = training.get("model_name")
    model_name = raw_model_name if isinstance(raw_model_name, str) and raw_model_name else "unknown"

    # Extract useful fields if present
    return {
        "file": str(path),
        "model_name": model_name,
        "sharpe_ratio": safe_to_float(metrics.get("sharpe_ratio"), 0.0),
        "total_return": safe_to_float(metrics.get("total_return"), 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/ab_summary.json")
    args = parser.parse_args()

    reports = find_training_reports()
    results: list[ModelSummary] = []
    # Group results by model_name
    grouped: dict[str, list[ReportMetrics]] = {}
    for r in reports:
        mt = extract_metrics_from_report(r)
        if mt is None:
            continue
        name = mt["model_name"]
        grouped.setdefault(name, []).append(mt)

    # Aggregate each group into a single summary entry
    for name, items in grouped.items():
        sharpe_vals = [item["sharpe_ratio"] for item in items]
        tr_vals = [item["total_return"] for item in items]

        agg: ModelSummary = {
            "model_name": name,
            "files": [item["file"] for item in items],
            "report_count": len(items),
            "mean_sharpe": sum(sharpe_vals) / len(sharpe_vals) if sharpe_vals else 0.0,
            "mean_total_return": sum(tr_vals) / len(tr_vals) if tr_vals else 0.0,
        }
        results.append(agg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, results, indent=2, ensure_ascii=False)

    # Print summary
    print(f"Found {len(reports)} training reports across {len(results)} model groups")
    best_sharpe = max((r["mean_sharpe"] for r in results), default=None)
    if best_sharpe is not None:
        print(f"Best sharpe found: {best_sharpe:.4f}")
    else:
        print("No sharpe values present in reports")

    print("Summary written to:", out_path)


if __name__ == "__main__":
    main()
