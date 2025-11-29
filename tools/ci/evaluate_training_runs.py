#!/usr/bin/env python3
"""
Simple script to aggregate training_report_*.json produced by AB tests/training runs
and write a summary JSON that CI can upload. The script prints aggregated metrics and
exits with 0. Non-zero exit codes may be used by CI to flag warnings.

Usage:
    python tools/ci/evaluate_training_runs.py --out reports/ab_summary.json
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def find_training_reports() -> List[Path]:
    p = Path("reports")
    if not p.exists():
        return []
    return list(p.glob("training_report_*.json"))


def extract_metrics_from_report(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    # Report may store metrics under 'training_stats' or 'metrics'
    metrics = obj.get("training_stats") or obj.get("metrics") or {}
    # Extract useful fields if present
    out = {
        "file": str(path),
        "model_name": obj.get("configuration", {})
        .get("training", {})
        .get("model_name"),
    }
    out.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/ab_summary.json")
    args = parser.parse_args()

    reports = find_training_reports()
    results = []
    # Group results by model_name
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in reports:
        mt = extract_metrics_from_report(r)
        if not mt:
            continue
        name = mt.get("model_name") or "unknown"
        grouped.setdefault(name, []).append(mt)

    # Aggregate each group into a single summary entry
    for name, items in grouped.items():
        agg: Dict[str, Any] = {"model_name": name, "files": [it.get("file") for it in items]}
        # compute averages for known metrics
        # We specifically care about sharpe_ratio and total_return
        sharpe_vals = [float(it.get("sharpe_ratio")) for it in items if it.get("sharpe_ratio") is not None]
        tr_vals = [float(it.get("total_return")) for it in items if it.get("total_return") is not None]
        agg["report_count"] = len(items)
        agg["mean_sharpe"] = sum(sharpe_vals) / len(sharpe_vals) if sharpe_vals else 0.0
        agg["mean_total_return"] = sum(tr_vals) / len(tr_vals) if tr_vals else 0.0
        results.append(agg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Print summary
    print(f"Found {len(results)} training reports")
    best_sharpe = None
    for r in results:
        if "sharpe_ratio" in r:
            try:
                s = float(r["sharpe_ratio"])
                if best_sharpe is None or s > best_sharpe:
                    best_sharpe = s
            except Exception:
                pass
    if best_sharpe is not None:
        print(f"Best sharpe found: {best_sharpe:.4f}")
    else:
        print("No sharpe values present in reports")

    print("Summary written to:", out_path)


if __name__ == "__main__":
    main()
