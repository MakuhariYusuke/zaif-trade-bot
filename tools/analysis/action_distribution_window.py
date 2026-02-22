#!/usr/bin/env python3
"""
Simple tool to aggregate action distributions from training reports and evaluation logs.
"""
import argparse
import json
import gzip
from pathlib import Path
from typing import TypedDict

from ztb.reporting.services.catalog import list_training_reports, load_training_report
from ztb.utils.safety import ensure_dict, safe_to_float, safe_to_int


class ActionDistribution(TypedDict):
    HOLD: float
    BUY: float
    SELL: float


def _normalize_action_distribution(payload: object) -> ActionDistribution:
    distribution = ensure_dict(payload)
    return {
        "HOLD": safe_to_float(distribution.get("HOLD"), 0.0),
        "BUY": safe_to_float(distribution.get("BUY"), 0.0),
        "SELL": safe_to_float(distribution.get("SELL"), 0.0),
    }


def _extract_from_report_file(path: Path, start: int, end: int) -> list[ActionDistribution]:
    results: list[ActionDistribution] = []
    data = load_training_report(path)
    if data is None:
        return results

    # Traverse and collect any action_distribution entries.
    def _collect(obj: object) -> None:
        if isinstance(obj, dict):
            if "action_distribution" in obj and "step" in obj:
                step = safe_to_int(obj.get("step"), -1)
                if start <= step < end:
                    results.append(_normalize_action_distribution(obj.get("action_distribution")))
            # Recurse
            for v in obj.values():
                _collect(v)
        elif isinstance(obj, list):
            for v in obj:
                _collect(v)

    _collect(data)
    return results


def _extract_from_jsonl_gz(path: Path, start: int, end: int) -> list[ActionDistribution]:
    results: list[ActionDistribution] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                # Many evaluation log lines have step/time
                if "action_distribution" in obj and "step" in obj:
                    step = safe_to_int(obj.get("step"), -1)
                    if start <= step < end:
                        results.append(_normalize_action_distribution(obj.get("action_distribution")))
    except Exception:
        pass
    return results


def aggregate(distributions: list[ActionDistribution]) -> dict[str, float]:
    if not distributions:
        return {}
    keys = ["HOLD", "BUY", "SELL"]
    sums = {k: 0.0 for k in keys}
    for d in distributions:
        for k in keys:
            sums[k] += safe_to_float(d.get(k), 0.0)
    n = len(distributions)
    return {k: sums[k] / n for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="reports", help="Path to reports directory")
    parser.add_argument("--eval-logs", default="logs", help="Path to logs or evaluation folder")
    parser.add_argument("--start", type=int, default=1000)
    parser.add_argument("--end", type=int, default=2000)
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    eval_dir = Path(args.eval_logs)

    collected: list[ActionDistribution] = []

    # Read JSON reports
    if reports_dir.exists():
        for p in list_training_reports(reports_dir=reports_dir):
            collected.extend(_extract_from_report_file(p, args.start, args.end))

    # Read jsonl.gz evaluation logs
    if eval_dir.exists():
        for p in eval_dir.glob("evaluation_history*.jsonl.gz"):
            collected.extend(_extract_from_jsonl_gz(p, args.start, args.end))

    agg = aggregate(collected)
    if not agg:
        print("No action_distribution entries found for steps in range")
    else:
        print(f"Action distribution aggregate for steps [{args.start}, {args.end}):")
        print(f"  HOLD: {safe_to_float(agg.get('HOLD'), 0.0):.3f}")
        print(f"  BUY:  {safe_to_float(agg.get('BUY'), 0.0):.3f}")
        print(f"  SELL: {safe_to_float(agg.get('SELL'), 0.0):.3f}")
        print(f"(Based on {len(collected)} evaluation snapshots)")


if __name__ == "__main__":
    main()
