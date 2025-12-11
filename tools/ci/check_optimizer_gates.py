#!/usr/bin/env python3
"""
Check optimizer gates from a summary JSON and exit with non-zero if none meet the gates.

This script evaluates an aggregated summary file (produced by
`tools/ci/evaluate_training_runs.py` or `tools/training/confirm_candidate.py`) and
applies gates on `mean_sharpe` and `mean_total_return`.  Use `--min-reports` to
ensure only candidates with sufficient `report_count` are considered.

Usage:
    python tools/ci/check_optimizer_gates.py --summary reports/mtf_optimizer_summary.json --sharpe 0.5 --return 0.05 --min-reports 3
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--sharpe", type=float, default=0.5)
    parser.add_argument("--return", dest="rett", type=float, default=0.05)
    parser.add_argument("--min-reports", dest="min_reports", type=int, default=1)
    args = parser.parse_args()

    p = Path(args.summary)
    if not p.exists():
        print(f"Summary file not found: {p}")
        return 1
    obj = json.loads(p.read_text(encoding="utf-8"))
    # find if any candidate meets both gates
    winners = []
    for row in obj:
        # If report_count present, only use candidate if it meets min_reports
        report_count = int(row.get("report_count", 0) or 0)
        if report_count < args.min_reports:
            continue
        sharpe = row.get("sharpe_ratio") or row.get("mean_sharpe") or 0.0
        tr = row.get("total_return") or row.get("mean_total_return") or 0.0
        try:
            if float(sharpe) >= args.sharpe and float(tr) >= args.rett:
                winners.append(row)
        except Exception:
            continue
    if not winners:
        print("No candidate met gating thresholds")
        return 2
    print(f"Winners: {len(winners)}")
    for w in winners:
        print(w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
