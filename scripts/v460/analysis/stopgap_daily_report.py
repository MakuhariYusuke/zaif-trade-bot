"""165# CLI: Stopgap Daily Health Report.

Usage:
    .venv\Scripts\python.exe scripts/v460/analysis/stopgap_daily_report.py
    .venv\Scripts\python.exe scripts/v460/analysis/stopgap_daily_report.py --window 48 --json
    .venv\Scripts\python.exe scripts/v460/analysis/stopgap_daily_report.py --output reports/health.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure project root on sys.path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.v460.lib.stopgap_health import (
    DailyHealthReport,
    generate_health_report,
    load_fill_records,
    print_health_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="165# Stopgap Daily Health Report (162# P1 統合)",
    )
    parser.add_argument(
        "--results-dir",
        default="results/v460/fill_test",
        help="fill_records ディレクトリ (default: results/v460/fill_test)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=168,
        help="評価ウィンドウ (hours, default: 168=7d)",
    )
    parser.add_argument(
        "--daily-limit",
        type=int,
        default=7,
        help="日次出力の最大日数 (default: 7)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON 出力のみ",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON 出力先ファイルパス (None=stdout)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"ERROR: results dir not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    records = load_fill_records(results_path)
    if not records:
        print("ERROR: no fill records found", file=sys.stderr)
        sys.exit(1)

    report = generate_health_report(
        records,
        window_hours=args.window,
        daily_limit=args.daily_limit,
    )

    if args.json or args.output:
        report_dict = asdict(report)
        json_str = json.dumps(report_dict, ensure_ascii=False, indent=2, default=str)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json_str, encoding="utf-8")
            print(f"Report saved to {out}")
        else:
            print(json_str)
    else:
        print_health_summary(report)


if __name__ == "__main__":
    main()
