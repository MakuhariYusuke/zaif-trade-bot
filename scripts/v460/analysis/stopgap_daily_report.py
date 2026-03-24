"""165# CLI: Stopgap Daily Health Report (162# P0 再現性固定対応).

Usage:
    .venv/Scripts/python.exe scripts/v460/analysis/stopgap_daily_report.py
    .venv/Scripts/python.exe scripts/v460/analysis/stopgap_daily_report.py --window 48 --json
    .venv/Scripts/python.exe scripts/v460/analysis/stopgap_daily_report.py --git-sha 955a78 --date-from 2026-02-25
    .venv/Scripts/python.exe scripts/v460/analysis/stopgap_daily_report.py --output reports/health.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections.abc import Sequence

# Ensure project root on sys.path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.v460.analysis.analysis_common import add_common_filter_args, add_output_args, write_json_output
from scripts.v460.lib.stopgap_health import (
    apply_filters,
    generate_health_report,
    load_fill_records,
    print_health_summary,
    serialize_health_report,
)

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="165# Stopgap Daily Health Report (162# P1 統合)",
    )
    add_common_filter_args(parser)
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
    add_output_args(parser)
    args = parser.parse_args(argv)

    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"ERROR: results dir not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    records = load_fill_records(results_path)
    if not records:
        print("ERROR: no fill records found", file=sys.stderr)
        sys.exit(1)

    # 162# P0: apply reproducibility filters
    records, filters = apply_filters(
        records,
        run_id=args.run_id,
        git_sha=args.git_sha,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    if not records:
        print("ERROR: no records after filter", file=sys.stderr)
        sys.exit(1)

    report = generate_health_report(
        records,
        window_hours=args.window,
        daily_limit=args.daily_limit,
        filters_applied=filters,
    )

    if args.json or args.output:
        report_dict = serialize_health_report(report)
        write_json_output(report_dict, args.output)
    else:
        print_health_summary(report)


if __name__ == "__main__":
    main()
