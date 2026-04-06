from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from scripts.v460.analysis.analysis_common import (
    add_common_filter_args,
    add_output_args,
    load_and_filter_records,
    write_json_output,
    write_output,
)
from scripts.v460.analysis.skip_gate_quality_analysis import build_skip_gate_quality_report

DEFAULT_JSON_OUTPUT = Path("analysis_results/708_skip_gate_quality.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="708# skip_gate quality analysis")
    add_common_filter_args(parser)
    add_output_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    records = load_and_filter_records(
        args.results_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        git_sha=args.git_sha,
        run_id=args.run_id,
    )
    report = build_skip_gate_quality_report(records)
    write_json_output(report.json_payload, DEFAULT_JSON_OUTPUT)
    if args.json:
        write_json_output(report.json_payload, getattr(args, "output", None))
    else:
        write_output(report.text_report, getattr(args, "output", None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
