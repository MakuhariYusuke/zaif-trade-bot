from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from scripts.v460.analysis.analysis_common import (
    add_output_args,
    add_standard_args,
    load_records_with_filters,
    write_json_output,
    write_output,
)
from scripts.v460.analysis.protocols import PROTOCOL_REGISTRY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分析 protocol CLI")
    parser.add_argument("--protocol", type=str, help="実行する protocol 名")
    parser.add_argument("--list", action="store_true", help="利用可能 protocol 一覧")
    parser.add_argument("--days", type=int, default=None, help="最新 N 日")
    parser.add_argument("--start", type=str, default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="終了日 YYYY-MM-DD")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results"),
        help="protocol 出力ディレクトリ",
    )
    add_standard_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list:
        lines = ["available protocols:"]
        for name, cls in sorted(PROTOCOL_REGISTRY.items()):
            lines.append(f"  {name}: {cls.description}")
        write_output("\n".join(lines))
        return 0

    if not args.protocol:
        parser.error("--protocol is required unless --list is used")

    protocol_cls = PROTOCOL_REGISTRY.get(args.protocol)
    if protocol_cls is None:
        parser.error(f"unknown protocol: {args.protocol}")

    records = load_records_with_filters(args)
    protocol = protocol_cls()
    result = protocol.execute(records, output_dir=args.output_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"protocol_{args.protocol}.json"
    text_path = output_dir / f"protocol_{args.protocol}.txt"
    write_json_output(result.json_payload, json_path)
    if args.json:
        write_json_output(result.json_payload, getattr(args, "output", None))
    else:
        write_output(result.text_report, text_path)
        write_output(result.text_report, getattr(args, "output", None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
