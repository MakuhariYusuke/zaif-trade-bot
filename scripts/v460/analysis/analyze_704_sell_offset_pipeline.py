from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Sequence

import yaml

from scripts.v460.analysis.analysis_common import (
    DEFAULT_RESULTS_DIR,
    Record,
    add_common_filter_args,
    add_output_args,
    load_and_filter_records,
    record_to_utc_hour,
    write_json_output,
    write_output,
)
from ztb.utils.safety import safe_to_finite

DEFAULT_JSON_OUTPUT = Path("analysis_results/704_sell_offset_pipeline.json")
DEFAULT_CONFIG_PATH = Path("configs/v460/fill_test.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="704# sell offset pipeline analysis")
    add_common_filter_args(parser)
    add_output_args(parser)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="fill_test YAML path",
    )
    return parser


def _parse_offset_stages(raw: object) -> dict[str, float]:
    payload = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(payload, dict):
        return {}

    parsed: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        finite = safe_to_finite(value)
        if finite is None:
            continue
        parsed[key] = float(finite)
    return parsed


def _extract_offset_stages(record: Record) -> dict[str, float]:
    executor = _parse_offset_stages(record.get("executor_offset_stages"))
    if executor:
        return executor
    return _parse_offset_stages(record.get("offset_stages"))


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    dx = [x - x_mean for x in xs]
    dy = [y - y_mean for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom <= 0.0:
        return None
    return float(sum(x * y for x, y in zip(dx, dy)) / denom)


def _bucket_counts(values: Sequence[float], *, edges: Sequence[float]) -> dict[str, int]:
    if not edges:
        return {}
    counts = [0] * (len(edges) + 1)
    for value in values:
        placed = False
        for idx, edge in enumerate(edges):
            if value < edge:
                counts[idx] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    labels: list[str] = [f"<{edges[0]:.2f}"]
    labels.extend(
        f"{edges[idx - 1]:.2f}-{edge:.2f}" for idx, edge in enumerate(edges[1:], start=1)
    )
    labels.append(f">={edges[-1]:.2f}")
    return {label: count for label, count in zip(labels, counts)}


def _load_sell_boost_hours(config_path: Path) -> set[int]:
    if not config_path.exists():
        return set()
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        return set()
    raw_hours = payload.get("sell_hour_offset_boost")
    if not isinstance(raw_hours, dict):
        return set()
    resolved: set[int] = set()
    for key, value in raw_hours.items():
        finite = safe_to_finite(value)
        if finite is None or finite <= 1.0:
            continue
        try:
            resolved.add(int(key))
        except (TypeError, ValueError):
            continue
    return resolved


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _analyze(records: list[Record], *, sell_boost_hours: set[int]) -> dict[str, object]:
    fills = [record for record in records if bool(record.get("filled"))]
    sell_fills = [record for record in fills if record.get("side") == "sell"]
    buy_fills = [record for record in fills if record.get("side") == "buy"]

    sell_offsets = [
        float(value)
        for record in sell_fills
        if (value := safe_to_finite(record.get("effective_offset_used"))) is not None
    ]
    buy_offsets = [
        float(value)
        for record in buy_fills
        if (value := safe_to_finite(record.get("effective_offset_used"))) is not None
    ]
    sell_captures = [
        float(value)
        for record in sell_fills
        if (value := safe_to_finite(record.get("spread_capture_bps"))) is not None
    ]
    buy_captures = [
        float(value)
        for record in buy_fills
        if (value := safe_to_finite(record.get("spread_capture_bps"))) is not None
    ]

    paired_sell: list[tuple[float, float]] = []
    for record in sell_fills:
        offset = safe_to_finite(record.get("effective_offset_used"))
        capture = safe_to_finite(record.get("spread_capture_bps"))
        if offset is None or capture is None:
            continue
        paired_sell.append((float(offset), float(capture)))

    stage_totals: dict[str, float] = {}
    stage_counts: dict[str, int] = {}
    for record in sell_fills:
        for key, value in _extract_offset_stages(record).items():
            stage_totals[key] = stage_totals.get(key, 0.0) + value
            stage_counts[key] = stage_counts.get(key, 0) + 1

    boosted_captures: list[float] = []
    non_boosted_captures: list[float] = []
    for record in sell_fills:
        capture = safe_to_finite(record.get("spread_capture_bps"))
        hour = record_to_utc_hour(record)
        if capture is None or hour is None:
            continue
        if hour in sell_boost_hours:
            boosted_captures.append(float(capture))
        else:
            non_boosted_captures.append(float(capture))

    correlation = _pearson_correlation(
        [offset for offset, _capture in paired_sell],
        [capture for _offset, capture in paired_sell],
    )
    stage_average = {
        key: stage_totals[key] / stage_counts[key]
        for key in sorted(stage_totals)
        if stage_counts[key] > 0
    }

    recommendation = "insufficient_data"
    if correlation is not None and correlation > 0.2 and (_mean(sell_captures) or -1.0) < 0.0:
        recommendation = "C: raise sell_hour_offset_boost baseline"
    elif (_mean(sell_offsets) or 0.0) < 0.03:
        recommendation = "A: raise global min_offset_ratio"
    elif (_mean(sell_captures) or 0.0) < 0.0:
        recommendation = "B: add sell-only offset floor"

    return {
        "analysis": "704_sell_offset_pipeline",
        "counts": {
            "records": len(records),
            "fills": len(fills),
            "sell_fills": len(sell_fills),
            "buy_fills": len(buy_fills),
            "paired_sell_offset_capture": len(paired_sell),
        },
        "effective_offset_ratio_distribution": {
            "sell_mean": _mean(sell_offsets),
            "buy_mean": _mean(buy_offsets),
            "sell_histogram": _bucket_counts(sell_offsets, edges=(0.01, 0.03, 0.05, 0.08)),
            "buy_histogram": _bucket_counts(buy_offsets, edges=(0.01, 0.03, 0.05, 0.08)),
        },
        "spread_capture_bps": {
            "sell_mean": _mean(sell_captures),
            "buy_mean": _mean(buy_captures),
            "sell_hour_offset_boost_hours": sorted(sell_boost_hours),
            "boost_hours_mean": _mean(boosted_captures),
            "non_boost_hours_mean": _mean(non_boosted_captures),
        },
        "offset_stage_contribution": {
            "mean_by_stage": stage_average,
            "count_by_stage": stage_counts,
        },
        "spread_capture_vs_effective_offset_ratio": {
            "correlation": correlation,
        },
        "recommendation": recommendation,
    }


def _render_summary(result: dict[str, object]) -> str:
    counts = result["counts"]
    offset_dist = result["effective_offset_ratio_distribution"]
    capture = result["spread_capture_bps"]
    corr = result["spread_capture_vs_effective_offset_ratio"]
    return "\n".join(
        [
            "704# sell offset pipeline analysis",
            f"- sell fills: {counts['sell_fills']}",
            f"- buy fills: {counts['buy_fills']}",
            f"- mean effective_offset_used sell/buy: {offset_dist['sell_mean']} / {offset_dist['buy_mean']}",
            f"- mean spread_capture_bps sell/buy: {capture['sell_mean']} / {capture['buy_mean']}",
            f"- boost-hours capture mean: {capture['boost_hours_mean']}",
            f"- non-boost-hours capture mean: {capture['non_boost_hours_mean']}",
            f"- corr(offset, capture) sell: {corr['correlation']}",
            f"- recommendation: {result['recommendation']}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    results_dir = Path(args.results_dir)
    if results_dir.exists():
        records = load_and_filter_records(
            str(results_dir),
            date_from=args.date_from,
            date_to=args.date_to,
            git_sha=args.git_sha,
            run_id=args.run_id,
            include_emergency=True,
            exit_on_empty=False,
        )
    else:
        records = []

    result = _analyze(records, sell_boost_hours=_load_sell_boost_hours(args.config))
    summary = _render_summary(result)

    json_output_path = Path(args.output) if args.json and args.output else DEFAULT_JSON_OUTPUT
    write_json_output(result, json_output_path)
    if args.json and args.output is None:
        write_json_output(result, None)
    if not args.json:
        write_output(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
