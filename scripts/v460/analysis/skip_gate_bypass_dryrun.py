from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import yaml

from scripts.v460.analysis.analysis_common import (
    DEFAULT_RESULTS_DIR,
    Record,
    add_common_filter_args,
    add_output_args,
    load_and_filter_records,
    safe_to_finite,
    write_json_output,
    write_output,
)

DEFAULT_JSON_OUTPUT = Path("analysis_results/710_skip_gate_bypass_dryrun.json")
DEFAULT_CONFIG_PATH = Path("configs/v460/fill_test.yaml")
DEFAULT_THRESHOLDS = (0.1, 0.2, 0.4, 0.6, 0.8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="skip_gate bypass dry-run analysis")
    add_common_filter_args(parser)
    add_output_args(parser)
    parser.add_argument(
        "--threshold-range",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="comma separated threshold candidates",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="fill_test YAML path",
    )
    return parser


def _parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _load_skip_gate_runtime(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return {
            "adaptive_threshold": None,
            "max_skip_rate": None,
            "bypass_mode": None,
            "bypass_mode_buy": None,
            "bypass_mode_sell": None,
        }
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    section = payload.get("skip_gate", {})
    if not isinstance(section, dict):
        return {}
    return {
        "adaptive_threshold": section.get("adaptive_threshold"),
        "max_skip_rate": section.get("max_skip_rate"),
        "bypass_mode": section.get("bypass_mode"),
        "bypass_mode_buy": section.get("bypass_mode_buy"),
        "bypass_mode_sell": section.get("bypass_mode_sell"),
    }


def _score_value(record: Record) -> float | None:
    as_prob = safe_to_finite(record.get("skip_gate_as_prob"))
    if as_prob is not None:
        return float(as_prob)
    score = safe_to_finite(record.get("skip_gate_score"))
    if score is not None:
        return float(score)
    return None


def _pnl_value(record: Record) -> float | None:
    value = safe_to_finite(record.get("post_fill_30s_pnl"))
    if value is None:
        return None
    return float(value)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _summarize_slice(records: list[Record], *, threshold: float, max_skip_rate: float | None) -> dict[str, object]:
    scored = [record for record in records if _score_value(record) is not None]
    blocked = [record for record in scored if (_score_value(record) or 0.0) >= threshold]
    passed = [record for record in scored if (_score_value(record) or 0.0) < threshold]
    blocked_pnl = [pnl for record in blocked if (pnl := _pnl_value(record)) is not None]
    passed_pnl = [pnl for record in passed if (pnl := _pnl_value(record)) is not None]
    block_rate = len(blocked) / len(scored) if scored else 0.0
    warnings: list[str] = []
    if max_skip_rate is not None and block_rate > max_skip_rate:
        warnings.append("exceeds_max_skip_rate")
    if block_rate > 0.5:
        warnings.append("block_rate_gt_0.5")
    return {
        "count": len(scored),
        "block_count": len(blocked),
        "block_rate": block_rate,
        "blocked_avg_pnl": _mean(blocked_pnl),
        "passed_avg_pnl": _mean(passed_pnl),
        "net_pnl_impact": (
            (_mean(passed_pnl) or 0.0) - (_mean(blocked_pnl) or 0.0)
            if blocked_pnl or passed_pnl
            else None
        ),
        "fill_rate_impact_estimation": 1.0 - block_rate * (1.0 - (max_skip_rate or 0.0)),
        "warnings": warnings,
    }


def build_bypass_dryrun_report(
    records: list[Record],
    *,
    thresholds: Sequence[float],
    runtime_cfg: dict[str, object],
) -> dict[str, object]:
    max_skip_rate_raw = safe_to_finite(runtime_cfg.get("max_skip_rate"))
    max_skip_rate = float(max_skip_rate_raw) if max_skip_rate_raw is not None else None
    side_values = ("buy", "sell")
    regime_values = ("ranging", "trending_up", "trending_down")

    threshold_report: dict[str, object] = {}
    for threshold in thresholds:
        key = f"{threshold:.3f}"
        side_breakdown = {
            side: _summarize_slice(
                [record for record in records if record.get("side") == side],
                threshold=threshold,
                max_skip_rate=max_skip_rate,
            )
            for side in side_values
        }
        regime_breakdown = {
            regime: _summarize_slice(
                [record for record in records if record.get("regime") == regime],
                threshold=threshold,
                max_skip_rate=max_skip_rate,
            )
            for regime in regime_values
        }
        threshold_report[key] = {
            "overall": _summarize_slice(records, threshold=threshold, max_skip_rate=max_skip_rate),
            "by_side": side_breakdown,
            "by_regime": regime_breakdown,
        }

    return {
        "analysis": "710_skip_gate_bypass_dryrun",
        "thresholds": list(thresholds),
        "runtime": runtime_cfg,
        "notes": {
            "adaptive_threshold_active": bool(runtime_cfg.get("adaptive_threshold")),
            "score_field_priority": ["skip_gate_as_prob", "skip_gate_score"],
        },
        "threshold_report": threshold_report,
    }


def _render_summary(report: dict[str, object]) -> str:
    lines = ["skip_gate bypass dry-run"]
    runtime = report["runtime"]
    lines.append(f"adaptive_threshold={runtime.get('adaptive_threshold')}")
    for threshold, payload in report["threshold_report"].items():
        overall = payload["overall"]
        lines.append(
            f"threshold={threshold} block_rate={overall['block_rate']:.3f} "
            f"blocked_avg={overall['blocked_avg_pnl']} passed_avg={overall['passed_avg_pnl']}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    thresholds = _parse_thresholds(args.threshold_range)
    runtime_cfg = _load_skip_gate_runtime(args.config)
    records = load_and_filter_records(
        args.results_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        git_sha=args.git_sha,
        run_id=args.run_id,
        exit_on_empty=False,
    )
    report = build_bypass_dryrun_report(
        records,
        thresholds=thresholds,
        runtime_cfg=runtime_cfg,
    )
    if args.json:
        write_json_output(report, Path(args.output) if args.output else DEFAULT_JSON_OUTPUT)
    else:
        write_output(_render_summary(report), Path(args.output) if args.output else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
