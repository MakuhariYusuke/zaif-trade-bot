from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

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
from scripts.v460.ml.calibration_batch import load_calibration_state

DEFAULT_JSON_OUTPUT = Path("analysis_results/710_entry_gate_ev_distribution.json")
DEFAULT_CALIBRATION_PATH = Path("models/v460/entry_gate_calibration.json")
THRESHOLDS = (-0.5, -1.0, -1.5, -1.8, -2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="entry-gate EV distribution analysis")
    add_common_filter_args(parser)
    add_output_args(parser)
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="new calibration map path",
    )
    parser.add_argument(
        "--baseline-calibration-path",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="baseline calibration map path",
    )
    return parser


def _calc_ev(stats: dict[str, float], *, probability_mode: str = "lcb") -> float:
    if probability_mode == "ucb":
        p_win = stats.get("p_win_ucb", stats.get("p_win_mean", 0.5))
    elif probability_mode == "mean":
        p_win = stats.get("p_win_mean", 0.5)
    else:
        p_win = stats.get("p_win_lcb", 0.0)
    avg_win = stats.get("avg_win", 0.0)
    avg_loss = stats.get("avg_loss", 0.0)
    return float(p_win * avg_win - (1.0 - p_win) * avg_loss)


def _record_regime(record: Record) -> str:
    value = record.get("regime")
    return value if isinstance(value, str) and value else "unknown"


def _record_side(record: Record) -> str:
    value = record.get("side")
    return value if isinstance(value, str) and value else "unknown"


def _action_for_side(side: str) -> float:
    return 0.3 if side == "buy" else -0.3


def _evaluate_records(records: list[Record], calibration_path: Path) -> list[dict[str, object]]:
    calibration = load_calibration_state(calibration_path)
    if calibration is None:
        return []
    evaluated: list[dict[str, object]] = []
    for record in records:
        side = _record_side(record)
        regime = _record_regime(record)
        stats_bundle = calibration.get_stats(regime, _action_for_side(side))
        fallback_stats = stats_bundle["fallback"]
        ev = _calc_ev(fallback_stats)
        pnl = safe_to_finite(record.get("post_fill_30s_pnl"))
        evaluated.append(
            {
                "side": side,
                "regime": regime,
                "ev": ev,
                "pnl": float(pnl) if pnl is not None else None,
            }
        )
    return evaluated


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p5": None, "p25": None, "median": None, "p75": None, "p95": None}
    values = sorted(values)
    last = len(values) - 1
    def _at(q: float) -> float:
        idx = int(round(q * last))
        return float(values[idx])
    return {
        "p5": _at(0.05),
        "p25": _at(0.25),
        "median": _at(0.5),
        "p75": _at(0.75),
        "p95": _at(0.95),
    }


def _threshold_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for threshold in THRESHOLDS:
        blocked = [row for row in rows if float(row["ev"]) < threshold]
        passed = [row for row in rows if float(row["ev"]) >= threshold]
        blocked_pnl = [float(row["pnl"]) for row in blocked if row["pnl"] is not None]
        passed_pnl = [float(row["pnl"]) for row in passed if row["pnl"] is not None]
        output[str(threshold)] = {
            "block_count": len(blocked),
            "pass_count": len(passed),
            "block_rate": len(blocked) / len(rows) if rows else 0.0,
            "pass_rate": len(passed) / len(rows) if rows else 0.0,
            "blocked_avg_pnl": (sum(blocked_pnl) / len(blocked_pnl)) if blocked_pnl else None,
            "passed_avg_pnl": (sum(passed_pnl) / len(passed_pnl)) if passed_pnl else None,
        }
    return output


def build_entry_gate_ev_distribution_report(
    records: list[Record],
    *,
    calibration_path: Path,
    baseline_calibration_path: Path,
) -> dict[str, object]:
    current_rows = _evaluate_records(records, calibration_path)
    baseline_rows = _evaluate_records(records, baseline_calibration_path)
    current_evs = [float(row["ev"]) for row in current_rows]
    baseline_evs = [float(row["ev"]) for row in baseline_rows]
    return {
        "analysis": "710_entry_gate_ev_distribution",
        "counts": {"records": len(records), "evaluated": len(current_rows)},
        "calibration_path": str(calibration_path),
        "baseline_calibration_path": str(baseline_calibration_path),
        "current_distribution": _percentiles(current_evs),
        "baseline_distribution": _percentiles(baseline_evs),
        "thresholds": {
            "current": _threshold_summary(current_rows),
            "baseline": _threshold_summary(baseline_rows),
        },
        "by_side": {
            side: _percentiles([float(row["ev"]) for row in current_rows if row["side"] == side])
            for side in ("buy", "sell")
        },
        "by_regime": {
            regime: _percentiles([float(row["ev"]) for row in current_rows if row["regime"] == regime])
            for regime in ("ranging", "trending_up", "trending_down", "unknown")
        },
        "warning": (
            "ev_distribution_still_concentrated"
            if current_evs and _percentiles(current_evs)["p75"] == _percentiles(current_evs)["p25"]
            else None
        ),
    }


def _render_summary(report: dict[str, object]) -> str:
    current = report["current_distribution"]
    return "\n".join(
        [
            "entry_gate ev distribution",
            f"records={report['counts']['records']} evaluated={report['counts']['evaluated']}",
            f"p25={current['p25']} median={current['median']} p75={current['p75']}",
            f"warning={report['warning']}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    records = load_and_filter_records(
        args.results_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        git_sha=args.git_sha,
        run_id=args.run_id,
        exit_on_empty=False,
    )
    report = build_entry_gate_ev_distribution_report(
        records,
        calibration_path=args.calibration_path,
        baseline_calibration_path=args.baseline_calibration_path,
    )
    if args.json:
        write_json_output(report, Path(args.output) if args.output else DEFAULT_JSON_OUTPUT)
    else:
        write_output(_render_summary(report), Path(args.output) if args.output else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
