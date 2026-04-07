from __future__ import annotations

import argparse
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
from scripts.v460.lib.obi_mode import VALID_RANGING_OBI_MODES, compute_ranging_obi_multiplier

DEFAULT_JSON_OUTPUT = Path("analysis_results/710_obi_mode_comparison.json")
DEFAULT_CONFIG_PATH = Path("configs/v460/fill_test.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OBI mode comparison")
    add_common_filter_args(parser)
    add_output_args(parser)
    parser.add_argument(
        "--modes",
        default="linear,absolute,quadratic,excess",
        help="comma separated OBI modes",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="fill_test YAML path",
    )
    return parser


def _load_obi_config(config_path: Path) -> dict[str, float | str]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "factor": float(payload.get("ranging_obi_asymmetry_factor", 0.3)),
        "threshold": float(payload.get("ranging_obi_threshold", 0.1)),
        "mode": str(payload.get("ranging_obi_mode", "linear")),
    }


def _parse_modes(raw: str) -> list[str]:
    modes = [token.strip() for token in raw.split(",") if token.strip()]
    return [mode for mode in modes if mode in VALID_RANGING_OBI_MODES]


def _band(imbalance: float) -> str:
    if imbalance < -0.25:
        return "sell_heavy"
    if imbalance < 0.0:
        return "mild_sell"
    if imbalance < 0.25:
        return "mild_buy"
    return "buy_heavy"


def build_obi_mode_report(
    records: list[Record],
    *,
    modes: Sequence[str],
    factor: float,
    threshold: float,
) -> dict[str, object]:
    relevant = [
        record for record in records
        if record.get("side") in {"buy", "sell"} and safe_to_finite(record.get("orderbook_imbalance")) is not None
    ]
    mode_rows: dict[str, list[dict[str, float | str]]] = {mode: [] for mode in modes}
    for record in relevant:
        side = str(record["side"])
        imbalance = float(safe_to_finite(record.get("orderbook_imbalance")) or 0.0)
        pnl = safe_to_finite(record.get("post_fill_30s_pnl"))
        for mode in modes:
            mult = compute_ranging_obi_multiplier(
                1.0,
                side=side,
                imbalance=imbalance,
                threshold=threshold,
                factor=factor,
                mode=mode,
            )
            mode_rows[mode].append(
                {
                    "side": side,
                    "imbalance": imbalance,
                    "band": _band(imbalance),
                    "multiplier": mult,
                    "boost": mult - 1.0,
                    "pnl": float(pnl) if pnl is not None else 0.0,
                }
            )

    report: dict[str, object] = {
        "analysis": "710_obi_mode_comparison",
        "counts": {"records": len(records), "relevant": len(relevant)},
        "factor": factor,
        "threshold": threshold,
        "modes": list(modes),
        "results": {},
    }
    for mode, rows in mode_rows.items():
        band_summary: dict[str, dict[str, float | int | None]] = {}
        for band in ("sell_heavy", "mild_sell", "mild_buy", "buy_heavy"):
            band_rows = [row for row in rows if row["band"] == band]
            boosts = [float(row["boost"]) for row in band_rows]
            pnls = [float(row["pnl"]) for row in band_rows]
            band_summary[band] = {
                "count": len(band_rows),
                "avg_boost": (sum(boosts) / len(boosts)) if boosts else None,
                "avg_pnl": (sum(pnls) / len(pnls)) if pnls else None,
                "pnl_weighted_impact_proxy": (
                    sum(float(row["boost"]) * -float(row["pnl"]) for row in band_rows) / len(band_rows)
                    if band_rows else None
                ),
            }
        report["results"][mode] = {
            "by_band": band_summary,
            "by_side": {
                side: {
                    "avg_boost": (
                        sum(float(row["boost"]) for row in rows if row["side"] == side)
                        / max(1, sum(1 for row in rows if row["side"] == side))
                    ),
                }
                for side in ("buy", "sell")
            },
        }
    return report


def _render_summary(report: dict[str, object]) -> str:
    lines = ["obi mode comparison"]
    for mode, payload in report["results"].items():
        buy_heavy = payload["by_band"]["buy_heavy"]
        sell_heavy = payload["by_band"]["sell_heavy"]
        lines.append(
            f"{mode}: buy_heavy_boost={buy_heavy['avg_boost']} sell_heavy_boost={sell_heavy['avg_boost']}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg = _load_obi_config(args.config)
    records = load_and_filter_records(
        args.results_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        git_sha=args.git_sha,
        run_id=args.run_id,
        exit_on_empty=False,
    )
    report = build_obi_mode_report(
        records,
        modes=_parse_modes(args.modes),
        factor=float(cfg["factor"]),
        threshold=float(cfg["threshold"]),
    )
    if args.json:
        write_json_output(report, Path(args.output) if args.output else DEFAULT_JSON_OUTPUT)
    else:
        write_output(_render_summary(report), Path(args.output) if args.output else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
