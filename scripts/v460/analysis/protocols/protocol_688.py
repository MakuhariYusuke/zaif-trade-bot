"""688# layered analysis protocol."""

from __future__ import annotations

import collections
from datetime import timezone
from pathlib import Path

import numpy as np

from scripts.v460.analysis.analyze_fill_logs import (
    section_adverse_selection,
    section_basic,
    section_cancel,
    section_git_sha,
    section_hourly,
    section_regime,
    section_side,
    section_spread,
)
from scripts.v460.analysis.analysis_common import Record, get_pnl, record_to_utc_hour
from . import AnalysisProtocol, ProtocolResult, register_protocol


def _filled(records: list[Record]) -> list[Record]:
    return [record for record in records if record.get("filled")]


def _pnls(records: list[Record]) -> list[float]:
    return [pnl for record in records if (pnl := get_pnl(record)) is not None]


def _avg(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _rate(num: int, den: int) -> float:
    return float(num / den * 100.0) if den else 0.0


def _spread_band(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 1500.0:
        return "0_1500"
    if value < 2500.0:
        return "1500_2500"
    return "2500_plus"


def _section_side_regime_cross(records: list[Record]) -> list[str]:
    lines = ["## Side × Regime cross"]
    buckets: dict[tuple[str, str], list[Record]] = collections.defaultdict(list)
    for record in records:
        side = str(record.get("side", "unknown"))
        regime = str(record.get("regime", "unknown"))
        buckets[(side, regime)].append(record)
    for (side, regime), group in sorted(buckets.items()):
        filled = _filled(group)
        lines.append(
            f"  {side}/{regime}: total={len(group)}, filled={len(filled)}, "
            f"fill_rate={_rate(len(filled), len(group)):.1f}%, avg_pnl30={(_avg(_pnls(filled)) or 0.0):+.2f}bps"
        )
    lines.append("")
    return lines


def _section_sell_hour_boost_effectiveness(records: list[Record]) -> list[str]:
    lines = ["## Sell hour boost effectiveness"]
    boosted = [
        record for record in records
        if record.get("side") == "sell"
        and isinstance(record.get("skip_gate_hour_offset"), (int, float))
        and float(record["skip_gate_hour_offset"]) > 0.0
    ]
    baseline = [
        record for record in records
        if record.get("side") == "sell"
        and not (
            isinstance(record.get("skip_gate_hour_offset"), (int, float))
            and float(record["skip_gate_hour_offset"]) > 0.0
        )
    ]
    lines.append(
        "  boosted: "
        f"n={len(boosted)}, avg_pnl30={(_avg(_pnls(_filled(boosted))) or 0.0):+.2f}bps"
    )
    lines.append(
        "  baseline: "
        f"n={len(baseline)}, avg_pnl30={(_avg(_pnls(_filled(baseline))) or 0.0):+.2f}bps"
    )
    lines.append("")
    return lines


def _basic_payload(records: list[Record]) -> dict[str, object]:
    filled = _filled(records)
    return {
        "total": len(records),
        "filled": len(filled),
        "fill_rate_pct": _rate(len(filled), len(records)),
        "avg_pnl30_bps": _avg(_pnls(filled)),
    }


def _side_payload(records: list[Record]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for side in ("buy", "sell"):
        side_records = [record for record in records if record.get("side") == side]
        filled = _filled(side_records)
        payload[side] = {
            "total": len(side_records),
            "filled": len(filled),
            "fill_rate_pct": _rate(len(filled), len(side_records)),
            "avg_pnl30_bps": _avg(_pnls(filled)),
            "adverse_selection_rate_pct": _rate(
                sum(1 for record in filled if record.get("adverse_selected")),
                len(filled),
            ),
        }
    return payload


def _cancel_payload(records: list[Record]) -> dict[str, object]:
    cancels = [record for record in records if not record.get("filled")]
    counts = collections.Counter(str(record.get("cancel_reason", "unknown")) for record in cancels)
    return {
        "total": len(cancels),
        "reasons": dict(counts),
    }


def _adverse_selection_payload(records: list[Record]) -> dict[str, object]:
    filled = _filled(records)
    adverse = [record for record in filled if record.get("adverse_selected")]
    severe = [
        record for record in adverse
        if (pnl := get_pnl(record)) is not None and pnl <= -10.0
    ]
    return {
        "count": len(adverse),
        "rate_pct": _rate(len(adverse), len(filled)),
        "avg_pnl30_bps": _avg(_pnls(adverse)),
        "severe_count": len(severe),
    }


def _spread_payload(records: list[Record]) -> dict[str, object]:
    buckets: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[Record]] = collections.defaultdict(list)
    for record in records:
        grouped[_spread_band(getattr(record, "get", lambda *_: None)("spread_at_order"))].append(record)
    for name, group in sorted(grouped.items()):
        filled = _filled(group)
        buckets[name] = {
            "total": len(group),
            "filled": len(filled),
            "fill_rate_pct": _rate(len(filled), len(group)),
            "avg_pnl30_bps": _avg(_pnls(filled)),
        }
    return buckets


def _hourly_payload(records: list[Record]) -> dict[str, object]:
    by_hour: dict[int, list[float]] = collections.defaultdict(list)
    for record in _filled(records):
        hour = record_to_utc_hour(record)
        pnl = get_pnl(record)
        if hour is None or pnl is None:
            continue
        by_hour[hour].append(pnl)
    return {
        f"{hour:02d}": {
            "count": len(values),
            "avg_pnl30_bps": _avg(values),
            "profitable_rate_pct": _rate(sum(1 for value in values if value > 0.0), len(values)),
        }
        for hour, values in sorted(by_hour.items())
    }


def _sha_payload(records: list[Record]) -> dict[str, object]:
    grouped: dict[str, list[Record]] = collections.defaultdict(list)
    for record in records:
        grouped[str(record.get("git_sha", "?"))].append(record)
    return {
        sha: {
            "total": len(group),
            "filled": len(_filled(group)),
            "avg_pnl30_bps": _avg(_pnls(_filled(group))),
        }
        for sha, group in sorted(grouped.items())
    }


def _regime_payload(records: list[Record]) -> dict[str, object]:
    grouped: dict[str, list[Record]] = collections.defaultdict(list)
    for record in records:
        grouped[str(record.get("regime", "unknown"))].append(record)
    return {
        regime: {
            "total": len(group),
            "filled": len(_filled(group)),
            "fill_rate_pct": _rate(len(_filled(group)), len(group)),
            "avg_pnl30_bps": _avg(_pnls(_filled(group))),
        }
        for regime, group in sorted(grouped.items())
    }


def _side_regime_payload(records: list[Record]) -> dict[str, object]:
    grouped: dict[str, list[Record]] = collections.defaultdict(list)
    for record in records:
        key = f"{record.get('side', 'unknown')}::{record.get('regime', 'unknown')}"
        grouped[key].append(record)
    return {
        key: {
            "total": len(group),
            "filled": len(_filled(group)),
            "fill_rate_pct": _rate(len(_filled(group)), len(group)),
            "avg_pnl30_bps": _avg(_pnls(_filled(group))),
        }
        for key, group in sorted(grouped.items())
    }


def _sell_hour_boost_payload(records: list[Record]) -> dict[str, object]:
    boosted = [
        record for record in records
        if record.get("side") == "sell"
        and isinstance(record.get("skip_gate_hour_offset"), (int, float))
        and float(record["skip_gate_hour_offset"]) > 0.0
    ]
    baseline = [
        record for record in records
        if record.get("side") == "sell"
        and not (
            isinstance(record.get("skip_gate_hour_offset"), (int, float))
            and float(record["skip_gate_hour_offset"]) > 0.0
        )
    ]
    return {
        "boosted": {
            "count": len(boosted),
            "avg_pnl30_bps": _avg(_pnls(_filled(boosted))),
        },
        "baseline": {
            "count": len(baseline),
            "avg_pnl30_bps": _avg(_pnls(_filled(baseline))),
        },
    }


@register_protocol
class Protocol688(AnalysisProtocol):
    protocol_name = "688"
    description = "688# layered NFQ/AS/spread/hour/regime analysis"

    def execute(
        self,
        records: list[Record],
        *,
        output_dir: Path | None = None,
    ) -> ProtocolResult:
        del output_dir
        warnings: list[str] = []
        if not records:
            warnings.append("no records matched the requested filters")

        sections = [
            section_basic(records),
            section_side(records),
            section_cancel(records),
            section_adverse_selection(records),
            section_spread(records),
            section_hourly(records),
            section_git_sha(records),
            section_regime(records),
            _section_side_regime_cross(records),
            _section_sell_hour_boost_effectiveness(records),
        ]
        text_report = "\n".join(
            line
            for section in sections
            for line in section
        ).strip()

        json_payload: dict[str, object] = {
            "protocol": "688",
            "warnings": warnings,
            "basic": _basic_payload(records),
            "side": _side_payload(records),
            "nfq": _cancel_payload(records),
            "adverse_selection": _adverse_selection_payload(records),
            "spread": _spread_payload(records),
            "hour": _hourly_payload(records),
            "sha": _sha_payload(records),
            "regime": _regime_payload(records),
            "side_regime_cross": _side_regime_payload(records),
            "sell_hour_offset_boost": _sell_hour_boost_payload(records),
        }
        return ProtocolResult(
            text_report=text_report,
            json_payload=json_payload,
            warnings=warnings,
        )
