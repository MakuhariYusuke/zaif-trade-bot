"""688# layered analysis protocol."""

from __future__ import annotations

import collections
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Protocol688Config:
    as_rate_warn_threshold: float = 0.25
    as_rate_alert_threshold: float = 0.35
    pnl_warn_threshold_bps: float = -0.5
    spread_bucket_edges: tuple[float, ...] = (1500.0, 2500.0, 3500.0)
    min_section_samples: int = 5


_DEFAULT_CONFIG = Protocol688Config()


def _record_str(record: Record, key: str, default: str = "") -> str:
    value = record.get(key)
    return value if isinstance(value, str) else default


def _record_bool(record: Record, key: str) -> bool:
    return bool(record.get(key))


def _record_optional_bool(record: Record, *keys: str) -> bool | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, bool):
            return value
    return None


def _record_float(record: Record, key: str) -> float | None:
    value = record.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _filled(records: list[Record]) -> list[Record]:
    return [record for record in records if _record_bool(record, "filled")]


def _pnls(records: list[Record]) -> list[float]:
    return [pnl for record in records if (pnl := get_pnl(record)) is not None]


def _avg(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _rate(num: int, den: int) -> float:
    return float(num / den * 100.0) if den else 0.0


def _spread_band(value: float | None, *, config: Protocol688Config) -> str:
    if value is None:
        return "unknown"
    edges = config.spread_bucket_edges
    if value < edges[0]:
        return f"0_{int(edges[0])}"
    if value < edges[1]:
        return f"{int(edges[0])}_{int(edges[1])}"
    if len(edges) > 2 and value < edges[2]:
        return f"{int(edges[1])}_{int(edges[2])}"
    return f"{int(edges[-1])}_plus"


def _section_side_regime_cross(records: list[Record]) -> list[str]:
    lines = ["## Side × Regime cross"]
    buckets: dict[tuple[str, str], list[Record]] = collections.defaultdict(list)
    for record in records:
        side = _record_str(record, "side", "unknown")
        regime = _record_str(record, "regime", "unknown")
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
        if _record_str(record, "side", "unknown") == "sell"
        and (_record_float(record, "skip_gate_hour_offset") or 0.0) > 0.0
    ]
    baseline = [
        record for record in records
        if _record_str(record, "side", "unknown") == "sell"
        and not (
            (_record_float(record, "skip_gate_hour_offset") or 0.0) > 0.0
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
        side_records = [record for record in records if _record_str(record, "side", "unknown") == side]
        filled = _filled(side_records)
        payload[side] = {
            "total": len(side_records),
            "filled": len(filled),
            "fill_rate_pct": _rate(len(filled), len(side_records)),
            "avg_pnl30_bps": _avg(_pnls(filled)),
            "adverse_selection_rate_pct": _rate(
                sum(1 for record in filled if _record_bool(record, "adverse_selected")),
                len(filled),
            ),
        }
    return payload


def _cancel_payload(records: list[Record]) -> dict[str, object]:
    cancels = [record for record in records if not _record_bool(record, "filled")]
    counts = collections.Counter(_record_str(record, "cancel_reason", "unknown") for record in cancels)
    return {
        "total": len(cancels),
        "reasons": dict(counts),
    }


def _nfq_payload(records: list[Record]) -> dict[str, object]:
    nfq_records = [
        record
        for record in records
        if not _record_bool(record, "filled")
        and _record_str(record, "cancel_reason", "") == "no_feasible_quote"
    ]
    regime_counts = collections.Counter(
        _record_str(record, "regime", "unknown") for record in nfq_records
    )
    side_counts = collections.Counter(
        _record_str(record, "side", "unknown") for record in nfq_records
    )
    total_cancels = sum(1 for record in records if not _record_bool(record, "filled"))
    return {
        "nfq_total": len(nfq_records),
        "cancel_total": total_cancels,
        "nfq_ratio": float(len(nfq_records) / total_cancels) if total_cancels else 0.0,
        "regime": dict(regime_counts),
        "side": dict(side_counts),
    }


def _adverse_selection_payload(
    records: list[Record],
    *,
    config: Protocol688Config,
) -> dict[str, object]:
    filled = _filled(records)
    adverse = [record for record in filled if _record_bool(record, "adverse_selected")]
    severe = [
        record for record in adverse
        if (pnl := get_pnl(record)) is not None and pnl <= config.pnl_warn_threshold_bps
    ]
    rate_pct = _rate(len(adverse), len(filled))
    severity = (
        "alert"
        if rate_pct >= config.as_rate_alert_threshold * 100.0
        else "warn"
        if rate_pct >= config.as_rate_warn_threshold * 100.0
        else "ok"
    )
    return {
        "count": len(adverse),
        "rate_pct": rate_pct,
        "avg_pnl30_bps": _avg(_pnls(adverse)),
        "severe_count": len(severe),
        "severity": severity,
    }


def _spread_payload(
    records: list[Record],
    *,
    config: Protocol688Config,
) -> dict[str, object]:
    buckets: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[Record]] = collections.defaultdict(list)
    for record in records:
        spread_value = record.get("spread_at_order")
        spread = float(spread_value) if isinstance(spread_value, (int, float)) else None
        grouped[_spread_band(spread, config=config)].append(record)
    for name, group in sorted(grouped.items()):
        filled = _filled(group)
        buckets[name] = {
            "total": len(group),
            "filled": len(filled),
            "fill_rate_pct": _rate(len(filled), len(group)),
            "avg_pnl30_bps": _avg(_pnls(filled)),
            "enough_samples": len(group) >= config.min_section_samples,
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
        grouped[_record_str(record, "git_sha", "?")].append(record)
    sha_rows: list[tuple[str, dict[str, object]]] = []
    for sha, group in grouped.items():
        filled = _filled(group)
        adverse_count = sum(
            1
            for record in filled
            if _record_optional_bool(record, "adverse_selected", "is_adverse") is True
        )
        avg_pnl30_bps = _avg(_pnls(filled))
        total_pnl_contribution_bps = float((avg_pnl30_bps or 0.0) * len(filled))
        sha_rows.append(
            (
                sha,
                {
                    "total": len(group),
                    "filled": len(filled),
                    "avg_pnl30_bps": avg_pnl30_bps,
                    "adverse_selection_count": adverse_count,
                    "adverse_selection_rate_pct": _rate(adverse_count, len(filled)),
                    "total_pnl_contribution_bps": total_pnl_contribution_bps,
                },
            )
        )
    sha_rows.sort(
        key=lambda item: (
            float(item[1]["total_pnl_contribution_bps"]),
            item[0],
        )
    )
    return dict(sha_rows)


def _regime_payload(records: list[Record]) -> dict[str, object]:
    grouped: dict[str, list[Record]] = collections.defaultdict(list)
    for record in records:
        grouped[_record_str(record, "regime", "unknown")].append(record)
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
        key = f"{_record_str(record, 'side', 'unknown')}::{_record_str(record, 'regime', 'unknown')}"
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
        if _record_str(record, "side", "unknown") == "sell"
        and (_record_float(record, "skip_gate_hour_offset") or 0.0) > 0.0
    ]
    baseline = [
        record for record in records
        if _record_str(record, "side", "unknown") == "sell"
        and not (
            (_record_float(record, "skip_gate_hour_offset") or 0.0) > 0.0
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

    def __init__(self, config: Protocol688Config = _DEFAULT_CONFIG) -> None:
        self._config = config

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
            "config": {
                "as_rate_warn_threshold": self._config.as_rate_warn_threshold,
                "as_rate_alert_threshold": self._config.as_rate_alert_threshold,
                "pnl_warn_threshold_bps": self._config.pnl_warn_threshold_bps,
                "spread_bucket_edges": list(self._config.spread_bucket_edges),
                "min_section_samples": self._config.min_section_samples,
            },
            "basic": _basic_payload(records),
            "side": _side_payload(records),
            "nfq": _nfq_payload(records),
            "cancels": _cancel_payload(records),
            "adverse_selection": _adverse_selection_payload(records, config=self._config),
            "spread": _spread_payload(records, config=self._config),
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
