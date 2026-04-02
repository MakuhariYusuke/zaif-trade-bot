"""695# Regime × spread adverse-selection deep dive."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from scripts.v460.analysis.analysis_common import Record, get_pnl


def build_regime_as_deep_dive_section(records: list[Record]) -> dict[str, object]:
    filled = [record for record in records if bool(record.get("filled"))]
    regime_spread: dict[str, dict[str, dict[str, float | int | None]]] = defaultdict(dict)
    ranging_buckets: dict[str, list[Record]] = defaultdict(list)
    veto_overlap = 0

    for record in filled:
        regime = _as_str(record.get("regime")) or "unknown"
        bucket = _spread_bucket(_as_float(record.get("spread_at_order")))
        bucket_records = [
            candidate
            for candidate in filled
            if (_as_str(candidate.get("regime")) or "unknown") == regime
            and _spread_bucket(_as_float(candidate.get("spread_at_order"))) == bucket
        ]
        regime_spread[regime][bucket] = _bucket_payload(bucket_records)
        if regime == "ranging":
            ranging_buckets[bucket].append(record)
        if record.get("trend_5s_guard_action") == "veto" or record.get("cancel_reason") == "trend_5s_sell_guard_veto":
            veto_overlap += 1

    return {
        "regime_spread_crosstab": regime_spread,
        "ranging_spread_attribution": {
            bucket: _bucket_payload(bucket_records)
            for bucket, bucket_records in sorted(ranging_buckets.items())
        },
        "trend_5s_veto_overlap_count": veto_overlap,
        "bypass_segment": _segment_payload(
            [record for record in records if bool(record.get("skip_gate_bypassed"))]
        ),
        "non_bypass_segment": _segment_payload(
            [record for record in records if not bool(record.get("skip_gate_bypassed"))]
        ),
    }


def _segment_payload(records: list[Record]) -> dict[str, float | int | None]:
    filled = [record for record in records if bool(record.get("filled"))]
    pnls = [pnl for record in filled if (pnl := get_pnl(record)) is not None]
    return {
        "total": len(records),
        "filled": len(filled),
        "fill_rate_pct": _rate_pct(len(filled), len(records)),
        "avg_pnl30_bps": float(mean(pnls)) if pnls else None,
        "adverse_selection_rate_pct": _rate_pct(
            sum(1 for record in filled if bool(record.get("adverse_selected"))),
            len(filled),
        ),
    }


def _bucket_payload(records: list[Record]) -> dict[str, float | int | None]:
    filled = [record for record in records if bool(record.get("filled"))]
    pnls = [pnl for record in filled if (pnl := get_pnl(record)) is not None]
    return {
        "total": len(records),
        "filled": len(filled),
        "fill_rate_pct": _rate_pct(len(filled), len(records)),
        "avg_pnl30_bps": float(mean(pnls)) if pnls else None,
        "adverse_selection_rate_pct": _rate_pct(
            sum(1 for record in filled if bool(record.get("adverse_selected"))),
            len(filled),
        ),
    }


def _spread_bucket(spread_jpy: float | None) -> str:
    if spread_jpy is None:
        return "unknown"
    if spread_jpy < 1500.0:
        return "0_1500"
    if spread_jpy < 2500.0:
        return "1500_2500"
    if spread_jpy < 3500.0:
        return "2500_3500"
    return "3500_plus"


def _rate_pct(num: int, den: int) -> float:
    return float(num / den * 100.0) if den else 0.0


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None

