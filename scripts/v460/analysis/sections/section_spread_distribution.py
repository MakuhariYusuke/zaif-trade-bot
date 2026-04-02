from __future__ import annotations

from statistics import mean

from scripts.v460.analysis.analysis_common import Record


def build_spread_distribution_section(
    records: list[Record],
    *,
    threshold_candidates_bps: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0),
) -> dict[str, object]:
    spread_bps_values = [value for record in records if (value := _record_spread_bps(record)) is not None]
    adverse_values = [
        value
        for record in records
        if (value := _record_spread_bps(record)) is not None and bool(record.get("adverse_selected"))
    ]
    quantiles = {
        "p10": _percentile(spread_bps_values, 10),
        "p25": _percentile(spread_bps_values, 25),
        "p50": _percentile(spread_bps_values, 50),
        "p75": _percentile(spread_bps_values, 75),
        "p90": _percentile(spread_bps_values, 90),
    }
    threshold_impact = {
        f"{candidate:.1f}": {
            "count": sum(1 for value in spread_bps_values if value < candidate),
            "rate_pct": _rate_pct(
                sum(1 for value in spread_bps_values if value < candidate),
                len(spread_bps_values),
            ),
        }
        for candidate in threshold_candidates_bps
    }
    return {
        "count": len(spread_bps_values),
        "avg_spread_bps": float(mean(spread_bps_values)) if spread_bps_values else None,
        "avg_adverse_selected_spread_bps": (
            float(mean(adverse_values)) if adverse_values else None
        ),
        "quantiles_bps": quantiles,
        "threshold_impact": threshold_impact,
    }


def _record_spread_bps(record: Record) -> float | None:
    direct = record.get("spread_bps")
    if isinstance(direct, (int, float)):
        return float(direct)
    spread = record.get("spread_at_order")
    if not isinstance(spread, (int, float)):
        return None
    for mid_key in ("mid_at_order", "mid_at_fill"):
        mid = record.get(mid_key)
        if isinstance(mid, (int, float)) and mid > 0:
            return float(spread) / float(mid) * 10_000.0
    return None


def _percentile(values: list[float], q: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _rate_pct(num: int, den: int) -> float:
    return float(num / den * 100.0) if den else 0.0
