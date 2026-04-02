"""695# Trend-5s veto counterfactual analysis."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean

from scripts.v460.analysis.analysis_common import Record, get_pnl


def build_trend_5s_counterfactual_section(records: list[Record]) -> dict[str, object]:
    """Analyze vetoed sells against passed boost controls."""
    sorted_records = sorted(records, key=lambda record: float(record.get("timestamp", 0.0)))
    veto_records = [
        record for record in sorted_records
        if record.get("cancel_reason") == "trend_5s_sell_guard_veto"
    ]
    control_records = [
        record for record in sorted_records
        if record.get("side") == "sell"
        and bool(record.get("filled"))
        and record.get("trend_5s_guard_action") == "boost"
    ]

    veto_pnl_30 = _collect_counterfactual_pnls(veto_records, sorted_records, horizon_sec=30.0)
    veto_pnl_60 = _collect_counterfactual_pnls(veto_records, sorted_records, horizon_sec=60.0)
    veto_pnl_120 = _collect_counterfactual_pnls(veto_records, sorted_records, horizon_sec=120.0)
    control_pnls = [pnl for record in control_records if (pnl := get_pnl(record)) is not None]

    control_positive_mean = _mean([value for value in control_pnls if value > 0.0])
    veto_mean = _mean(veto_pnl_30)
    control_mean = _mean(control_pnls)
    veto_benefit = max(-(veto_mean or 0.0), 0.0) * len(veto_pnl_30)
    opportunity_cost = max(control_positive_mean or 0.0, 0.0) * len(veto_records)

    return {
        "veto_group": {
            "count": len(veto_records),
            "counterfactual_pnl_30s_bps": _distribution(veto_pnl_30),
            "counterfactual_pnl_60s_bps": _distribution(veto_pnl_60),
            "counterfactual_pnl_120s_bps": _distribution(veto_pnl_120),
            "negative_rate_pct": _negative_rate_pct(veto_pnl_30),
        },
        "control_group": {
            "count": len(control_records),
            "actual_pnl_30s_bps": _distribution(control_pnls),
            "adverse_selection_rate_pct": _rate_pct(
                sum(1 for record in control_records if bool(record.get("adverse_selected"))),
                len(control_records),
            ),
        },
        "value_of_veto_bps": None if veto_mean is None or control_mean is None else control_mean - veto_mean,
        "net_impact_bps": veto_benefit - opportunity_cost,
        "warnings": _build_warnings(veto_records, control_records, veto_pnl_30),
    }


def _collect_counterfactual_pnls(
    veto_records: Iterable[Record],
    ordered_records: list[Record],
    *,
    horizon_sec: float,
) -> list[float]:
    values: list[float] = []
    for record in veto_records:
        counterfactual = _estimate_counterfactual_sell_pnl_bps(
            record,
            ordered_records,
            horizon_sec=horizon_sec,
        )
        if counterfactual is not None:
            values.append(counterfactual)
    return values


def _estimate_counterfactual_sell_pnl_bps(
    record: Record,
    ordered_records: list[Record],
    *,
    horizon_sec: float,
) -> float | None:
    order_price = _as_float(record.get("order_price"))
    timestamp = _as_float(record.get("timestamp"))
    if order_price is None or order_price <= 0.0 or timestamp is None:
        return None
    future_mid = _find_future_mid(ordered_records, timestamp=timestamp, horizon_sec=horizon_sec)
    if future_mid is None:
        return None
    return (order_price - future_mid) / order_price * 10_000.0


def _find_future_mid(
    records: list[Record],
    *,
    timestamp: float,
    horizon_sec: float,
) -> float | None:
    target_ts = timestamp + horizon_sec
    for record in records:
        record_ts = _as_float(record.get("timestamp"))
        if record_ts is None or record_ts < target_ts:
            continue
        for key in ("mid_at_order", "mid_at_fill", "mid_30s_after", "mid_60s_after", "mid_120s_after"):
            value = _as_float(record.get(key))
            if value is not None and value > 0.0:
                return value
    return None


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": _mean(ordered),
        "p10": _percentile(ordered, 0.10),
        "p90": _percentile(ordered, 0.90),
    }


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    index = min(max(int((len(values) - 1) * ratio), 0), len(values) - 1)
    return values[index]


def _negative_rate_pct(values: list[float]) -> float:
    return _rate_pct(sum(1 for value in values if value < 0.0), len(values))


def _rate_pct(num: int, den: int) -> float:
    return float(num / den * 100.0) if den else 0.0


def _mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _build_warnings(
    veto_records: list[Record],
    control_records: list[Record],
    veto_pnls: list[float],
) -> list[str]:
    warnings: list[str] = []
    if not veto_records:
        warnings.append("no trend_5s veto records")
    if not control_records:
        warnings.append("no trend_5s boost control records")
    if veto_records and not veto_pnls:
        warnings.append("no counterfactual mids available for veto records")
    return warnings

