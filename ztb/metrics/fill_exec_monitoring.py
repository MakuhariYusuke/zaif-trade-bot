"""Informational G1.1 execution checks derived from round-trip metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, SupportsFloat, cast

from ztb.metrics.fill_round_trip_metrics import compute_round_trip_metrics

if TYPE_CHECKING:
    from ztb.metrics.fill_quality import FillRecord


def build_exec_monitoring_checks(
    records: Sequence["FillRecord"],
    thresholds: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Build informational G1.1 execution checks from round-trip metrics."""
    filled_recs = [record for record in records if record.filled]
    if len(filled_recs) < 2:
        return {}

    rt_metrics, _ = compute_round_trip_metrics(list(records))
    if rt_metrics.total_pairs <= 0:
        return {}

    min_rt_pnl = float(cast(SupportsFloat, thresholds.get("min_round_trip_pnl_mean", -2.0)))
    max_inventory = float(cast(SupportsFloat, thresholds.get("max_net_inventory", 5)))
    return {
        "E6_round_trip_pnl": {
            "value": rt_metrics.pnl_mean_bps,
            "threshold": min_rt_pnl,
            "pass": rt_metrics.pnl_mean_bps >= min_rt_pnl,
            "pairs": rt_metrics.total_pairs,
            "median": rt_metrics.pnl_median_bps,
            "total_jpy": rt_metrics.pnl_total_jpy,
            "informational": True,
        },
        "E7_net_inventory": {
            "value": abs(rt_metrics.net_inventory),
            "threshold": max_inventory,
            "pass": abs(rt_metrics.net_inventory) <= max_inventory,
            "net_inventory": rt_metrics.net_inventory,
            "unpaired_buys": rt_metrics.unpaired_buys,
            "unpaired_sells": rt_metrics.unpaired_sells,
            "informational": True,
        },
    }


__all__ = ["build_exec_monitoring_checks"]
