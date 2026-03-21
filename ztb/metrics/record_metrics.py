"""Shared fill-record metric aggregation helpers.

This module hosts reusable metric extraction used by v460 analysis and
judgment flows. It sits under ``ztb.metrics`` because it owns metric
calculation, while higher-level comparison/report orchestration stays in
``scripts/v460`` or ``ztb.adaptation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np

from ztb.io.json_io import JSONObject
from ztb.metrics.fill_quality import PnlAccumulator, PnlWinAccumulator, format_utc_day
from ztb.utils.safety import safe_to_finite

MetricRecord = JSONObject


class BaseMetrics(TypedDict):
    n_total: int
    n_filled: int
    fill_rate: float
    avg_pnl30_bps: float
    std_pnl30_bps: float
    downside_p10_bps: float
    downside_p05_bps: float
    profitable_rate: float
    calendar_days: int
    pnl30_array: np.ndarray


class ExtendedMetrics(BaseMetrics):
    as_rate: float
    avg_as_loss_bps: float
    reprice_rate: float
    avg_reprice_drift_bps: float
    vg_trigger_rate: float


@dataclass
class MetricsAccumulator:
    """Shared aggregation state for fill-record metrics."""

    n_total: int = 0
    n_filled: int = 0
    pnl_values: list[float] = field(default_factory=list)
    pnl_summary: PnlWinAccumulator = field(default_factory=PnlWinAccumulator)
    calendar_days: set[str] = field(default_factory=set)
    as_count: int = 0
    as_pnl_acc: PnlAccumulator = field(default_factory=PnlAccumulator)
    repriced_count: int = 0
    drift_acc: PnlAccumulator = field(default_factory=PnlAccumulator)
    vg_triggered_count: int = 0

    def add(self, record: MetricRecord) -> None:
        """Reflect one record into the aggregate."""
        self.n_total += 1
        if not record.get("filled"):
            return

        self.n_filled += 1
        pnl_value = safe_to_finite(record.get("post_fill_30s_pnl"))
        if pnl_value is not None:
            self.pnl_values.append(pnl_value)
            self.pnl_summary.add(pnl_value)

        ts = safe_to_finite(record.get("timestamp"))
        day_key = format_utc_day(ts)
        if day_key is not None:
            self.calendar_days.add(day_key)

        if record.get("adverse_selected"):
            self.as_count += 1
            self.as_pnl_acc.add(pnl_value)

        if (record.get("reprice_count") or 0) > 0:
            self.repriced_count += 1
            self.drift_acc.add(safe_to_finite(record.get("reprice_drift_bps")))

        if record.get("vg_boost_factor") is not None:
            self.vg_triggered_count += 1

    def to_base_metrics(self) -> BaseMetrics:
        """Return base metrics for the current aggregate."""
        fill_rate = self.n_filled / self.n_total if self.n_total > 0 else 0.0
        if self.pnl_values:
            arr = np.array(self.pnl_values, dtype=float)
            avg_pnl30 = self.pnl_summary.mean_bps
            std_pnl30 = float(np.std(arr))
            p10 = float(np.percentile(arr, 10))
            p05 = float(np.percentile(arr, 5))
            profitable = self.pnl_summary.win_rate
        else:
            # 160# bugfix: keep insufficient values as NaN to avoid false PASS.
            arr = np.array([], dtype=float)
            avg_pnl30 = float("nan")
            std_pnl30 = 0.0
            p10 = float("nan")
            p05 = float("nan")
            profitable = 0.0

        return {
            "n_total": self.n_total,
            "n_filled": self.n_filled,
            "fill_rate": fill_rate,
            "avg_pnl30_bps": avg_pnl30,
            "std_pnl30_bps": std_pnl30,
            "downside_p10_bps": p10,
            "downside_p05_bps": p05,
            "profitable_rate": profitable,
            "calendar_days": len(self.calendar_days),
            "pnl30_array": arr,
        }

    def to_extended_metrics(self) -> ExtendedMetrics:
        """Return base metrics plus AS/reprice/VG extensions."""
        base = self.to_base_metrics()
        n_filled = self.n_filled
        base.update(
            {
                "as_rate": self.as_count / n_filled if n_filled > 0 else 0.0,
                "avg_as_loss_bps": self.as_pnl_acc.mean_bps if self.as_pnl_acc.count else 0.0,
                "reprice_rate": self.repriced_count / n_filled if n_filled > 0 else 0.0,
                "avg_reprice_drift_bps": self.drift_acc.mean_bps if self.drift_acc.count else 0.0,
                "vg_trigger_rate": self.vg_triggered_count / n_filled if n_filled > 0 else 0.0,
            }
        )
        return base


def compute_base_metrics(records: list[MetricRecord]) -> BaseMetrics:
    """Compute base metrics from fill records."""
    acc = MetricsAccumulator()
    for record in records:
        acc.add(record)
    return acc.to_base_metrics()


def compute_extended_metrics(records: list[MetricRecord]) -> ExtendedMetrics:
    """Compute base metrics plus AS/reprice/VG extensions."""
    acc = MetricsAccumulator()
    for record in records:
        acc.add(record)
    return acc.to_extended_metrics()


__all__ = [
    "MetricRecord",
    "BaseMetrics",
    "ExtendedMetrics",
    "MetricsAccumulator",
    "compute_base_metrics",
    "compute_extended_metrics",
]
