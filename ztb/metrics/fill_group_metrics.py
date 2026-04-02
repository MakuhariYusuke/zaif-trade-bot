"""Grouped fill metrics shared helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from ztb.metrics.pnl_accumulators import PnlAccumulator

if TYPE_CHECKING:
    from ztb.metrics.fill_quality import FillRecord


@dataclass
class GroupedMetricsBase:
    """グループ集計で共通となる損益・AS 指標."""

    count: int = 0
    filled: int = 0
    pnl_mean_bps: float = 0.0
    as_ratio: float = 0.0


@dataclass
class RegimeMetrics(GroupedMetricsBase):
    """レジーム別の集計指標."""

    regime: str = "unknown"
    fill_rate: float = 0.0
    queue_wait_median_sec: float = 0.0


@dataclass
class HourlyMetrics(GroupedMetricsBase):
    """UTC 時間帯別の集計指標."""

    utc_hour: int = 0


def _summarize_filled_records(
    records: list[FillRecord],
    *,
    include_queue_wait: bool = False,
) -> tuple[int, float, float, float]:
    """filled レコードの共通集計を単一パスで算出."""
    filled_count = 0
    pnl_acc = PnlAccumulator()
    as_total = 0
    as_positive = 0
    queue_waits: list[float] = []

    for record in records:
        if not record.filled:
            continue
        filled_count += 1
        pnl_acc.add(record.post_fill_30s_pnl)
        if record.adverse_selected is not None:
            as_total += 1
            if record.adverse_selected:
                as_positive += 1
        if include_queue_wait and record.queue_wait_sec > 0:
            queue_waits.append(record.queue_wait_sec)

    as_ratio = (as_positive / as_total) if as_total else 0.0
    wait_median = float(np.median(queue_waits)) if queue_waits else 0.0
    return filled_count, pnl_acc.mean_bps, as_ratio, wait_median


def compute_regime_metrics(records: list[FillRecord]) -> list[RegimeMetrics]:
    """レジーム別にメトリクスを算出."""
    groups: dict[str, list[FillRecord]] = defaultdict(list)
    for record in records:
        groups[record.regime or "unknown"].append(record)

    result: list[RegimeMetrics] = []
    for regime_name in sorted(groups):
        grouped_records = groups[regime_name]
        filled_count, pnl_mean, as_ratio, wait_median = _summarize_filled_records(
            grouped_records,
            include_queue_wait=True,
        )
        result.append(
            RegimeMetrics(
                regime=regime_name,
                count=len(grouped_records),
                filled=filled_count,
                fill_rate=filled_count / len(grouped_records) if grouped_records else 0.0,
                pnl_mean_bps=pnl_mean,
                as_ratio=as_ratio,
                queue_wait_median_sec=wait_median,
            )
        )
    return result


def compute_hourly_metrics(records: list[FillRecord]) -> list[HourlyMetrics]:
    """UTC 時間帯別にメトリクスを算出."""
    groups: dict[int, list[FillRecord]] = defaultdict(list)
    for record in records:
        utc_hour = datetime.fromtimestamp(record.timestamp, tz=timezone.utc).hour
        groups[utc_hour].append(record)

    result: list[HourlyMetrics] = []
    for hour in range(24):
        if hour not in groups:
            continue
        grouped_records = groups[hour]
        filled_count, pnl_mean, as_ratio, _ = _summarize_filled_records(
            grouped_records,
            include_queue_wait=False,
        )
        result.append(
            HourlyMetrics(
                utc_hour=hour,
                count=len(grouped_records),
                filled=filled_count,
                pnl_mean_bps=pnl_mean,
                as_ratio=as_ratio,
            )
        )
    return result


__all__ = [
    "GroupedMetricsBase",
    "RegimeMetrics",
    "HourlyMetrics",
    "compute_regime_metrics",
    "compute_hourly_metrics",
]
