from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final, Protocol, Sequence, SupportsFloat, cast

import numpy as np
from scipy import stats

_SECONDS_PER_DAY: Final[float] = 86_400.0


class FillMetricRecord(Protocol):
    timestamp: float
    filled: bool
    cancelled: bool
    queue_wait_sec: float
    post_fill_30s_pnl: float | None
    post_fill_60s_pnl: float | None
    post_fill_120s_pnl: float | None
    adverse_selected: bool | None
    adverse_selected_raw: bool | None
    cancel_reason: str | None
    skip_gate_skipped: bool | None


@dataclass
class DailyFillCount:
    total: int = 0
    filled: int = 0

    def add(self, *, filled: bool) -> None:
        self.total += 1
        if filled:
            self.filled += 1


@dataclass
class FillMetricScan:
    total: int
    filled_count: int = 0
    cancelled_count: int = 0
    cancel_reason_breakdown: dict[str, int] = field(default_factory=dict)
    skip_gate_count: int = 0
    daily_fill_rates: list[float] = field(default_factory=list)
    wait_times: list[float] = field(default_factory=list)
    pnl_values: list[float] = field(default_factory=list)
    pnl60_values: list[float] = field(default_factory=list)
    pnl120_values: list[float] = field(default_factory=list)
    as_coverage: int = 0
    as_raw_coverage: int = 0
    n_adverse: int = 0
    n_adverse_raw: int = 0


def format_utc_day(timestamp: object, *, compact: bool = True) -> str | None:
    try:
        ts = float(cast(SupportsFloat | str | bytes | bytearray, timestamp))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ts):
        return None
    try:
        pattern = "%Y%m%d" if compact else "%Y-%m-%d"
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(pattern)
    except (OverflowError, OSError, ValueError):
        return None


def utc_day_bucket(timestamp: object) -> int | None:
    try:
        ts = float(cast(SupportsFloat | str | bytes | bytearray, timestamp))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ts):
        return None
    try:
        return int(ts // _SECONDS_PER_DAY)
    except (OverflowError, ValueError):
        return None


def mean_and_one_sided_pvalue(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean_val = float(np.mean(values))
    if len(values) < 2:
        return mean_val, 1.0

    t_stat, two_sided_p = stats.ttest_1samp(list(values), 0.0)
    if t_stat < 0:
        return mean_val, float(two_sided_p / 2)
    return mean_val, 1.0 - float(two_sided_p / 2)


def scan_fill_metric_inputs(records: Sequence[FillMetricRecord]) -> FillMetricScan:
    scan = FillMetricScan(total=len(records))
    daily_groups: dict[int, DailyFillCount] = {}

    for record in records:
        if record.filled:
            scan.filled_count += 1
            if record.queue_wait_sec > 0:
                scan.wait_times.append(record.queue_wait_sec)

            if record.post_fill_30s_pnl is not None:
                scan.pnl_values.append(record.post_fill_30s_pnl)
            if record.post_fill_60s_pnl is not None:
                scan.pnl60_values.append(record.post_fill_60s_pnl)
            if record.post_fill_120s_pnl is not None:
                scan.pnl120_values.append(record.post_fill_120s_pnl)

            if record.adverse_selected is not None:
                scan.as_coverage += 1
                if record.adverse_selected:
                    scan.n_adverse += 1

            if record.adverse_selected_raw is not None:
                scan.as_raw_coverage += 1
                if record.adverse_selected_raw:
                    scan.n_adverse_raw += 1

        if record.cancelled:
            scan.cancelled_count += 1
            reason = record.cancel_reason or "unknown"
            scan.cancel_reason_breakdown[reason] = (
                scan.cancel_reason_breakdown.get(reason, 0) + 1
            )
        if record.skip_gate_skipped is True or record.cancel_reason == "skip_gate":
            scan.skip_gate_count += 1

        day_key = utc_day_bucket(record.timestamp)
        if day_key is None:
            continue
        day_stats = daily_groups.get(day_key)
        if day_stats is None:
            day_stats = DailyFillCount()
            daily_groups[day_key] = day_stats
        day_stats.add(filled=record.filled)

    for day_key in sorted(daily_groups):
        day_stats = daily_groups[day_key]
        scan.daily_fill_rates.append(
            day_stats.filled / day_stats.total if day_stats.total > 0 else 0.0
        )

    return scan
