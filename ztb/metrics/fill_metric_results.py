from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy import stats

from ztb.metrics.fill_metrics_core import mean_and_one_sided_pvalue, scan_fill_metric_inputs
from ztb.utils.dataclass_utils import shallow_asdict


class SupportsFillMetricRecord(Protocol):
    filled: bool
    cancelled: bool
    queue_wait_sec: float
    post_fill_30s_pnl: float | None
    adverse_selected: bool | None
    adverse_selected_raw: bool | None
    post_fill_60s_pnl: float | None
    post_fill_120s_pnl: float | None
    cancel_reason: str | None
    timestamp: float
    skip_gate_skipped: bool | None


@dataclass
class FillMetrics:
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    fill_rate_p90: float = 0.0
    cancel_ratio: float = 0.0
    queue_wait_median_sec: float = 0.0
    post_fill_30s_pnl_mean: float = 0.0
    post_fill_30s_pnl_pvalue: float = 1.0
    adverse_selection_ratio: float = 0.0
    adverse_selection_ratio_raw: float = 0.0
    daily_fill_rates: list[float] = field(default_factory=list)
    measurement_days: int = 0
    sample_sufficient: bool = False
    as_coverage: int = 0
    as_raw_coverage: int = 0
    attempted_orders: int = 0
    skip_gate_count: int = 0
    skip_gate_ratio: float = 0.0
    attempted_fill_rate: float = 0.0
    attempted_cancel_ratio: float = 0.0
    overall_fill_rate: float = 0.0
    post_fill_30s_pnl_ci_upper: float = 0.0
    post_fill_60s_pnl_mean: float = 0.0
    post_fill_60s_pnl_pvalue: float = 1.0
    post_fill_120s_pnl_mean: float = 0.0
    post_fill_120s_pnl_pvalue: float = 1.0
    cancel_reason_breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return shallow_asdict(self)


def compute_fill_metrics(records: list[SupportsFillMetricRecord]) -> FillMetrics:
    if not records:
        return FillMetrics()

    scan = scan_fill_metric_inputs(records)
    fill_rate_p90 = (
        float(np.percentile(scan.daily_fill_rates, 10))
        if scan.daily_fill_rates
        else 0.0
    )
    cancel_ratio = scan.cancelled_count / scan.total if scan.total > 0 else 0.0
    queue_wait_median = float(np.median(scan.wait_times)) if scan.wait_times else 0.0
    pnl_mean, pnl_pvalue = mean_and_one_sided_pvalue(scan.pnl_values)
    adverse_ratio = scan.n_adverse / scan.as_coverage if scan.as_coverage else 0.0
    adverse_ratio_raw = (
        scan.n_adverse_raw / scan.as_raw_coverage if scan.as_raw_coverage else adverse_ratio
    )
    sample_sufficient = scan.total >= 200 and len(scan.daily_fill_rates) >= 7
    attempted_orders = scan.total - scan.skip_gate_count
    skip_gate_ratio = scan.skip_gate_count / scan.total if scan.total > 0 else 0.0
    attempted_fill_rate = (
        scan.filled_count / attempted_orders if attempted_orders > 0 else 0.0
    )
    attempted_cancel_ratio = (
        (attempted_orders - scan.filled_count) / attempted_orders
        if attempted_orders > 0
        else 0.0
    )
    overall_fill_rate = scan.filled_count / scan.total if scan.total > 0 else 0.0

    pnl_ci_upper = 0.0
    if scan.pnl_values and len(scan.pnl_values) >= 2:
        se = float(np.std(scan.pnl_values, ddof=1)) / np.sqrt(len(scan.pnl_values))
        t_crit = float(stats.t.ppf(0.975, df=len(scan.pnl_values) - 1))
        pnl_ci_upper = pnl_mean + t_crit * se

    pnl60_mean, pnl60_pvalue = mean_and_one_sided_pvalue(scan.pnl60_values)
    pnl120_mean, pnl120_pvalue = mean_and_one_sided_pvalue(scan.pnl120_values)

    return FillMetrics(
        total_orders=scan.total,
        filled_orders=scan.filled_count,
        cancelled_orders=scan.cancelled_count,
        fill_rate_p90=fill_rate_p90,
        cancel_ratio=cancel_ratio,
        queue_wait_median_sec=queue_wait_median,
        post_fill_30s_pnl_mean=pnl_mean,
        post_fill_30s_pnl_pvalue=pnl_pvalue,
        adverse_selection_ratio=adverse_ratio,
        adverse_selection_ratio_raw=adverse_ratio_raw,
        daily_fill_rates=scan.daily_fill_rates,
        measurement_days=len(scan.daily_fill_rates),
        sample_sufficient=sample_sufficient,
        as_coverage=scan.as_coverage,
        as_raw_coverage=scan.as_raw_coverage,
        attempted_orders=attempted_orders,
        skip_gate_count=scan.skip_gate_count,
        skip_gate_ratio=skip_gate_ratio,
        attempted_fill_rate=attempted_fill_rate,
        attempted_cancel_ratio=attempted_cancel_ratio,
        overall_fill_rate=overall_fill_rate,
        post_fill_30s_pnl_ci_upper=pnl_ci_upper,
        post_fill_60s_pnl_mean=pnl60_mean,
        post_fill_60s_pnl_pvalue=pnl60_pvalue,
        post_fill_120s_pnl_mean=pnl120_mean,
        post_fill_120s_pnl_pvalue=pnl120_pvalue,
        cancel_reason_breakdown=scan.cancel_reason_breakdown,
    )


__all__ = ["FillMetrics", "compute_fill_metrics"]
