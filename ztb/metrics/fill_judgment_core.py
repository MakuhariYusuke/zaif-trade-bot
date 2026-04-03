from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, SupportsFloat, cast


class SupportsExecJudgmentMetrics(Protocol):
    fill_rate_p90: float
    cancel_ratio: float
    queue_wait_median_sec: float
    post_fill_30s_pnl_mean: float
    post_fill_30s_pnl_pvalue: float
    adverse_selection_ratio: float
    adverse_selection_ratio_raw: float
    sample_sufficient: bool
    total_orders: int
    measurement_days: int


class SupportsQuickJudgmentMetrics(Protocol):
    attempted_fill_rate: float
    attempted_cancel_ratio: float
    queue_wait_median_sec: float
    post_fill_30s_pnl_mean: float
    post_fill_30s_pnl_pvalue: float
    skip_gate_ratio: float


class SupportsFullStructuralMetrics(Protocol):
    attempted_fill_rate: float
    overall_fill_rate: float
    attempted_cancel_ratio: float
    queue_wait_median_sec: float
    adverse_selection_ratio: float
    skip_gate_ratio: float
    measurement_days: int
    attempted_orders: int


class SupportsFullJudgmentMetrics(Protocol):
    post_fill_30s_pnl_mean: float
    post_fill_30s_pnl_pvalue: float
    post_fill_30s_pnl_ci_upper: float
    post_fill_60s_pnl_mean: float
    post_fill_60s_pnl_pvalue: float
    post_fill_120s_pnl_mean: float
    post_fill_120s_pnl_pvalue: float

    def to_dict(self) -> dict: ...


def build_exec_gate_checks(
    metrics: SupportsExecJudgmentMetrics,
    thresholds: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Build the core E1-E5 checks for G1.1 exec judgment."""
    min_fill = float(cast(SupportsFloat, thresholds.get("min_fill_rate_p90", 0.90)))
    max_cancel = float(cast(SupportsFloat, thresholds.get("max_cancel_ratio", 0.30)))
    max_wait = float(
        cast(SupportsFloat, thresholds.get("max_queue_wait_median_sec", 60))
    )
    min_pnl = float(cast(SupportsFloat, thresholds.get("min_post_fill_30s_pnl", 0.0)))
    max_adverse = float(
        cast(SupportsFloat, thresholds.get("max_adverse_selection_ratio", 0.20))
    )

    if metrics.post_fill_30s_pnl_mean >= min_pnl:
        pnl_pass = True
    elif metrics.post_fill_30s_pnl_pvalue >= 0.05:
        pnl_pass = True
    else:
        pnl_pass = False

    return {
        "E1_fill_rate_p90": {
            "value": metrics.fill_rate_p90,
            "threshold": min_fill,
            "pass": metrics.fill_rate_p90 >= min_fill,
        },
        "E2_cancel_ratio": {
            "value": metrics.cancel_ratio,
            "threshold": max_cancel,
            "pass": metrics.cancel_ratio <= max_cancel,
        },
        "E3_queue_wait_median": {
            "value": metrics.queue_wait_median_sec,
            "threshold": max_wait,
            "pass": metrics.queue_wait_median_sec <= max_wait,
        },
        "E4_post_fill_pnl": {
            "value": metrics.post_fill_30s_pnl_mean,
            "threshold": min_pnl,
            "pvalue": metrics.post_fill_30s_pnl_pvalue,
            "pass": pnl_pass,
        },
        "E5_adverse_selection": {
            "value": metrics.adverse_selection_ratio,
            "threshold": max_adverse,
            "pass": metrics.adverse_selection_ratio <= max_adverse,
        },
        "E5_adverse_selection_raw": {
            "value": metrics.adverse_selection_ratio_raw,
            "threshold": max_adverse,
            "pass": metrics.adverse_selection_ratio_raw <= max_adverse,
            "informational": True,
        },
    }


def resolve_exec_judgment_type(metrics: SupportsExecJudgmentMetrics) -> str:
    """Resolve PROVISIONAL/INTERIM/FINAL for G1.1 exec gate."""
    if metrics.sample_sufficient:
        return "FINAL"
    if metrics.total_orders >= 200 and metrics.measurement_days >= 3:
        return "INTERIM"
    return "PROVISIONAL"


def build_quick_gate_checks(
    metrics: SupportsQuickJudgmentMetrics,
    thresholds: Mapping[str, object],
    *,
    cumulative_loss_jpy: float,
) -> dict[str, dict[str, object]]:
    """Build the K1-K6 checks for G1.1 quick judgment."""
    min_att_fill = float(
        cast(SupportsFloat, thresholds.get("min_attempted_fill_rate", 0.60))
    )
    max_att_cancel = float(
        cast(SupportsFloat, thresholds.get("max_attempted_cancel_ratio", 0.40))
    )
    max_wait = float(
        cast(SupportsFloat, thresholds.get("max_queue_wait_median_sec", 120))
    )
    pnl_kill_p = float(cast(SupportsFloat, thresholds.get("pnl_kill_p_threshold", 0.02)))
    pnl_kill_mean = float(
        cast(SupportsFloat, thresholds.get("pnl_kill_mean_threshold", -0.8))
    )
    max_loss = float(
        cast(SupportsFloat, thresholds.get("max_cumulative_loss_jpy", 10000))
    )
    max_skip = float(cast(SupportsFloat, thresholds.get("max_skip_gate_ratio", 0.25)))

    pnl_is_significant = metrics.post_fill_30s_pnl_pvalue < pnl_kill_p
    pnl_is_large_loss = metrics.post_fill_30s_pnl_mean <= pnl_kill_mean

    return {
        "K1_attempted_fill_rate": {
            "value": metrics.attempted_fill_rate,
            "threshold": min_att_fill,
            "pass": metrics.attempted_fill_rate >= min_att_fill,
        },
        "K2_attempted_cancel_ratio": {
            "value": metrics.attempted_cancel_ratio,
            "threshold": max_att_cancel,
            "pass": metrics.attempted_cancel_ratio <= max_att_cancel,
        },
        "K3_queue_wait_median": {
            "value": metrics.queue_wait_median_sec,
            "threshold": max_wait,
            "pass": metrics.queue_wait_median_sec <= max_wait,
        },
        "K4_pnl_kill": {
            "value": metrics.post_fill_30s_pnl_mean,
            "pvalue": metrics.post_fill_30s_pnl_pvalue,
            "threshold_p": pnl_kill_p,
            "threshold_mean": pnl_kill_mean,
            "significant": pnl_is_significant,
            "large_loss": pnl_is_large_loss,
            "pass": not (pnl_is_significant and pnl_is_large_loss),
        },
        "K5_cumulative_loss": {
            "value": cumulative_loss_jpy,
            "threshold": max_loss,
            "pass": cumulative_loss_jpy < max_loss,
        },
        "K6_skip_gate_ratio": {
            "value": metrics.skip_gate_ratio,
            "threshold": max_skip,
            "pass": metrics.skip_gate_ratio <= max_skip,
        },
    }


def build_full_gate_structural_checks(
    metrics: SupportsFullStructuralMetrics,
    thresholds: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Build the non-PnL checks for G1.2 full judgment."""
    min_att_fill = float(
        cast(SupportsFloat, thresholds.get("min_attempted_fill_rate", 0.70))
    )
    min_overall_fill = float(
        cast(SupportsFloat, thresholds.get("min_overall_fill_rate", 0.62))
    )
    max_att_cancel = float(
        cast(SupportsFloat, thresholds.get("max_attempted_cancel_ratio", 0.30))
    )
    max_wait = float(
        cast(SupportsFloat, thresholds.get("max_queue_wait_median_sec", 60))
    )
    max_as = float(
        cast(SupportsFloat, thresholds.get("max_adverse_selection_ratio", 0.30))
    )
    max_skip = float(cast(SupportsFloat, thresholds.get("max_skip_gate_ratio", 0.20)))
    min_days = int(float(cast(SupportsFloat, thresholds.get("min_calendar_days", 7))))
    min_n = int(
        float(cast(SupportsFloat, thresholds.get("min_attempted_samples", 500)))
    )

    return {
        "F1_attempted_fill_rate": {
            "value": metrics.attempted_fill_rate,
            "threshold": min_att_fill,
            "pass": metrics.attempted_fill_rate >= min_att_fill,
        },
        "F1b_overall_fill_rate": {
            "value": metrics.overall_fill_rate,
            "threshold": min_overall_fill,
            "pass": metrics.overall_fill_rate >= min_overall_fill,
        },
        "F2_attempted_cancel_ratio": {
            "value": metrics.attempted_cancel_ratio,
            "threshold": max_att_cancel,
            "pass": metrics.attempted_cancel_ratio <= max_att_cancel,
        },
        "F3_queue_wait_median": {
            "value": metrics.queue_wait_median_sec,
            "threshold": max_wait,
            "pass": metrics.queue_wait_median_sec <= max_wait,
        },
        "F5_adverse_selection": {
            "value": metrics.adverse_selection_ratio,
            "threshold": max_as,
            "pass": metrics.adverse_selection_ratio <= max_as,
        },
        "F6_skip_gate_ratio": {
            "value": metrics.skip_gate_ratio,
            "threshold": max_skip,
            "pass": metrics.skip_gate_ratio <= max_skip,
        },
        "F7_calendar_days": {
            "value": metrics.measurement_days,
            "threshold": min_days,
            "pass": metrics.measurement_days >= min_days,
        },
        "F8_n_attempted": {
            "value": metrics.attempted_orders,
            "threshold": min_n,
            "pass": metrics.attempted_orders >= min_n,
        },
    }


def build_full_gate_pnl_checks(
    metrics: SupportsFullJudgmentMetrics,
    thresholds: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Build G1.2 full-gate PnL checks with Holm correction and mean-floor guard."""

    pnl_alpha = float(cast(SupportsFloat, thresholds.get("pnl_alpha", 0.05)))
    pnl_tests = [
        (
            "F4_pnl_30s",
            metrics.post_fill_30s_pnl_mean,
            metrics.post_fill_30s_pnl_pvalue,
            metrics.post_fill_30s_pnl_ci_upper,
        ),
        (
            "F4b_pnl_60s",
            metrics.post_fill_60s_pnl_mean,
            metrics.post_fill_60s_pnl_pvalue,
            None,
        ),
        (
            "F4c_pnl_120s",
            metrics.post_fill_120s_pnl_mean,
            metrics.post_fill_120s_pnl_pvalue,
            None,
        ),
    ]

    raw_pvals = [(name, pvalue) for name, _, pvalue, _ in pnl_tests]
    sorted_pvals = sorted(raw_pvals, key=lambda item: item[1])
    holm_adjusted: dict[str, float] = {}
    total_tests = len(sorted_pvals)
    for rank, (name, pvalue_raw) in enumerate(sorted_pvals):
        holm_adjusted[name] = min(pvalue_raw * (total_tests - rank), 1.0)

    checks: dict[str, dict[str, object]] = {}
    for name, mean_val, pvalue_raw, ci_upper in pnl_tests:
        pvalue_holm = holm_adjusted[name]
        pnl_pass = mean_val >= 0 or pvalue_holm >= pnl_alpha
        check_data: dict[str, object] = {
            "value": mean_val,
            "pvalue_raw": pvalue_raw,
            "pvalue_holm": round(pvalue_holm, 6),
            "alpha": pnl_alpha,
            "pass": pnl_pass,
        }
        if ci_upper is not None:
            check_data["ci_upper"] = ci_upper
        checks[name] = check_data

    pnl_mean_floor = float(
        cast(SupportsFloat, thresholds.get("pnl_mean_floor_bps", -0.10))
    )
    pnl_mean_hard_floor = float(
        cast(SupportsFloat, thresholds.get("pnl_mean_hard_floor_bps", -0.50))
    )
    pnl_30s_mean = metrics.post_fill_30s_pnl_mean
    if pnl_30s_mean >= pnl_mean_floor:
        mean_floor_pass = True
        mean_floor_watch = False
    elif pnl_30s_mean >= pnl_mean_hard_floor:
        mean_floor_pass = True
        mean_floor_watch = True
    else:
        mean_floor_pass = False
        mean_floor_watch = False

    checks["F4d_pnl_mean_floor"] = {
        "value": pnl_30s_mean,
        "floor": pnl_mean_floor,
        "hard_floor": pnl_mean_hard_floor,
        "pass": mean_floor_pass,
        "watch": mean_floor_watch,
    }
    checks["F4_pnl"] = {
        "value": metrics.post_fill_30s_pnl_mean,
        "pvalue": metrics.post_fill_30s_pnl_pvalue,
        "ci_upper": metrics.post_fill_30s_pnl_ci_upper,
        "alpha": pnl_alpha,
        "pass": checks["F4_pnl_30s"]["pass"],
    }
    return checks


__all__ = [
    "build_exec_gate_checks",
    "build_full_gate_pnl_checks",
    "build_full_gate_structural_checks",
    "build_quick_gate_checks",
    "resolve_exec_judgment_type",
]
