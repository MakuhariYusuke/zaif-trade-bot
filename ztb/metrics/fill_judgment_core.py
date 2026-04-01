from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, SupportsFloat, cast


class SupportsFullJudgmentMetrics(Protocol):
    post_fill_30s_pnl_mean: float
    post_fill_30s_pnl_pvalue: float
    post_fill_30s_pnl_ci_upper: float
    post_fill_60s_pnl_mean: float
    post_fill_60s_pnl_pvalue: float
    post_fill_120s_pnl_mean: float
    post_fill_120s_pnl_pvalue: float


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


def resolve_gate_result(
    checks: Mapping[str, Mapping[str, object]],
) -> tuple[str, bool]:
    """Resolve PASS/WATCH/FAIL from per-check payloads."""

    all_pass = all(bool(check["pass"]) for check in checks.values())
    is_watch = any(bool(check.get("watch", False)) for check in checks.values())

    if not all_pass:
        return "FAIL", is_watch
    if is_watch:
        return "WATCH", True
    return "PASS", False


__all__ = ["build_full_gate_pnl_checks", "resolve_gate_result"]
