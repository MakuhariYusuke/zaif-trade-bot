from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, SupportsFloat, cast

from ztb.metrics.fill_exec_monitoring import build_exec_monitoring_checks
from ztb.metrics.fill_judgment_core import (
    build_exec_gate_checks,
    build_full_gate_pnl_checks,
    build_full_gate_structural_checks,
    build_gate_payload,
    build_quick_gate_checks,
    build_quick_watch_detail,
    resolve_exec_judgment_type,
    resolve_gate_result,
)

if TYPE_CHECKING:
    from ztb.metrics.fill_metric_results import FillMetrics
    from ztb.metrics.fill_quality import FillRecord


def build_g1_1_exec_judgment(
    metrics: "FillMetrics",
    thresholds: Mapping[str, object],
    records: Sequence["FillRecord"] | None = None,
) -> dict[str, object]:
    checks = build_exec_gate_checks(metrics, thresholds)
    if records is not None:
        checks.update(build_exec_monitoring_checks(records, thresholds))

    gate_checks = {key: value for key, value in checks.items() if not value.get("informational")}
    all_pass = all(check["pass"] for check in gate_checks.values())
    return cast(
        dict[str, object],
        build_gate_payload(
        gate="G1.1-exec",
        gate_result="PASS" if all_pass else "FAIL",
        checks=checks,
        metrics=metrics,
        extras={
            "judgment_type": resolve_exec_judgment_type(metrics),
            "sample_sufficient": metrics.sample_sufficient,
        },
        ),
    )


def build_g1_1_quick_judgment(
    metrics: "FillMetrics",
    thresholds: Mapping[str, object],
    *,
    cumulative_loss_jpy: float = 0.0,
) -> dict[str, object]:
    checks = build_quick_gate_checks(
        metrics,
        thresholds,
        cumulative_loss_jpy=cumulative_loss_jpy,
    )
    all_pass = all(check["pass"] for check in checks.values())

    pnl_watch_p = float(
        cast(SupportsFloat, thresholds.get("pnl_watch_p_threshold", 0.05))
    )
    pnl_watch_mean = float(
        cast(SupportsFloat, thresholds.get("pnl_watch_mean_threshold", -0.3))
    )
    is_watch = (
        all_pass
        and metrics.post_fill_30s_pnl_pvalue < pnl_watch_p
        and metrics.post_fill_30s_pnl_mean < pnl_watch_mean
    )
    gate_result = "FAIL" if not all_pass else "WATCH" if is_watch else "PASS"
    return cast(
        dict[str, object],
        build_gate_payload(
        gate="G1.1-quick",
        gate_result=gate_result,
        checks=checks,
        metrics=metrics,
        watch=is_watch,
        watch_detail=(
            build_quick_watch_detail(
                metrics,
                pnl_watch_p=pnl_watch_p,
                pnl_watch_mean=pnl_watch_mean,
            )
            if is_watch
            else None
        ),
        ),
    )


def build_g1_2_full_judgment(
    metrics: "FillMetrics",
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    checks = build_full_gate_structural_checks(metrics, thresholds)
    checks.update(build_full_gate_pnl_checks(metrics, thresholds))

    gate_result, is_watch = resolve_gate_result(checks)
    return cast(
        dict[str, object],
        build_gate_payload(
        gate="G1.2-full",
        gate_result=gate_result,
        checks=checks,
        metrics=metrics,
        watch=is_watch,
        ),
    )


__all__ = [
    "build_g1_1_exec_judgment",
    "build_g1_1_quick_judgment",
    "build_g1_2_full_judgment",
]
