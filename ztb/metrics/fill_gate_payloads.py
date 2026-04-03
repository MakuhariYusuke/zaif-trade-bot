from __future__ import annotations

from collections.abc import Mapping

from ztb.metrics.fill_judgment_core import (
    SupportsFullJudgmentMetrics,
    SupportsQuickJudgmentMetrics,
)


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


def build_gate_payload(
    *,
    gate: str,
    gate_result: str,
    checks: Mapping[str, Mapping[str, object]],
    metrics: SupportsFullJudgmentMetrics,
    watch: bool = False,
    watch_detail: Mapping[str, object] | None = None,
    extras: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a consistent gate result payload for fill judgments."""

    payload: dict[str, object] = {
        "gate": gate,
        "gate_result": gate_result,
        "checks": dict(checks),
        "watch": watch,
        "metrics_summary": metrics.to_dict(),
    }
    if watch_detail is not None:
        payload["watch_detail"] = dict(watch_detail)
    if extras is not None:
        payload.update(extras)
    return payload


def build_quick_watch_detail(
    metrics: SupportsQuickJudgmentMetrics,
    *,
    pnl_watch_p: float,
    pnl_watch_mean: float,
) -> dict[str, object]:
    """Build the WATCH detail payload for G1.1 quick judgments."""

    return {
        "pnl_mean": metrics.post_fill_30s_pnl_mean,
        "pnl_pvalue": metrics.post_fill_30s_pnl_pvalue,
        "watch_thresholds": {"p": pnl_watch_p, "mean": pnl_watch_mean},
    }


__all__ = [
    "build_gate_payload",
    "build_quick_watch_detail",
    "resolve_gate_result",
]
