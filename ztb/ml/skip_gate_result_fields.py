"""Shared skip-gate decision-to-result field helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ztb.ml.skip_gate_contracts import SkipDecisionLike


@dataclass(frozen=True)
class SkipDecisionResultFields:
    """Normalized result fields derived from a skip-gate decision."""

    skipped: bool
    score: float
    reason: str
    model_used: str
    as_prob: float | None
    threshold_used: float | None
    hour_offset: float
    price_velocity_bps: float | None
    # 642# 可観測性
    forced_pass: bool
    side_skip_rate: float | None


@dataclass(frozen=True)
class SkipFillRecordExtraFields:
    """Normalized extra payload for build_skip_fill_record()."""

    skip_gate_skipped: bool
    skip_gate_score: float
    skip_gate_reason: str
    skip_gate_model_used: str
    skip_gate_as_prob: float | None
    skip_gate_threshold_used: float | None
    skip_gate_hour_offset: float | None
    orderbook_imbalance: float | None
    bid_depth_total: float | None
    ask_depth_total: float | None
    price_velocity_bps: float | None
    trend_5s_guard_triggered: bool | None
    trend_5s_guard_action: str | None
    trend_5s_at_order: float | None
    ev_score_pretrade: float | None
    decision_path: str | None


def resolve_skip_gate_model_tag(
    *,
    decision_model_used: str,
    side: str,
    has_side_specific_model: bool,
) -> str:
    """Resolve the user-facing model tag for skip-gate telemetry."""
    if decision_model_used == "ev_weighted":
        return f"ev_weighted_{side}"
    return f"side_{side}" if has_side_specific_model else "unified"


def build_skip_decision_result_fields(
    decision: SkipDecisionLike,
    *,
    side: str,
    has_side_specific_model: bool,
    hour_offset: float,
    price_velocity_bps: float | None,
) -> SkipDecisionResultFields:
    """Convert a skip-gate decision into normalized result metadata."""
    model_tag = resolve_skip_gate_model_tag(
        decision_model_used=decision.model_used,
        side=side,
        has_side_specific_model=has_side_specific_model,
    )
    return SkipDecisionResultFields(
        skipped=decision.should_skip,
        score=decision.predicted_pnl_bps,
        reason=decision.reason,
        model_used=f"{decision.model_used}:{model_tag}",
        as_prob=decision.as_probability,
        threshold_used=decision.threshold_used,
        hour_offset=hour_offset,
        price_velocity_bps=price_velocity_bps,
        forced_pass=getattr(decision, "forced_pass", False),
        side_skip_rate=getattr(decision, "side_skip_rate", None),
    )


def build_skip_fill_record_extra_fields(
    *,
    score: float,
    reason: str,
    model_used: str,
    orderbook_imbalance: float | None,
    bid_depth_total: float | None,
    ask_depth_total: float | None,
    as_prob: float | None = None,
    threshold_used: float | None = None,
    hour_offset: float | None = None,
    price_velocity_bps: float | None = None,
    trend_5s_guard_triggered: bool | None = None,
    trend_5s_guard_action: str | None = None,
    trend_5s_at_order: float | None = None,
    ev_score_pretrade: float | None = None,
    decision_path: str | None = None,
) -> SkipFillRecordExtraFields:
    """Build canonical extra payload for skip FillRecord early returns."""
    return SkipFillRecordExtraFields(
        skip_gate_skipped=True,
        skip_gate_score=score,
        skip_gate_reason=reason,
        skip_gate_model_used=model_used,
        skip_gate_as_prob=as_prob,
        skip_gate_threshold_used=threshold_used,
        skip_gate_hour_offset=hour_offset,
        orderbook_imbalance=orderbook_imbalance,
        bid_depth_total=bid_depth_total,
        ask_depth_total=ask_depth_total,
        price_velocity_bps=price_velocity_bps,
        trend_5s_guard_triggered=trend_5s_guard_triggered,
        trend_5s_guard_action=trend_5s_guard_action,
        trend_5s_at_order=trend_5s_at_order,
        ev_score_pretrade=ev_score_pretrade,
        decision_path=decision_path,
    )


__all__ = [
    "SkipDecisionResultFields",
    "SkipFillRecordExtraFields",
    "build_skip_decision_result_fields",
    "build_skip_fill_record_extra_fields",
    "resolve_skip_gate_model_tag",
]
