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
    )


__all__ = [
    "SkipDecisionResultFields",
    "build_skip_decision_result_fields",
    "resolve_skip_gate_model_tag",
]
