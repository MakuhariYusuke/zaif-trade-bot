"""Guard pipeline observability helpers for FillRecord serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class GuardPipelineResult:
    """Typed nested view of pre-order guard decisions."""

    entry_gate_ev_bps: float | None
    entry_gate_action: str | None
    entry_gate_reason: str | None
    spread_at_decision_bps: float | None
    regime_at_decision: str | None
    trend_5s_value_bps: float | None
    trend_5s_action: str | None
    skip_gate_score: float | None
    skip_gate_action: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_gate_ev_bps": self.entry_gate_ev_bps,
            "entry_gate_action": self.entry_gate_action,
            "entry_gate_reason": self.entry_gate_reason,
            "spread_at_decision_bps": self.spread_at_decision_bps,
            "regime_at_decision": self.regime_at_decision,
            "trend_5s_value_bps": self.trend_5s_value_bps,
            "trend_5s_action": self.trend_5s_action,
            "skip_gate_score": self.skip_gate_score,
            "skip_gate_action": self.skip_gate_action,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> GuardPipelineResult:
        return cls(
            entry_gate_ev_bps=_as_float(payload.get("entry_gate_ev_bps")),
            entry_gate_action=_as_str(payload.get("entry_gate_action")),
            entry_gate_reason=_as_str(payload.get("entry_gate_reason")),
            spread_at_decision_bps=_as_float(payload.get("spread_at_decision_bps")),
            regime_at_decision=_as_str(payload.get("regime_at_decision")),
            trend_5s_value_bps=_as_float(payload.get("trend_5s_value_bps")),
            trend_5s_action=_as_str(payload.get("trend_5s_action")),
            skip_gate_score=_as_float(payload.get("skip_gate_score")),
            skip_gate_action=_as_str(payload.get("skip_gate_action")),
        )


class SupportsFillRecordGuardInputs(Protocol):
    cancel_reason: str | None
    entry_gate_ev: float | None
    entry_gate_blocked: bool | None
    entry_gate_guard_suppressed: bool | None
    spread_bps: float | None
    regime_at_order: str | None
    entry_gate_regime: str | None
    trend_5s_at_order: float | None
    trend_5s_guard_action: str | None
    skip_gate_score: float | None
    skip_gate_skipped: bool | None
    skip_gate_bypassed: bool | None
    skip_gate_reason: str | None


def build_guard_pipeline_result(
    *,
    cancel_reason: str | None,
    entry_gate_ev: float | None,
    entry_gate_blocked: bool | None,
    entry_gate_guard_suppressed: bool | None,
    spread_bps: float | None,
    regime_at_order: str | None,
    entry_gate_regime: str | None,
    trend_5s_at_order: float | None,
    trend_5s_guard_action: str | None,
    skip_gate_score: float | None,
    skip_gate_skipped: bool | None,
    skip_gate_bypassed: bool | None,
    skip_gate_reason: str | None,
) -> GuardPipelineResult | None:
    """Build a typed nested guard payload from existing flat telemetry."""

    entry_gate_action = _resolve_entry_gate_action(
        entry_gate_ev=entry_gate_ev,
        entry_gate_blocked=entry_gate_blocked,
        entry_gate_guard_suppressed=entry_gate_guard_suppressed,
    )
    entry_gate_reason = _resolve_entry_gate_reason(
        cancel_reason=cancel_reason,
        entry_gate_action=entry_gate_action,
    )
    skip_gate_action = _resolve_skip_gate_action(
        skip_gate_skipped=skip_gate_skipped,
        skip_gate_bypassed=skip_gate_bypassed,
        skip_gate_score=skip_gate_score,
        skip_gate_reason=skip_gate_reason,
    )

    if (
        entry_gate_action is None
        and entry_gate_reason is None
        and spread_bps is None
        and regime_at_order is None
        and entry_gate_regime is None
        and trend_5s_at_order is None
        and trend_5s_guard_action is None
        and skip_gate_score is None
        and skip_gate_action is None
    ):
        return None

    return GuardPipelineResult(
        entry_gate_ev_bps=entry_gate_ev,
        entry_gate_action=entry_gate_action,
        entry_gate_reason=entry_gate_reason,
        spread_at_decision_bps=spread_bps,
        regime_at_decision=regime_at_order or entry_gate_regime,
        trend_5s_value_bps=trend_5s_at_order,
        trend_5s_action=trend_5s_guard_action,
        skip_gate_score=skip_gate_score,
        skip_gate_action=skip_gate_action,
    )


def build_fill_record_guard_pipeline(
    record: SupportsFillRecordGuardInputs,
) -> GuardPipelineResult | None:
    """Build a guard pipeline payload directly from a FillRecord-like object."""
    return build_guard_pipeline_result(
        cancel_reason=record.cancel_reason,
        entry_gate_ev=record.entry_gate_ev,
        entry_gate_blocked=record.entry_gate_blocked,
        entry_gate_guard_suppressed=record.entry_gate_guard_suppressed,
        spread_bps=record.spread_bps,
        regime_at_order=record.regime_at_order,
        entry_gate_regime=record.entry_gate_regime,
        trend_5s_at_order=record.trend_5s_at_order,
        trend_5s_guard_action=record.trend_5s_guard_action,
        skip_gate_score=record.skip_gate_score,
        skip_gate_skipped=record.skip_gate_skipped,
        skip_gate_bypassed=record.skip_gate_bypassed,
        skip_gate_reason=record.skip_gate_reason,
    )


def _resolve_entry_gate_action(
    *,
    entry_gate_ev: float | None,
    entry_gate_blocked: bool | None,
    entry_gate_guard_suppressed: bool | None,
) -> str | None:
    if entry_gate_ev is None:
        return None
    if bool(entry_gate_blocked) and not bool(entry_gate_guard_suppressed):
        return "block"
    if entry_gate_ev <= 0.0:
        return "bypass"
    return "allow"


def _resolve_entry_gate_reason(
    *,
    cancel_reason: str | None,
    entry_gate_action: str | None,
) -> str | None:
    if cancel_reason and cancel_reason.startswith("entry_gate_"):
        return cancel_reason.removeprefix("entry_gate_")
    if entry_gate_action in {"block", "bypass"}:
        return "ev_negative"
    return None


def _resolve_skip_gate_action(
    *,
    skip_gate_skipped: bool | None,
    skip_gate_bypassed: bool | None,
    skip_gate_score: float | None,
    skip_gate_reason: str | None,
) -> str | None:
    if bool(skip_gate_bypassed):
        return "bypass"
    if bool(skip_gate_skipped):
        return "block"
    if skip_gate_score is not None or skip_gate_reason is not None:
        return "allow"
    return None


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = [
    "GuardPipelineResult",
    "build_fill_record_guard_pipeline",
    "build_guard_pipeline_result",
]
