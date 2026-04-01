"""Canonical helpers for skip-gate early-return FillRecord assembly."""

from __future__ import annotations

from dataclasses import dataclass

from ztb.metrics.fill_quality import FillRecord, build_skip_fill_record
from ztb.ml.skip_gate_result_fields import SkipFillRecordExtraFields


@dataclass(frozen=True)
class SkipFillRecordContext:
    """Core context required to build a skip-gate FillRecord."""

    cycle_id: str
    timestamp: float
    side: str
    order_price: float
    order_quantity: float
    cancel_reason: str
    spread_at_order: float | None
    spread_offset_ratio: float
    run_id: str
    git_sha: str | None
    regime_value: str | None
    decision_trace_id: str | None = None


def build_skip_fill_record_from_context(
    *,
    context: SkipFillRecordContext,
    extra_fields: SkipFillRecordExtraFields,
) -> FillRecord:
    """Build a skip-gate FillRecord from canonical context and payload."""
    return build_skip_fill_record(
        cycle_id=context.cycle_id,
        timestamp=context.timestamp,
        side=context.side,
        order_price=context.order_price,
        order_quantity=context.order_quantity,
        cancel_reason=context.cancel_reason,
        run_id=context.run_id,
        git_sha=context.git_sha,
        spread_at_order=context.spread_at_order,
        spread_offset_ratio=context.spread_offset_ratio,
        regime=context.regime_value,
        decision_trace_id=context.decision_trace_id,
        skip_gate_skipped=extra_fields.skip_gate_skipped,
        skip_gate_score=extra_fields.skip_gate_score,
        skip_gate_reason=extra_fields.skip_gate_reason,
        skip_gate_model_used=extra_fields.skip_gate_model_used,
        skip_gate_as_prob=extra_fields.skip_gate_as_prob,
        skip_gate_threshold_used=extra_fields.skip_gate_threshold_used,
        skip_gate_hour_offset=extra_fields.skip_gate_hour_offset,
        orderbook_imbalance=extra_fields.orderbook_imbalance,
        bid_depth_total=extra_fields.bid_depth_total,
        ask_depth_total=extra_fields.ask_depth_total,
        price_velocity_bps=extra_fields.price_velocity_bps,
        trend_5s_guard_triggered=extra_fields.trend_5s_guard_triggered,
        trend_5s_guard_action=extra_fields.trend_5s_guard_action,
        trend_5s_at_order=extra_fields.trend_5s_at_order,
        ev_score_pretrade=extra_fields.ev_score_pretrade,
        decision_path=extra_fields.decision_path,
        skip_gate_budget_regime=extra_fields.skip_gate_budget_regime,
        skip_gate_budget_remaining=extra_fields.skip_gate_budget_remaining,
        skip_gate_budget_exhausted=extra_fields.skip_gate_budget_exhausted,
    )


__all__ = [
    "SkipFillRecordContext",
    "build_skip_fill_record_from_context",
]
