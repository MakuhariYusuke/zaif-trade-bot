from __future__ import annotations

from dataclasses import dataclass

from ztb.metrics.fill_quality import FillRecord
from ztb.ml.skip_gate_fill_record import (
    SkipFillRecordContext,
    build_skip_fill_record_from_context,
)
from ztb.ml.skip_gate_result_fields import (
    SkipDecisionResultFields,
    SkipFillRecordExtraFields,
    build_skip_decision_result_fields,
    build_skip_fill_record_extra_fields,
    resolve_skip_gate_model_tag,
)


@dataclass
class _DecisionStub:
    should_skip: bool
    predicted_pnl_bps: float
    threshold_bps: float
    reason: str
    model_used: str
    as_probability: float | None
    threshold_used: float | None
    features_used: int
    forced_pass: bool = False
    side_skip_rate: float | None = None


class TestSkipGateResultFieldsMigration:
    def test_resolve_model_tag_for_ev_weighted(self) -> None:
        assert resolve_skip_gate_model_tag(
            decision_model_used="ev_weighted",
            side="sell",
            has_side_specific_model=True,
        ) == "ev_weighted_sell"

    def test_resolve_model_tag_for_side_specific_model(self) -> None:
        assert resolve_skip_gate_model_tag(
            decision_model_used="pnl",
            side="buy",
            has_side_specific_model=True,
        ) == "side_buy"

    def test_build_result_fields_for_unified_model(self) -> None:
        decision = _DecisionStub(
            should_skip=False,
            predicted_pnl_bps=1.25,
            threshold_bps=0.0,
            reason="pass",
            model_used="pnl",
            as_probability=0.42,
            threshold_used=-0.1,
            features_used=12,
        )

        fields = build_skip_decision_result_fields(
            decision,
            side="buy",
            has_side_specific_model=False,
            hour_offset=0.2,
            price_velocity_bps=1.5,
        )

        assert fields == SkipDecisionResultFields(
            skipped=False,
            score=1.25,
            reason="pass",
            model_used="pnl:unified",
            as_prob=0.42,
            threshold_used=-0.1,
            hour_offset=0.2,
            price_velocity_bps=1.5,
            forced_pass=False,
            side_skip_rate=None,
            budget_regime=None,
            budget_remaining=None,
            budget_exhausted=False,
        )

    def test_build_skip_fill_record_extra_fields(self) -> None:
        fields = build_skip_fill_record_extra_fields(
            score=-1.5,
            reason="rule_skip_unknown_sell",
            model_used="rule",
            orderbook_imbalance=0.2,
            bid_depth_total=10.0,
            ask_depth_total=8.0,
            as_prob=0.9,
            threshold_used=0.1,
            hour_offset=0.3,
            price_velocity_bps=4.5,
            ev_score_pretrade=-0.2,
            decision_path="ev_normal_skip",
        )

        assert fields == SkipFillRecordExtraFields(
            skip_gate_skipped=True,
            skip_gate_score=-1.5,
            skip_gate_reason="rule_skip_unknown_sell",
            skip_gate_model_used="rule",
            skip_gate_as_prob=0.9,
            skip_gate_threshold_used=0.1,
            skip_gate_hour_offset=0.3,
            orderbook_imbalance=0.2,
            bid_depth_total=10.0,
            ask_depth_total=8.0,
            price_velocity_bps=4.5,
            trend_5s_guard_triggered=None,
            trend_5s_guard_action=None,
            trend_5s_at_order=None,
            as_trailing_gate_action=None,
            as_trailing_gate_rate=None,
            as_trailing_gate_offset_mult=None,
            ev_score_pretrade=-0.2,
            decision_path="ev_normal_skip",
            skip_gate_budget_regime=None,
            skip_gate_budget_remaining=None,
            skip_gate_budget_exhausted=None,
        )

    def test_build_skip_fill_record_extra_fields_defaults_optional_to_none(self) -> None:
        fields = build_skip_fill_record_extra_fields(
            score=0.0,
            reason="pass",
            model_used="pnl:unified",
            orderbook_imbalance=None,
            bid_depth_total=None,
            ask_depth_total=None,
        )

        assert fields == SkipFillRecordExtraFields(
            skip_gate_skipped=True,
            skip_gate_score=0.0,
            skip_gate_reason="pass",
            skip_gate_model_used="pnl:unified",
            skip_gate_as_prob=None,
            skip_gate_threshold_used=None,
            skip_gate_hour_offset=None,
            orderbook_imbalance=None,
            bid_depth_total=None,
            ask_depth_total=None,
            price_velocity_bps=None,
            trend_5s_guard_triggered=None,
            trend_5s_guard_action=None,
            trend_5s_at_order=None,
            as_trailing_gate_action=None,
            as_trailing_gate_rate=None,
            as_trailing_gate_offset_mult=None,
            ev_score_pretrade=None,
            decision_path=None,
            skip_gate_budget_regime=None,
            skip_gate_budget_remaining=None,
            skip_gate_budget_exhausted=None,
        )

    def test_build_skip_fill_record_from_context(self) -> None:
        context = SkipFillRecordContext(
            cycle_id="c1",
            timestamp=123.0,
            side="buy",
            order_price=10.0,
            order_quantity=0.01,
            cancel_reason="skip_gate",
            spread_at_order=5.0,
            spread_offset_ratio=0.03,
            run_id="run-x",
            git_sha="abc123",
            regime_value="ranging",
        )
        extra = build_skip_fill_record_extra_fields(
            score=-0.5,
            reason="rule_skip_unknown_sell",
            model_used="rule",
            orderbook_imbalance=0.1,
            bid_depth_total=5.0,
            ask_depth_total=6.0,
            as_prob=0.2,
            threshold_used=0.0,
            hour_offset=0.1,
            price_velocity_bps=2.5,
            ev_score_pretrade=-0.3,
            decision_path="rule",
        )

        record = build_skip_fill_record_from_context(context=context, extra_fields=extra)

        assert isinstance(record, FillRecord)
        assert record.cycle_id == "c1"
        assert record.cancel_reason == "skip_gate"
        assert record.skip_gate_model_used == "rule"
        assert record.price_velocity_bps == 2.5
