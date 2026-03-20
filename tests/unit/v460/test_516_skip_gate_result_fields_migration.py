from __future__ import annotations

from dataclasses import dataclass

from ztb.ml.skip_gate_result_fields import (
    SkipDecisionResultFields,
    build_skip_decision_result_fields,
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
        )
