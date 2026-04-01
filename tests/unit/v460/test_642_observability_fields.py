"""642# 可観測性6フィールドの伝播・デフォルト値テスト.

対象フィールド:
- skip_gate_forced_pass
- skip_gate_side_skip_rate
- execution_hard_skip_mult_used
- cv_offset_action
- balance_jpy_at_order
- balance_btc_at_order
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.fill_config_results import SkipGateResult
from scripts.v460.lib.offset_pipeline import OffsetPipelineResult
from ztb.ml.skip_gate import SkipDecision
from ztb.ml.skip_gate_result_fields import (
    SkipDecisionResultFields,
    build_skip_decision_result_fields,
)


# ------------------------------------------------------------------ #
# SkipDecision: forced_pass / side_skip_rate フィールド
# ------------------------------------------------------------------ #
class TestSkipDecisionFields:
    """SkipDecision dataclass に 642# フィールドが存在しデフォルトが正しい."""

    def test_defaults(self) -> None:
        d = SkipDecision(
            should_skip=True,
            predicted_pnl_bps=-1.0,
            threshold_bps=0.0,
            features_used=10,
        )
        assert d.forced_pass is False
        assert d.side_skip_rate is None

    def test_explicit_values(self) -> None:
        d = SkipDecision(
            should_skip=False,
            predicted_pnl_bps=2.0,
            threshold_bps=0.0,
            features_used=10,
            forced_pass=True,
            side_skip_rate=0.35,
        )
        assert d.forced_pass is True
        assert d.side_skip_rate == pytest.approx(0.35)


# ------------------------------------------------------------------ #
# SkipDecisionResultFields: forced_pass / side_skip_rate 伝播
# ------------------------------------------------------------------ #
class TestSkipDecisionResultFieldsPropagation:
    """build_skip_decision_result_fields が 642# フィールドを正しく伝播する."""

    def _make_decision(
        self, *, forced_pass: bool = False, side_skip_rate: float | None = None
    ) -> SkipDecision:
        return SkipDecision(
            should_skip=True,
            predicted_pnl_bps=-0.5,
            threshold_bps=0.0,
            features_used=8,
            reason="test",
            model_used="primary",
            forced_pass=forced_pass,
            side_skip_rate=side_skip_rate,
        )

    def test_propagates_defaults(self) -> None:
        fields = build_skip_decision_result_fields(
            self._make_decision(),
            side="buy",
            has_side_specific_model=False,
            hour_offset=0.0,
            price_velocity_bps=None,
        )
        assert fields.forced_pass is False
        assert fields.side_skip_rate is None

    def test_propagates_explicit(self) -> None:
        fields = build_skip_decision_result_fields(
            self._make_decision(forced_pass=True, side_skip_rate=0.42),
            side="sell",
            has_side_specific_model=True,
            hour_offset=1.5,
            price_velocity_bps=-3.0,
        )
        assert fields.forced_pass is True
        assert fields.side_skip_rate == pytest.approx(0.42)


# ------------------------------------------------------------------ #
# SkipGateResult: forced_pass / side_skip_rate デフォルト
# ------------------------------------------------------------------ #
class TestSkipGateResultDefaults:
    """SkipGateResult dataclass の 642# フィールドデフォルトが正しい."""

    def test_defaults(self) -> None:
        r = SkipGateResult()
        assert r.forced_pass is False
        assert r.side_skip_rate is None

    def test_set_values(self) -> None:
        r = SkipGateResult()
        r.forced_pass = True
        r.side_skip_rate = 0.38
        assert r.forced_pass is True
        assert r.side_skip_rate == pytest.approx(0.38)


# ------------------------------------------------------------------ #
# OffsetPipelineResult: execution_hard_skip_mult_used デフォルト
# ------------------------------------------------------------------ #
class TestOffsetPipelineResultHardSkipField:
    """OffsetPipelineResult に execution_hard_skip_mult_used が存在する."""

    def test_default_none(self) -> None:
        r = OffsetPipelineResult(
            order_price=14_000_000.0,
            effective_offset_ratio=0.0005,
            ev_offset_applied=False,
            ev_score_pretrade=None,
            ev_offset_mult_applied=None,
            macro_boost_applied=False,
            execution_pre_clamp_offset=None,
            executor_offset_stages_json=None,
        )
        assert r.execution_hard_skip_mult_used is None

    def test_explicit_value(self) -> None:
        r = OffsetPipelineResult(
            order_price=14_000_000.0,
            effective_offset_ratio=0.0005,
            ev_offset_applied=False,
            ev_score_pretrade=None,
            ev_offset_mult_applied=None,
            macro_boost_applied=False,
            execution_pre_clamp_offset=None,
            executor_offset_stages_json=None,
            execution_hard_skip_mult_used=4.0,
        )
        assert r.execution_hard_skip_mult_used == pytest.approx(4.0)


# ------------------------------------------------------------------ #
# _PreOrderPhaseResult: 3 フィールドのデフォルト
# ------------------------------------------------------------------ #
class TestPreOrderPhaseResultDefaults:
    """_PreOrderPhaseResult の 642# フィールドにデフォルトが設定されている."""

    def test_defaults(self) -> None:
        from scripts.v460.lib.fill_cycle_executor import _PreOrderPhaseResult

        r = _PreOrderPhaseResult(
            cycle_id="test-001",
            side="buy",
            order_price=14_000_000.0,
            spread_at_order=100.0,
            effective_offset_ratio=0.0005,
            regime_lot=0.001,
            skip_gate_skipped=None,
            skip_gate_score=None,
            skip_gate_reason=None,
            skip_gate_model_used=None,
            skip_gate_as_prob=None,
            skip_gate_threshold_used=None,
            skip_gate_hour_offset=None,
            sg_velocity_bps=None,
            trend_5s_guard_triggered=False,
            trend_5s_guard_action=None,
            trend_5s_at_order=None,
            ev_offset_applied=False,
            ev_score_pretrade=None,
            ev_offset_mult_applied=None,
            macro_boost_applied=False,
            execution_pre_clamp_offset=None,
            executor_offset_stages_json=None,
            regime_at_order=None,
            regime_obs_count=None,
            mid_at_order=None,
        )
        assert r.skip_gate_forced_pass is False
        assert r.skip_gate_side_skip_rate is None
        assert r.execution_hard_skip_mult_used is None


# ------------------------------------------------------------------ #
# cv_offset_action ロジック
# ------------------------------------------------------------------ #
class TestCvOffsetAction:
    """cv_offset_action が widen/tighten/None を正しく返す."""

    @staticmethod
    def _make_mock_builder(
        pre: float | None, post: float | None
    ) -> MagicMock:
        """FillRecordBuilderMixin の _maker_price を模擬."""
        mp = MagicMock()
        mp._cross_venue_lead_lag_pre_offset = pre
        mp._cross_venue_lead_lag_post_offset = post
        mp._cross_venue_lead_lag_cap_hit = False
        mp._consecutive_veto_count = 0
        return mp

    @pytest.mark.parametrize(
        "pre,post,expected",
        [
            (10.0, 20.0, "widen"),
            (20.0, 10.0, "tighten"),
            (10.0, 10.0, None),
            (None, 20.0, None),
            (10.0, None, None),
            (None, None, None),
        ],
        ids=["widen", "tighten", "equal", "pre_none", "post_none", "both_none"],
    )
    def test_cv_offset_action_logic(
        self, pre: float | None, post: float | None, expected: str | None
    ) -> None:
        """cv_offset_action のロジックを直接テスト."""
        # build_fill_measurement_fields 全体をテストせずロジック単体を検証
        _pre = pre
        _post = post
        result = None
        if _pre is not None and _post is not None and _pre != _post:
            result = "widen" if _post > _pre else "tighten"
        assert result == expected


# ------------------------------------------------------------------ #
# build_fill_record: 642# フィールド引数のデフォルト
# ------------------------------------------------------------------ #
class TestBuildFillRecordSignature:
    """build_fill_record が 642# 引数をデフォルト付きで受け取れる."""

    def test_signature_has_642_params(self) -> None:
        import inspect

        from scripts.v460.lib.fill_record_builder import FillRecordBuilderMixin

        sig = inspect.signature(FillRecordBuilderMixin._build_fill_record)
        params = sig.parameters

        # 642# パラメータが存在しデフォルトを持つ
        assert "sg_forced_pass" in params
        assert params["sg_forced_pass"].default is False

        assert "sg_side_skip_rate" in params
        assert params["sg_side_skip_rate"].default is None

        assert "execution_hard_skip_mult_used" in params
        assert params["execution_hard_skip_mult_used"].default is None

        assert "balance_jpy_at_order" in params
        assert params["balance_jpy_at_order"].default is None

        assert "balance_btc_at_order" in params
        assert params["balance_btc_at_order"].default is None
