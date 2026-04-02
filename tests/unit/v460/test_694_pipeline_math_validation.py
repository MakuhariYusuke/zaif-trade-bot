from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.multiplicative_pipeline import MultiplicativePipelineMixin


@dataclass
class _PipelineResultStub:
    order_price: float
    effective_offset_ratio: float
    ev_offset_applied: bool
    ev_score_pretrade: float | None
    ev_offset_mult_applied: float | None
    macro_boost_applied: bool
    execution_pre_clamp_offset: float | None
    executor_offset_stages_json: str | None
    execution_hard_skip_mult_used: float | None = None
    early_return_record: object | None = None


class _Harness(MultiplicativePipelineMixin):
    def __init__(self, config: FillTestConfig) -> None:
        self.config = config
        self._maker_price = SimpleNamespace(last_vg_triggered=False, get_robust_inputs=lambda side: (0.0, 0.0))
        self._last_macro_trend = None
        self._alert_offset_mult = 1.0
        self._cycle_count = 100
        self._current_regime_value = lambda: "unknown"

    def _apply_offset_multiplier(
        self,
        *,
        side: str,
        order_price: float,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        offset_mult: float | None,
        aggressive_when_multiplier_gt_one: bool = False,
    ) -> tuple[float, float, float | None, float | None]:
        del side, order_price, spread_at_order, aggressive_when_multiplier_gt_one
        if offset_mult is None:
            return 0.0, effective_offset_ratio, None, None
        new_ratio = effective_offset_ratio * offset_mult
        return 0.0, new_ratio, float(offset_mult), float(new_ratio - effective_offset_ratio)

    def _make_cycle_skip_record(self, **kwargs: object) -> dict[str, object]:
        return dict(kwargs)

    def _recalc_price_with_new_offset(
        self,
        side: str,
        order_price: float,
        spread_at_order: float,
        effective_offset_ratio: float,
        new_offset_ratio: float,
    ) -> float:
        del side, order_price, spread_at_order, effective_offset_ratio, new_offset_ratio
        return 0.0


def _run_pipeline(
    config: FillTestConfig,
    *,
    ev_score: float | None,
    vel_mult: float | None,
    tox_mult: float,
    vg_threshold: float = 1.0,
    velocity_bps: float = 12.0,
) -> _PipelineResultStub:
    harness = _Harness(config)
    with patch("scripts.v460.lib.offset_pipeline.OffsetPipelineResult", _PipelineResultStub):
        return harness._apply_offset_pipeline_multiplicative(
            side="sell",
            order_price=100.0,
            spread_at_order=10.0,
            effective_offset_ratio=100.0,
            sg_ev_score=ev_score,
            sg_velocity_offset_mult=vel_mult,
            sg_velocity_bps=velocity_bps,
            sg_toxic_veto_offset_mult=None,
            sg_trend_5s_guard_offset_mult=None,
            sg_as_trailing_offset_mult=None,
            trending_offset_mult=None,
            toxicity_offset_mult=tox_mult,
            sidecar_offset_bps=0.0,
            cycle_id="cycle_694",
            decision_trace_id="dt_694",
        )


class TestPipelineMathValidation:
    def test_all_stages_enabled_math(self) -> None:
        config = FillTestConfig(
            offset_ev_stage_enabled=True,
            skip_gate_ev_as_offset_enabled=True,
            skip_gate_ev_offset_sensitivity=1.0,
            skip_gate_ev_offset_min_mult=0.5,
            skip_gate_ev_offset_max_mult=2.0,
            offset_toxicity_stage_enabled=True,
            offset_vg_supplement_enabled=True,
            volatility_guard_velocity_threshold_bps=1.0,
            volatility_guard_offset_boost_factor=1.1,
            execution_final_clamp_enabled=False,
        )
        result = _run_pipeline(config, ev_score=0.2, vel_mult=None, tox_mult=1.4)
        # current runtime applies widening-only toxicity stage:
        # 100 × 1.2 × 1.4 × 1.1 = 184.8
        assert result.effective_offset_ratio == pytest.approx(184.8)

    def test_ev_stage_disabled(self) -> None:
        config = FillTestConfig(
            offset_ev_stage_enabled=False,
            skip_gate_ev_as_offset_enabled=True,
            offset_toxicity_stage_enabled=True,
            offset_vg_supplement_enabled=True,
            volatility_guard_velocity_threshold_bps=1.0,
            volatility_guard_offset_boost_factor=1.1,
            execution_final_clamp_enabled=False,
        )
        result = _run_pipeline(config, ev_score=0.5, vel_mult=None, tox_mult=1.4)
        # 100 × 1.0 × 1.4 × 1.1 = 154.0
        assert result.effective_offset_ratio == pytest.approx(154.0)
        payload = json.loads(result.executor_offset_stages_json or "{}")
        assert payload["ev"] is None

    def test_toxicity_stage_disabled(self) -> None:
        config = FillTestConfig(
            offset_ev_stage_enabled=True,
            skip_gate_ev_as_offset_enabled=True,
            skip_gate_ev_offset_sensitivity=1.0,
            skip_gate_ev_offset_min_mult=0.5,
            skip_gate_ev_offset_max_mult=2.0,
            offset_toxicity_stage_enabled=False,
            offset_vg_supplement_enabled=True,
            volatility_guard_velocity_threshold_bps=1.0,
            volatility_guard_offset_boost_factor=1.1,
            execution_final_clamp_enabled=False,
        )
        result = _run_pipeline(config, ev_score=0.2, vel_mult=None, tox_mult=1.4)
        # 100 × 1.2 × 1.0 × 1.1 = 132.0
        assert result.effective_offset_ratio == pytest.approx(132.0)

    def test_vg_stage_disabled(self) -> None:
        config = FillTestConfig(
            offset_ev_stage_enabled=True,
            skip_gate_ev_as_offset_enabled=True,
            skip_gate_ev_offset_sensitivity=1.0,
            skip_gate_ev_offset_min_mult=0.5,
            skip_gate_ev_offset_max_mult=2.0,
            offset_toxicity_stage_enabled=True,
            offset_vg_supplement_enabled=False,
            execution_final_clamp_enabled=False,
        )
        result = _run_pipeline(config, ev_score=0.2, vel_mult=None, tox_mult=1.4)
        # 100 × 1.2 × 1.4 × 1.0 = 168.0
        assert result.effective_offset_ratio == pytest.approx(168.0)

    def test_all_stages_disabled(self) -> None:
        config = FillTestConfig(
            offset_ev_stage_enabled=False,
            offset_toxicity_stage_enabled=False,
            offset_vg_supplement_enabled=False,
            execution_final_clamp_enabled=False,
        )
        result = _run_pipeline(config, ev_score=0.5, vel_mult=None, tox_mult=0.8)
        assert result.effective_offset_ratio == pytest.approx(100.0)

    def test_pipeline_stats_log_interval(self) -> None:
        config = FillTestConfig(execution_final_clamp_enabled=False)
        harness = _Harness(config)
        with (
            patch("scripts.v460.lib.offset_pipeline.OffsetPipelineResult", _PipelineResultStub),
            patch("scripts.v460.lib.multiplicative_pipeline.logger.info") as info_log,
        ):
            harness._apply_offset_pipeline_multiplicative(
                side="sell",
                order_price=100.0,
                spread_at_order=10.0,
                effective_offset_ratio=100.0,
                sg_ev_score=None,
                sg_velocity_offset_mult=None,
                sg_velocity_bps=12.0,
                trending_offset_mult=None,
                toxicity_offset_mult=1.0,
                sidecar_offset_bps=0.0,
                cycle_id="cycle_stats",
                decision_trace_id="dt_stats",
            )
        assert any("[690# pipeline_stats]" in str(call.args[0]) for call in info_log.call_args_list)
