from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

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


class _PipelineHarness(MultiplicativePipelineMixin):
    def __init__(self, config: FillTestConfig) -> None:
        self.config = config
        self._maker_price = SimpleNamespace(
            last_vg_triggered=False,
            get_robust_inputs=lambda side: (0.0, 0.0),
        )
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
        del side, spread_at_order, aggressive_when_multiplier_gt_one
        if offset_mult is None:
            return order_price, effective_offset_ratio, None, None
        new_ratio = effective_offset_ratio * offset_mult
        return order_price + 1.0, new_ratio, float(offset_mult), 1.0

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
        del side, spread_at_order, effective_offset_ratio, new_offset_ratio
        return order_price


def _run_pipeline(config: FillTestConfig) -> _PipelineResultStub:
    harness = _PipelineHarness(config)
    with patch(
        "scripts.v460.lib.offset_pipeline.OffsetPipelineResult",
        _PipelineResultStub,
    ):
        return harness._apply_offset_pipeline_multiplicative(
            side="sell",
            order_price=100.0,
            spread_at_order=10.0,
            effective_offset_ratio=0.05,
            sg_ev_score=-0.3,
            sg_velocity_offset_mult=1.2,
            sg_velocity_bps=12.0,
            sg_toxic_veto_offset_mult=1.1,
            sg_trend_5s_guard_offset_mult=1.3,
            trending_offset_mult=None,
            toxicity_offset_mult=1.4,
            sidecar_offset_bps=0.0,
            cycle_id="cycle_690",
            decision_trace_id="dt_690",
        )


class TestOffsetPipelineStageToggles:
    def test_offset_ev_stage_disabled_skips_ev_stage(self) -> None:
        result = _run_pipeline(
            FillTestConfig(
                offset_ev_stage_enabled=False,
                skip_gate_ev_as_offset_enabled=True,
                execution_final_clamp_enabled=False,
            )
        )

        payload = json.loads(result.executor_offset_stages_json or "{}")
        assert payload["ev"] is None
        assert result.ev_offset_applied is False

    def test_offset_ev_stage_enabled_preserves_ev_stage(self) -> None:
        result = _run_pipeline(
            FillTestConfig(
                offset_ev_stage_enabled=True,
                skip_gate_ev_as_offset_enabled=True,
                execution_final_clamp_enabled=False,
            )
        )

        payload = json.loads(result.executor_offset_stages_json or "{}")
        assert payload["ev"] is not None
        assert result.ev_offset_applied is True

    def test_offset_toxicity_stage_disabled_skips_toxicity(self) -> None:
        result = _run_pipeline(
            FillTestConfig(
                offset_toxicity_stage_enabled=False,
                execution_final_clamp_enabled=False,
            )
        )

        payload = json.loads(result.executor_offset_stages_json or "{}")
        assert payload["toxicity"] is None

    def test_offset_vg_supplement_stage_disabled_skips_vg_supplement(self) -> None:
        result = _run_pipeline(
            FillTestConfig(
                offset_vg_supplement_enabled=False,
                volatility_guard_velocity_threshold_bps=1.0,
                volatility_guard_offset_boost_factor=1.5,
                execution_final_clamp_enabled=False,
            )
        )

        payload = json.loads(result.executor_offset_stages_json or "{}")
        assert payload["vg_supp"] is None

    def test_exec_stages_json_keeps_disabled_stages_as_null(self) -> None:
        result = _run_pipeline(
            FillTestConfig(
                offset_ev_stage_enabled=False,
                offset_toxicity_stage_enabled=False,
                offset_vg_supplement_enabled=False,
                execution_final_clamp_enabled=False,
            )
        )

        payload = json.loads(result.executor_offset_stages_json or "{}")
        assert payload["ev"] is None
        assert payload["toxicity"] is None
        assert payload["vg_supp"] is None

    def test_pipeline_stats_logs_every_hundred_cycles(self) -> None:
        config = FillTestConfig(execution_final_clamp_enabled=False)
        harness = _PipelineHarness(config)

        with (
            patch("scripts.v460.lib.offset_pipeline.OffsetPipelineResult", _PipelineResultStub),
            patch("scripts.v460.lib.multiplicative_pipeline.logger.info") as info_log,
        ):
            harness._apply_offset_pipeline_multiplicative(
                side="sell",
                order_price=100.0,
                spread_at_order=10.0,
                effective_offset_ratio=0.05,
                sg_ev_score=None,
                sg_velocity_offset_mult=None,
                sg_velocity_bps=None,
                trending_offset_mult=None,
                toxicity_offset_mult=1.0,
                sidecar_offset_bps=0.0,
                cycle_id="cycle_690_stats",
                decision_trace_id="dt_stats",
            )

        assert any("[690# pipeline_stats]" in str(call.args[0]) for call in info_log.call_args_list)
