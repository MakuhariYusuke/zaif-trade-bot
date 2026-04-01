from __future__ import annotations

import asyncio

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_results import SkipGateResult
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from tests.unit.v460.test_skip_gate_v3 import _AdapterStub, _make_bypassed_evaluator
from ztb.trading.common import cancel_reasons as CR


def _make_direct_evaluator(config: FillTestConfig) -> SkipGateEvaluator:
    evaluator = SkipGateEvaluator.__new__(SkipGateEvaluator)
    evaluator._config = config
    return evaluator


def _evaluate_guard(
    config: FillTestConfig,
    *,
    side: str = "sell",
    mid_trend_bps: float | None = None,
) -> SkipGateResult:
    evaluator = _make_bypassed_evaluator(config)
    evaluator._skip_gate.config.use_ob_features = False
    evaluator._ob_fetch_fail_count = 0
    evaluator._ob_fetch_total_count = 0
    return asyncio.run(
        evaluator.evaluate(
            side=side,
            cycle_id="trend_guard",
            order_price=15_000_000.0,
            spread_at_order=2000.0,
            effective_offset_ratio=0.05,
            adapter=_AdapterStub(),
            symbol="btc_jpy",
            current_lot=0.001,
            run_id="test_run",
            git_sha=None,
            regime_value="ranging",
            last_imbalance=0.05,
            last_bid_depth=100.0,
            last_ask_depth=100.0,
            imbalance_enabled=True,
            mid_trend_bps=mid_trend_bps,
        ),
    )


class TestTrend5sSellGuard:
    def test_sell_soft_boost(self) -> None:
        evaluator = _make_direct_evaluator(FillTestConfig(
            skip_gate_enabled=True,
            skip_gate_model_path="models/v460/skip_gate_rb30.pkl",
            trend_5s_sell_guard_enabled=True,
            trend_5s_sell_guard_threshold_bps=0.5,
            trend_5s_sell_guard_hard_veto_threshold_bps=3.0,
            trend_5s_sell_guard_offset_boost_factor=1.5,
        ))
        action, mult = evaluator._apply_trend_5s_sell_guard(side="sell", trend_5s_bps=1.0)
        assert action == "boost"
        assert mult == pytest.approx(1.5)

    def test_sell_hard_veto(self) -> None:
        cfg = FillTestConfig(
            skip_gate_enabled=True,
            skip_gate_model_path="models/v460/skip_gate_rb30.pkl",
            trend_5s_sell_guard_enabled=True,
            trend_5s_sell_guard_threshold_bps=0.5,
            trend_5s_sell_guard_hard_veto_threshold_bps=3.0,
        )
        result = _evaluate_guard(cfg, mid_trend_bps=4.0)
        assert result.skipped is True
        assert result.reason == "rule_trend_5s_sell_guard_veto"
        assert result.early_return_record is not None
        assert result.early_return_record.cancel_reason == CR.TREND_5S_SELL_GUARD_VETO
        assert result.early_return_record.trend_5s_guard_action == "veto"
        assert result.early_return_record.trend_5s_at_order == pytest.approx(4.0)

    def test_sell_below_threshold(self) -> None:
        evaluator = _make_direct_evaluator(FillTestConfig(
            skip_gate_enabled=True,
            skip_gate_model_path="models/v460/skip_gate_rb30.pkl",
            trend_5s_sell_guard_enabled=True,
        ))
        action, mult = evaluator._apply_trend_5s_sell_guard(side="sell", trend_5s_bps=0.3)
        assert action == "none"
        assert mult is None

    def test_buy_not_affected(self) -> None:
        evaluator = _make_direct_evaluator(FillTestConfig(
            skip_gate_enabled=True,
            skip_gate_model_path="models/v460/skip_gate_rb30.pkl",
            trend_5s_sell_guard_enabled=True,
        ))
        action, mult = evaluator._apply_trend_5s_sell_guard(side="buy", trend_5s_bps=5.0)
        assert action == "none"
        assert mult is None

    def test_disabled(self) -> None:
        evaluator = _make_direct_evaluator(FillTestConfig(
            skip_gate_enabled=True,
            skip_gate_model_path="models/v460/skip_gate_rb30.pkl",
            trend_5s_sell_guard_enabled=False,
        ))
        action, mult = evaluator._apply_trend_5s_sell_guard(side="sell", trend_5s_bps=5.0)
        assert action == "none"
        assert mult is None

    def test_negative_trend_5s(self) -> None:
        evaluator = _make_direct_evaluator(FillTestConfig(
            skip_gate_enabled=True,
            skip_gate_model_path="models/v460/skip_gate_rb30.pkl",
            trend_5s_sell_guard_enabled=True,
        ))
        action, mult = evaluator._apply_trend_5s_sell_guard(side="sell", trend_5s_bps=-2.0)
        assert action == "none"
        assert mult is None

    def test_boost_factor_applied(self) -> None:
        evaluator = _make_direct_evaluator(FillTestConfig(
            trend_5s_sell_guard_enabled=True,
            trend_5s_sell_guard_threshold_bps=0.5,
            trend_5s_sell_guard_hard_veto_threshold_bps=3.0,
            trend_5s_sell_guard_offset_boost_factor=1.8,
        ))
        action, mult = evaluator._apply_trend_5s_sell_guard(
            side="sell",
            trend_5s_bps=0.8,
        )
        assert action == "boost"
        assert mult == pytest.approx(1.8)


class TestTrend5sConfig:
    def test_yaml_loading(self) -> None:
        cfg = FillTestConfig.from_yaml(
            {
                "trend_5s_sell_guard": {
                    "enabled": True,
                    "threshold_bps": 0.5,
                    "hard_veto_threshold_bps": 3.0,
                    "offset_boost_factor": 1.5,
                },
            },
        )
        assert cfg.trend_5s_sell_guard.enabled is True
        assert cfg.trend_5s_sell_guard.threshold_bps == pytest.approx(0.5)

    def test_default_values(self) -> None:
        cfg = FillTestConfig()
        guard = cfg.trend_5s_sell_guard
        assert guard.enabled is False
        assert guard.threshold_bps == pytest.approx(0.5)
        assert guard.hard_veto_threshold_bps == pytest.approx(3.0)
        assert guard.offset_boost_factor == pytest.approx(1.5)
