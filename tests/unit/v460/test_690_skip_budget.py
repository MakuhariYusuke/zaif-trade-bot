from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_results import SkipGateResult
from scripts.v460.lib.skip_gate_budget import BucketedSkipBudget
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from ztb.ml.skip_gate import SkipDecision


class _AdapterStub:
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[object]:
        del symbol, limit
        return []

    async def get_orderbook(self, symbol: str, depth: int = 1) -> None:
        del symbol, depth
        return None


class _StaticGate:
    def __init__(self, decision: SkipDecision) -> None:
        self._decision = decision
        self.config = type("Cfg", (), {"use_ob_features": False})()
        self.metadata: dict[str, object] = {}
        self.feature_cols: list[str] = []

    def evaluate(
        self,
        features: dict[str, object],
        *,
        side: str | None = None,
        regime: str | None = None,
        threshold_offset: float = 0.0,
    ) -> SkipDecision:
        del features, side, regime, threshold_offset
        return self._decision


def _make_config(**overrides: object) -> FillTestConfig:
    base = FillTestConfig(
        skip_gate_enabled=True,
        skip_gate_budget_enabled=True,
        skip_gate_budget_window_min=60,
        skip_gate_budget_limits={"default": 1},
        skip_gate_primary_max_consecutive_skip=0,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _make_evaluator(config: FillTestConfig, decision: SkipDecision) -> SkipGateEvaluator:
    with patch(
        "scripts.v460.lib.skip_gate_evaluator.SkipGateEvaluator.__init__",
        lambda self, *args, **kwargs: None,
    ):
        evaluator = SkipGateEvaluator.__new__(SkipGateEvaluator)
    evaluator._config = config
    evaluator._project_root = Path("/tmp")
    evaluator._skip_gate = _StaticGate(decision)
    evaluator._gate_buy = None
    evaluator._gate_sell = None
    evaluator._gate_path_buy = None
    evaluator._gate_path_sell = None
    evaluator._model_file_hash_buy = ""
    evaluator._model_file_hash_sell = ""
    evaluator._gate_alt_buy = None
    evaluator._gate_alt_sell = None
    evaluator._gate_path_alt_buy = None
    evaluator._gate_path_alt_sell = None
    evaluator._model_file_hash_alt_buy = ""
    evaluator._model_file_hash_alt_sell = ""
    evaluator._ev_consecutive_skip_count = 0
    evaluator._primary_consecutive_skip_count = 0
    evaluator._toxic_veto_consecutive_count = 0
    evaluator._ob_fetch_fail_count = 0
    evaluator._ob_fetch_total_count = 0
    evaluator._gate_path = None
    evaluator._model_file_hash = ""
    evaluator._last_reload_check = 0.0
    evaluator._budget = (
        BucketedSkipBudget(config) if config.skip_gate_budget_enabled else None
    )
    evaluator._check_and_reload_model = lambda: None
    evaluator._try_ev_weighted_decision = lambda *args, **kwargs: None
    return evaluator


def _skip_decision(reason: str = "model_skip") -> SkipDecision:
    return SkipDecision(
        should_skip=True,
        predicted_pnl_bps=-1.5,
        threshold_bps=0.0,
        features_used=12,
        reason=reason,
        model_used="primary",
        as_probability=0.8,
        threshold_used=0.5,
    )


def _run_eval(
    evaluator: SkipGateEvaluator,
    *,
    side: str = "buy",
    regime: str | None = "unknown",
    now_ts: float = 1_700_000_000.0,
) -> SkipGateResult:
    async def _inner() -> SkipGateResult:
        return await evaluator.evaluate(
            side=side,
            cycle_id="cycle_1",
            order_price=100.0,
            spread_at_order=1.0,
            effective_offset_ratio=0.05,
            adapter=_AdapterStub(),
            symbol="btc_jpy",
            current_lot=0.01,
            run_id="run_1",
            git_sha="sha_1",
            regime_value=regime,
            last_imbalance=0.1,
            last_bid_depth=1.0,
            last_ask_depth=1.0,
            imbalance_enabled=False,
            decision_trace_id="dt_690",
        )

    with (
        patch("scripts.v460.lib.skip_gate_evaluator.time.time", return_value=now_ts),
        patch("scripts.v460.lib.skip_gate_budget.time.time", return_value=now_ts),
    ):
        return asyncio.run(_inner())


class TestBucketedSkipBudgetConfig:
    def test_get_budget_limit_uses_default_and_side_override(self) -> None:
        cfg = _make_config(
            skip_gate_budget_limits={
                "default": 8,
                "trending_up": {"sell": 3, "buy": 12},
            }
        )

        assert cfg.get_budget_limit("trending_up", "sell") == 3
        assert cfg.get_budget_limit("trending_up", "buy") == 12
        assert cfg.get_budget_limit("unknown", "buy") == 8

    def test_from_yaml_parses_budget_settings(self) -> None:
        cfg = FillTestConfig.from_yaml(
            {
                "skip_gate": {
                    "enabled": True,
                    "budget_enabled": True,
                    "budget_window_min": 30,
                    "budget_limits": {
                        "default": 6,
                        "trending_up": {"sell": 2, "buy": 10},
                    },
                }
            }
        )

        assert cfg.skip_gate_budget_enabled is True
        assert cfg.skip_gate_budget_window_min == 30
        assert cfg.skip_gate_budget_limits == {
            "default": 6,
            "trending_up": {"sell": 2, "buy": 10},
        }


class TestBucketedSkipBudgetEvaluator:
    def test_budget_disabled_keeps_legacy_behavior(self) -> None:
        cfg = _make_config(skip_gate_budget_enabled=False)
        evaluator = _make_evaluator(cfg, _skip_decision())

        result = _run_eval(evaluator)

        assert result.skipped is True
        assert result.budget_regime is None
        assert result.budget_remaining is None
        assert result.early_return_record is not None
        assert result.early_return_record.skip_gate_budget_remaining is None

    def test_budget_limit_forces_pass_when_exhausted(self) -> None:
        cfg = _make_config(skip_gate_budget_limits={"default": 1})
        evaluator = _make_evaluator(cfg, _skip_decision())

        first = _run_eval(evaluator, now_ts=1000.0)
        second = _run_eval(evaluator, now_ts=1001.0)

        assert first.skipped is True
        assert first.budget_remaining == 0
        assert first.budget_exhausted is False
        assert second.skipped is False
        assert second.reason == "budget_exhausted_pass"
        assert second.budget_exhausted is True
        assert second.budget_remaining == 0

    def test_window_rotation_resets_budget(self) -> None:
        cfg = _make_config(skip_gate_budget_limits={"default": 1}, skip_gate_budget_window_min=1)
        evaluator = _make_evaluator(cfg, _skip_decision())

        first = _run_eval(evaluator, now_ts=1000.0)
        second = _run_eval(evaluator, now_ts=1061.0)

        assert first.skipped is True
        assert second.skipped is True
        assert second.budget_exhausted is False

    def test_regime_side_buckets_are_independent(self) -> None:
        cfg = _make_config(
            skip_gate_budget_limits={
                "default": 1,
                "trending_up": {"sell": 1, "buy": 3},
            }
        )
        evaluator = _make_evaluator(cfg, _skip_decision())

        sell_first = _run_eval(evaluator, side="sell", regime="trending_up", now_ts=1000.0)
        sell_second = _run_eval(evaluator, side="sell", regime="trending_up", now_ts=1001.0)
        buy_first = _run_eval(evaluator, side="buy", regime="trending_up", now_ts=1002.0)

        assert sell_first.skipped is True
        assert sell_second.reason == "budget_exhausted_pass"
        assert buy_first.skipped is True
        assert buy_first.budget_remaining == 2

    def test_budget_counters_follow_raw_skip_even_in_bypass_mode(self) -> None:
        cfg = _make_config(skip_gate_bypass_mode=True, skip_gate_budget_limits={"default": 1})
        evaluator = _make_evaluator(cfg, _skip_decision())

        result = _run_eval(evaluator, now_ts=1000.0)

        assert result.skipped is False
        assert result.bypassed is True
        assert result.budget_remaining == 0
        assert evaluator._budget is not None
        with patch("scripts.v460.lib.skip_gate_budget.time.time", return_value=1000.0):
            assert evaluator._budget.get_state("unknown", "buy").skip_count == 1

    def test_budget_ceiling_updates_without_resetting_counts(self) -> None:
        cfg = _make_config(skip_gate_budget_limits={"default": 1})
        evaluator = _make_evaluator(cfg, _skip_decision())

        first = _run_eval(evaluator, now_ts=1000.0)
        cfg.skip_gate_budget_limits = {"default": 3}
        second = _run_eval(evaluator, now_ts=1001.0)

        assert first.skipped is True
        assert second.skipped is True
        assert second.budget_exhausted is False
        assert second.budget_remaining == 1
        assert evaluator._budget is not None
        with patch("scripts.v460.lib.skip_gate_budget.time.time", return_value=1001.0):
            assert evaluator._budget.get_state("unknown", "buy").skip_count == 2

    def test_primary_safety_valve_coexists_with_budget(self) -> None:
        cfg = _make_config(
            skip_gate_budget_limits={"default": 10},
            skip_gate_primary_max_consecutive_skip=2,
        )
        evaluator = _make_evaluator(cfg, _skip_decision())

        first = _run_eval(evaluator, now_ts=1000.0)
        second = _run_eval(evaluator, now_ts=1001.0)

        assert first.skipped is True
        assert second.skipped is False
        assert second.reason == "primary_safety_valve_pass"
        assert second.budget_exhausted is False

    def test_fill_record_contains_budget_observability(self) -> None:
        cfg = _make_config(skip_gate_budget_limits={"ranging": {"buy": 1}})
        evaluator = _make_evaluator(cfg, _skip_decision())

        result = _run_eval(evaluator, regime="ranging", now_ts=1000.0)

        assert result.early_return_record is not None
        assert result.early_return_record.skip_gate_budget_regime == "ranging"
        assert result.early_return_record.skip_gate_budget_remaining == 0
        assert result.early_return_record.skip_gate_budget_exhausted is False
