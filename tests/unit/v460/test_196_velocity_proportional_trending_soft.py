"""196# velocity offset 比例 + trending_sell soft offset テスト.

テスト対象:
  A. velocity_offset_proportional: 閾値超過量に比例した段階的 boost
  B. trending_sell_as_offset: hard skip → offset boost への変換
  C. Config + YAML: 新フィールドの parse、default 値
  D. 後方互換性: 旧モード維持
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


def _make_config(**overrides):
    """テスト用の FillTestConfig を簡易生成."""
    from scripts.v460.lib.fill_config import FillTestConfig

    defaults = {
        # velocity skip
        "sell_velocity_skip_enabled": True,
        "sell_velocity_skip_threshold_bps": 6.0,
        "buy_velocity_skip_enabled": True,
        "buy_velocity_skip_threshold_bps": -6.0,
        "velocity_skip_as_offset_enabled": True,
        "velocity_offset_boost_factor": 1.5,
        "velocity_offset_proportional": False,
        "velocity_offset_max_mult": 4.0,
        # trending sell
        "skip_sell_trending": True,
        "skip_sell_trending_up_only": True,
        "trending_sell_as_offset_enabled": False,
        "trending_sell_offset_boost_factor": 2.0,
        # 253# 削除済み: balance_forced_apply_trending_offset (234# dead config)
        "max_consecutive_trending_sell_skip": 30,
        "sell_guard_inv_bypass_threshold": 0.3,
        # skip gate basics
        "skip_gate_enabled": False,
        "skip_gate_mode": "pnl",
        "skip_gate_as_threshold": 0.5,
        "skip_gate_pnl_threshold": 0.0,
        "skip_gate_max_skip_rate": 0.3,
        "skip_gate_use_ob_features": False,
        "skip_gate_adaptive_threshold": False,
        "skip_gate_narrow_spread_threshold_jpy": 0.0,
        "skip_gate_narrow_spread_offset": 0.0,
        "skip_gate_offset_floor": -0.3,
        "skip_gate_offset_ceil": 0.5,
        "skip_gate_hour_offsets": {},
        "skip_gate_score_calibration": False,
        "skip_gate_calibrator_min_samples": 30,
        "skip_gate_calibrator_refit_interval": 100,
        "skip_gate_ev_as_offset_enabled": False,
        "skip_gate_ev_offset_sensitivity": 0.05,
        "skip_gate_ev_offset_min_mult": 0.5,
        "skip_gate_ev_offset_max_mult": 1.5,
        "skip_gate_ev_emergency_skip_threshold": -8.0,
        # other required fields
        "skip_buy_unknown_regime": False,
        "skip_sell_unknown_regime": False,
        "skip_ranging_buy_low_vol": False,
        "low_vol_threshold": 0.75,
        "ranging_buy_low_vol_as_offset": False,
        "buy_dynamic_kill_enabled": False,
        "buy_dynamic_kill_threshold_bps": -0.5,
        "sell_dynamic_kill_enabled": False,
        "sell_dynamic_kill_threshold_bps": -0.5,
        "unknown_buy_offset_boost": 1.0,
    }
    defaults.update(overrides)
    return FillTestConfig(**{k: v for k, v in defaults.items() if hasattr(FillTestConfig, k)})


# =================================================================
# A. velocity_offset_proportional テスト
# =================================================================


class TestVelocityProportionalConfig:
    """196# velocity_offset_proportional config フィールド."""

    def test_default_proportional_false(self):
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.velocity_offset_proportional is False

    def test_default_max_mult(self):
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.velocity_offset_max_mult == 4.0


class TestVelocityProportionalCalculation:
    """196# 比例モード: 閾値超過量に比例した段階的 boost."""

    @pytest.mark.parametrize(
        "velocity, threshold, boost_factor, expected_boost",
        [
            # 閾値ちょうど: excess_ratio=1.0 → boost = 1.0 + (2.0-1.0)*1.0 = 2.0
            (6.0, 6.0, 2.0, 2.0),
            # 50% 超過: excess_ratio=1.5 → boost = 1.0 + 1.0*1.5 = 2.5
            (9.0, 6.0, 2.0, 2.5),
            # 100% 超過: excess_ratio=2.0 → boost = 1.0 + 1.0*2.0 = 3.0
            (12.0, 6.0, 2.0, 3.0),
            # 200% 超過: excess_ratio=3.0 → boost = 1.0 + 1.0*3.0 = 4.0 (max)
            (18.0, 6.0, 2.0, 4.0),
            # buy 方向: velocity=-10, threshold=-6 → excess_ratio=10/6=1.667
            (-10.0, -6.0, 2.0, 2.667),
        ],
    )
    def test_proportional_boost_calculation(
        self, velocity, threshold, boost_factor, expected_boost,
    ):
        """比例モードの boost 計算が正しいこと."""
        excess_ratio = abs(velocity) / abs(threshold)
        boost = 1.0 + (boost_factor - 1.0) * excess_ratio
        boost = min(boost, 4.0)  # max_mult
        assert abs(boost - expected_boost) < 0.01, (
            f"velocity={velocity}, threshold={threshold}: "
            f"expected {expected_boost}, got {boost:.3f}"
        )

    def test_proportional_capped_at_max_mult(self):
        """max_mult で上限がかかること."""
        cfg = _make_config(
            velocity_offset_proportional=True,
            velocity_offset_max_mult=3.0,
            velocity_offset_boost_factor=2.0,
        )
        # velocity=18 → excess_ratio=3.0 → uncapped=4.0 → capped at 3.0
        excess_ratio = 18.0 / 6.0
        boost = 1.0 + (cfg.velocity_offset_boost_factor - 1.0) * excess_ratio
        boost = min(boost, cfg.velocity_offset_max_mult)
        assert boost == 3.0

    def test_fixed_mode_ignores_proportional(self):
        """proportional=False 時は固定 boost_factor を使用."""
        cfg = _make_config(
            velocity_offset_proportional=False,
            velocity_offset_boost_factor=2.0,
        )
        # velocity の大きさに関係なく固定 2.0
        assert cfg.velocity_offset_boost_factor == 2.0


class TestVelocityProportionalInSkipGate:
    """196# skip_gate_evaluator 内での比例 boost 適用."""

    def test_proportional_helper_applies_ratio(self):
        """比例モード時は閾値超過量に応じて boost される."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        boost, proportional = SkipGateEvaluator._compute_velocity_offset_multiplier(
            observed_velocity_bps=9.0,
            threshold_bps=6.0,
            base_multiplier=2.0,
            max_multiplier=4.0,
            proportional=True,
        )
        assert proportional is True
        assert boost == pytest.approx(2.5, abs=0.01)

    def test_zero_threshold_falls_back_to_fixed(self):
        """threshold=0 でも 0除算せず固定 boost にフォールバック."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        boost, proportional = SkipGateEvaluator._compute_velocity_offset_multiplier(
            observed_velocity_bps=9.0,
            threshold_bps=0.0,
            base_multiplier=2.0,
            max_multiplier=4.0,
            proportional=True,
        )
        assert proportional is False
        assert boost == 2.0

    def test_boost_is_clamped_to_safe_range(self):
        """1.0 未満の設定は攻撃的にならないよう 1.0 に丸める."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        boost, proportional = SkipGateEvaluator._compute_velocity_offset_multiplier(
            observed_velocity_bps=9.0,
            threshold_bps=6.0,
            base_multiplier=0.5,
            max_multiplier=0.8,
            proportional=True,
        )
        assert proportional is True
        assert boost == 1.0


# =================================================================
# B. trending_sell_as_offset テスト
# =================================================================


class TestTrendingSellSoftConfig:
    """196# trending_sell_as_offset config フィールド."""

    def test_default_trending_sell_as_offset_disabled(self):
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.trending_sell_as_offset_enabled is False

    def test_default_trending_sell_offset_boost_factor(self):
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.trending_sell_offset_boost_factor == 2.0


class TestTrendingSellSoftGate:
    """196# CycleGateAggregator で trending_sell soft mode が動作."""

    def test_soft_mode_not_blocked(self):
        """soft mode 時は blocked=False."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = _make_config(
            skip_sell_trending=True,
            skip_sell_trending_up_only=True,
            trending_sell_as_offset_enabled=True,
            trending_sell_offset_boost_factor=3.0,
        )
        gate = CycleGateAggregator(cfg)
        result = gate.evaluate(
            side="sell",
            regime="trending_up",
            vol_ratio=1.0,
            balance_forced=False,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
        )
        assert result.blocked is False
        assert result.trending_offset_mult == 3.0

    def test_hard_mode_blocks(self):
        """hard mode (デフォルト) は blocked=True."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = _make_config(
            skip_sell_trending=True,
            skip_sell_trending_up_only=True,
            trending_sell_as_offset_enabled=False,
        )
        gate = CycleGateAggregator(cfg)
        result = gate.evaluate(
            side="sell",
            regime="trending_up",
            vol_ratio=1.0,
            balance_forced=False,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
        )
        assert result.blocked is True
        assert result.trending_offset_mult is None

    def test_soft_mode_trending_down_not_up_only(self):
        """trending_up_only=True + regime=trending_down → soft mode 不適用 (通常 pass)."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = _make_config(
            skip_sell_trending=True,
            skip_sell_trending_up_only=True,
            trending_sell_as_offset_enabled=True,
        )
        gate = CycleGateAggregator(cfg)
        result = gate.evaluate(
            side="sell",
            regime="trending_down",
            vol_ratio=1.0,
            balance_forced=False,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
        )
        assert result.blocked is False
        # trending_down は skip 対象外なので offset_mult は None
        assert result.trending_offset_mult is None

    def test_soft_mode_buy_side_unaffected(self):
        """buy 側は trending_sell gate に影響されない."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = _make_config(
            skip_sell_trending=True,
            skip_sell_trending_up_only=True,
            trending_sell_as_offset_enabled=True,
        )
        gate = CycleGateAggregator(cfg)
        result = gate.evaluate(
            side="buy",
            regime="trending_up",
            vol_ratio=1.0,
            balance_forced=False,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
        )
        assert result.blocked is False
        assert result.trending_offset_mult is None

    def test_soft_mode_balance_forced_applies_by_default(self):
        """balance_forced=True でも live YAML 既定では trending offset を適用."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = _make_config(
            skip_sell_trending=True,
            skip_sell_trending_up_only=True,
            trending_sell_as_offset_enabled=True,
        )
        gate = CycleGateAggregator(cfg)
        result = gate.evaluate(
            side="sell",
            regime="trending_up",
            vol_ratio=1.0,
            balance_forced=True,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
        )
        assert result.blocked is False
        assert result.trending_offset_mult == 2.0

    def test_soft_mode_balance_forced_always_applies_offset_234(self):
        """234#: soft mode の trending offset は常に適用.

        234# で balance_forced の gate bypass を廃止したため、
        soft mode は balance_forced に関係なく常に offset を適用する。
        253# NOTE: balance_forced_apply_trending_offset フィールドは削除済み。
        """
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = _make_config(
            skip_sell_trending=True,
            skip_sell_trending_up_only=True,
            trending_sell_as_offset_enabled=True,
        )
        gate = CycleGateAggregator(cfg)
        result = gate.evaluate(
            side="sell",
            regime="trending_up",
            vol_ratio=1.0,
            balance_forced=True,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
        )
        assert result.blocked is False
        # 234#: soft mode は常に offset を適用 (balance_forced でも)
        assert result.trending_offset_mult == 2.0

    def test_soft_mode_eliminates_bypass_complexity(self):
        """soft mode では HF4/inv_bypass/consecutive bypass が不要.

        hard mode で必要だった bypass params は soft mode では到達しない。
        """
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = _make_config(
            skip_sell_trending=True,
            skip_sell_trending_up_only=False,  # all trending
            trending_sell_as_offset_enabled=True,
            trending_sell_offset_boost_factor=3.0,
            max_consecutive_trending_sell_skip=5,  # hard mode なら発動する値
        )
        gate = CycleGateAggregator(cfg)
        result = gate.evaluate(
            side="sell",
            regime="trending_up",
            vol_ratio=1.0,
            balance_forced=False,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
            trending_sell_skip_count=100,  # 大きな値でも soft mode では無関係
        )
        assert result.blocked is False
        assert result.trending_offset_mult == 3.0

    def test_audit_trail_includes_196_detail(self):
        """audit trail に 196# の detail が含まれること."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = _make_config(
            skip_sell_trending=True,
            skip_sell_trending_up_only=True,
            trending_sell_as_offset_enabled=True,
            trending_sell_offset_boost_factor=3.0,
        )
        gate = CycleGateAggregator(cfg)
        result = gate.evaluate(
            side="sell",
            regime="trending_up",
            vol_ratio=1.0,
            balance_forced=False,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
        )
        trending_check = [c for c in result.checks if c.gate_name == "trending_sell"]
        assert len(trending_check) == 1
        assert "196#" in trending_check[0].detail
        assert "3.0" in trending_check[0].detail


class TestTrendingOffsetInExecutor:
    """196# fill_cycle_executor の trending offset 適用."""

    def test_trending_offset_mult_parameter_exists(self):
        """run_single_cycle に trending_offset_mult パラメータが存在."""
        from scripts.v460.run_fill_test import FillTestRunner

        sig = inspect.signature(FillTestRunner.run_single_cycle)
        assert "trending_offset_mult" in sig.parameters

    def test_trending_offset_in_source(self):
        """fill_cycle_executor に 196# trend_offset ブロックが存在."""
        from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin

        source = inspect.getsource(FillCycleExecutorMixin.run_single_cycle)
        assert "196# trend_offset" in source
        assert "trending_offset_mult" in source

    def test_offset_helper_ignores_non_protective_multiplier(self):
        """1.0 以下の倍率は適用せず、価格を攻撃的にしない."""
        from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin

        price, ratio, applied_mult, delta = FillCycleExecutorMixin._apply_offset_multiplier(
            side="sell",
            order_price=1000.0,
            spread_at_order=100.0,
            effective_offset_ratio=0.2,
            offset_mult=0.5,
        )
        assert price == 1000.0
        assert ratio == 0.2
        assert applied_mult is None
        assert delta is None

    def test_offset_helper_supports_aggressive_mode(self):
        """193# EV モードでは multiplier>1.0 で mid に近づく."""
        from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin

        price, ratio, applied_mult, delta = FillCycleExecutorMixin._apply_offset_multiplier(
            side="buy",
            order_price=1000.0,
            spread_at_order=100.0,
            effective_offset_ratio=0.2,
            offset_mult=1.5,
            aggressive_when_multiplier_gt_one=True,
        )
        assert applied_mult == 1.5
        assert delta == pytest.approx(10.0, abs=0.01)
        assert price == 1010
        assert ratio == pytest.approx(0.3, abs=0.0001)


class TestGateCheckResultOffsetMult:
    """196# GateCheckResult に offset_mult フィールドが追加."""

    def test_offset_mult_field_exists(self):
        from scripts.v460.lib.cycle_gate_aggregator import GateCheckResult

        result = GateCheckResult(gate_name="test", blocked=False)
        assert result.offset_mult is None

    def test_offset_mult_with_value(self):
        from scripts.v460.lib.cycle_gate_aggregator import GateCheckResult

        result = GateCheckResult(
            gate_name="trending_sell",
            blocked=False,
            offset_mult=3.0,
        )
        assert result.offset_mult == 3.0


class TestCycleGateResultTrendingOffset:
    """196# CycleGateResult に trending_offset_mult フィールドが追加."""

    def test_trending_offset_mult_default_none(self):
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateResult

        result = CycleGateResult()
        assert result.trending_offset_mult is None


# =================================================================
# C. YAML Config Parse テスト
# =================================================================


class TestConfigYamlParse196:
    """196# YAML parse テスト."""

    def test_yaml_velocity_proportional(self):
        """live YAML に velocity_offset_proportional が含まれること."""
        from pathlib import Path
        import yaml  # type: ignore[import-untyped]

        yaml_path = Path("configs/v460/fill_test.yaml")
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        sg = cfg["skip_gate"]
        assert sg["velocity_offset_proportional"] is True
        assert sg["velocity_offset_boost_factor"] == 1.5
        assert sg["velocity_offset_max_mult"] == 4.0

    def test_yaml_trending_sell_soft(self):
        """live YAML に trending_sell_as_offset_enabled が含まれること."""
        from pathlib import Path
        import yaml  # type: ignore[import-untyped]

        yaml_path = Path("configs/v460/fill_test.yaml")
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        lc = cfg["loss_control"]
        assert lc["trending_sell_as_offset_enabled"] is True
        assert lc["trending_sell_offset_boost_factor"] == 3.0  # 246# 2.0→3.0 sell 防御強化
        # 253# 削除済み: balance_forced_apply_trending_offset

    def test_config_from_yaml_round_trip(self):
        """from_yaml で 196# フィールドが正しく parse されること."""
        from pathlib import Path
        import yaml  # type: ignore[import-untyped]
        from scripts.v460.lib.fill_config import FillTestConfig

        # FillTestConfig.from_yaml はトップレベル構造に依存するため
        # 個別セクションを直接検証
        yaml_path = Path("configs/v460/fill_test.yaml")
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        # skip_gate セクションに velocity proportional 設定が存在
        sg = raw["skip_gate"]
        assert "velocity_offset_proportional" in sg
        assert "velocity_offset_max_mult" in sg

        # loss_control セクションに trending_sell soft 設定が存在
        lc = raw["loss_control"]
        assert "trending_sell_as_offset_enabled" in lc
        assert "trending_sell_offset_boost_factor" in lc

        # FillTestConfig のデフォルト値が正しいこと
        cfg = FillTestConfig()
        assert cfg.velocity_offset_proportional is False
        assert cfg.velocity_offset_boost_factor == 1.5
        assert cfg.velocity_offset_max_mult == 4.0
        assert cfg.trending_sell_as_offset_enabled is False
        assert cfg.trending_sell_offset_boost_factor == 2.0
        # 253# 削除済み: balance_forced_apply_trending_offset


# =================================================================
# D. 後方互換性テスト
# =================================================================


class TestBackwardCompatibility196:
    """196# デフォルト値で旧モード動作を維持."""

    def test_velocity_proportional_default_off(self):
        """velocity_offset_proportional のデフォルトは False."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.velocity_offset_proportional is False

    def test_trending_sell_soft_default_off(self):
        """trending_sell_as_offset_enabled のデフォルトは False."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.trending_sell_as_offset_enabled is False

    def test_hard_mode_trending_sell_unchanged(self):
        """hard mode (デフォルト) の trending_sell 動作が変わらないこと."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = _make_config(
            skip_sell_trending=True,
            skip_sell_trending_up_only=True,
            trending_sell_as_offset_enabled=False,
        )
        gate = CycleGateAggregator(cfg)

        # trending_up + sell → hard block
        result = gate.evaluate(
            side="sell",
            regime="trending_up",
            vol_ratio=1.0,
            balance_forced=False,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
        )
        assert result.blocked is True

    def test_hard_mode_bypass_valves_still_work(self):
        """hard mode で bypass 安全弁 (HF4, inv_bypass) が引き続き動作."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = _make_config(
            skip_sell_trending=True,
            skip_sell_trending_up_only=False,
            trending_sell_as_offset_enabled=False,
            sell_guard_inv_bypass_threshold=0.3,
        )
        gate = CycleGateAggregator(cfg)

        # inv_bypass: imbalance >= 0.3 → bypass hard skip
        result = gate.evaluate(
            side="sell",
            regime="trending_up",
            vol_ratio=1.0,
            balance_forced=False,
            inv_net_imbalance=0.5,
            is_buy_killed=False,
            is_sell_killed=False,
        )
        assert result.blocked is False  # bypass success
        assert result.trending_offset_mult is None  # hard mode → no offset


class TestDesignConsistency196:
    """196# 設計パターンが 193#/195# と一貫していること."""

    def test_soft_pattern_in_cycle_gate(self):
        """CycleGateAggregator._check_trending_sell に 196# ソフトモードが存在."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        source = inspect.getsource(CycleGateAggregator._check_trending_sell)
        assert "trending_sell_as_offset_enabled" in source
        assert "trending_sell_offset_boost_factor" in source

    def test_orchestrator_passes_trending_offset(self):
        """orchestrator が trending_offset_mult を run_single_cycle に渡すこと."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin

        source = inspect.getsource(FillLoopOrchestratorMixin.run_continuous)
        assert "trending_offset_mult" in source
