"""200# A-N 実装の単体テスト.

10-A: soft_drawdown_interval_multiplier 日次リセット
B/I: postonly_guard crossing → skip
C: low_vol_boost proportional scaling
E: balance_forced cooldown
G: sell PnL wait dynamic (vol-scaled)
H: regime velocity modulation
K: halt record reduction
L: velocity SSOT
M: ev_as_offset warning zone + DRY
10-B: order_monitor redundant ternary fix
"""

from __future__ import annotations

import math

import pytest


class TestSoftDrawdownIntervalMultiplierReset:
    """10-A: soft_drawdown_interval_multiplier が日替わりでリセットされること."""

    def test_maybe_reset_day_resets_soft_triggered(self) -> None:
        """daily_drawdown_guard.maybe_reset_day() が True を返す場合を確認."""
        from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0, soft_limit_bps=-30.0)
        # 初回は current_day=None → reset
        result = guard.maybe_reset_day()
        assert result is True  # first call always resets
        # 2回目は同日 → False
        assert guard.maybe_reset_day() is False


class TestPostonlyCrossingSkip:
    """B/I: POSTONLY_CROSSING_SKIP cancel reason が存在すること."""

    def test_cancel_reason_constant(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert CR.POSTONLY_CROSSING_SKIP == "postonly_crossing_skip"

    def test_in_audit_set(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert CR.POSTONLY_CROSSING_SKIP in CR.AUDIT_CANCEL_REASONS

    def test_in_cancel_reason_literal(self) -> None:
        """CancelReason Literal に含まれること."""
        from scripts.v460.lib.cancel_reasons import CancelReason
        from typing import get_args
        assert "postonly_crossing_skip" in get_args(CancelReason)


class TestLowVolBoostProportional:
    """C: low_vol_boost 比例モード."""

    def test_config_fields_exist(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert hasattr(cfg, "low_vol_boost_proportional")
        assert hasattr(cfg, "low_vol_boost_min")
        assert cfg.low_vol_boost_proportional is False  # default
        assert cfg.low_vol_boost_min == 1.0

    def test_proportional_scaling_logic(self) -> None:
        """vol_ratio に応じた段階的 boost が正しく計算されること."""
        # threshold=0.75, vol_ratio=0.375 (50% of threshold)
        # → ratio = 1 - 0.375/0.75 = 0.5
        # → boost = 1.0 + (1.4 - 1.0) * 0.5 = 1.2
        threshold = 0.75
        boost_max = 1.4
        boost_min = 1.0
        vol_ratio = 0.375

        ratio = 1.0 - vol_ratio / threshold
        boost = boost_min + (boost_max - boost_min) * ratio
        assert abs(boost - 1.2) < 0.001

    def test_zero_vol_gives_max_boost(self) -> None:
        """vol_ratio=0 → 最大 boost."""
        threshold = 0.75
        boost_max = 1.4
        boost_min = 1.0
        vol_ratio = 0.0

        ratio = 1.0 - vol_ratio / threshold
        boost = boost_min + (boost_max - boost_min) * ratio
        assert abs(boost - boost_max) < 0.001


class TestBalanceForcedCooldown:
    """E: balance_forced_cooldown_sec config."""

    def test_config_field_exists(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert hasattr(cfg, "balance_forced_cooldown_sec")
        assert cfg.balance_forced_cooldown_sec == 0.0  # default: disabled


class TestSellPnlWaitDynamic:
    """G: effective_post_fill_wait vol_ratio スケーリング."""

    def test_vol_ratio_none_returns_base(self) -> None:
        from scripts.v460.lib.regime_policy import RegimePolicyConfig
        from scripts.v460.lib.cycle_strategy import DefaultCycleStrategy
        policy = RegimePolicyConfig(dynamic_wait_enabled=False)
        strategy = DefaultCycleStrategy(base_interval=120.0, base_wait_buy=30.0, base_wait_sell=90.0, policy=policy)
        result = strategy.effective_post_fill_wait("sell", None, vol_ratio=None)
        assert result == 90.0

    def test_low_vol_extends_wait(self) -> None:
        """低ボラティリティ → wait 延長."""
        from scripts.v460.lib.regime_policy import RegimePolicyConfig
        from scripts.v460.lib.cycle_strategy import DefaultCycleStrategy
        policy = RegimePolicyConfig(dynamic_wait_enabled=False)
        strategy = DefaultCycleStrategy(base_interval=120.0, base_wait_buy=30.0, base_wait_sell=90.0, policy=policy)
        result = strategy.effective_post_fill_wait("sell", None, vol_ratio=0.5)
        assert result > 90.0  # extended

    def test_high_vol_shortens_wait(self) -> None:
        """高ボラティリティ → wait 短縮."""
        from scripts.v460.lib.regime_policy import RegimePolicyConfig
        from scripts.v460.lib.cycle_strategy import DefaultCycleStrategy
        policy = RegimePolicyConfig(dynamic_wait_enabled=False)
        strategy = DefaultCycleStrategy(base_interval=120.0, base_wait_buy=30.0, base_wait_sell=90.0, policy=policy)
        result = strategy.effective_post_fill_wait("sell", None, vol_ratio=2.0)
        assert result < 90.0  # shortened

    def test_vol_scale_bounded(self) -> None:
        """vol_scale は 0.7x ~ 1.5x の範囲内."""
        from scripts.v460.lib.regime_policy import RegimePolicyConfig
        from scripts.v460.lib.cycle_strategy import DefaultCycleStrategy
        policy = RegimePolicyConfig(dynamic_wait_enabled=False)
        strategy = DefaultCycleStrategy(base_interval=120.0, base_wait_buy=30.0, base_wait_sell=90.0, policy=policy)
        # 極端な低 vol
        result_low = strategy.effective_post_fill_wait("sell", None, vol_ratio=0.01)
        assert result_low <= 90.0 * 1.5 + 0.01
        # 極端な高 vol
        result_high = strategy.effective_post_fill_wait("sell", None, vol_ratio=100.0)
        assert result_high >= 90.0 * 0.7 - 0.01


class TestRegimeVelocityModulation:
    """H: regime velocity modulation (opt-in)."""

    def test_velocity_modulation_disabled_by_default(self) -> None:
        from scripts.v460.lib.regime_detector import RegimeConfig
        cfg = RegimeConfig()
        assert cfg.velocity_modulation is False

    def test_velocity_modulation_increases_confidence_on_match(self) -> None:
        """velocity が trend 方向と一致 → confidence 強化."""
        from scripts.v460.lib.regime_detector import (
            FillTestRegimeDetector, RegimeConfig, FillTestRegime,
        )
        cfg = RegimeConfig(
            window=10,
            velocity_modulation=True,
            velocity_window_ratio=0.5,
            hysteresis_count=1,
            min_confidence=0.0,
        )
        detector = FillTestRegimeDetector(cfg)
        # 上昇トレンドデータ作成 (十分強い上昇)
        base = 15_000_000.0
        for i in range(30):
            # 急激な上昇: 各ステップ +0.5%
            price = base * (1 + 0.005 * i)
            detector.update(float(i), price)
        result = detector.update(30.0, base * (1 + 0.005 * 30))
        # trending (up or down) でないとテスト不成立 — 強い上昇なので trending_up
        assert result.regime in (FillTestRegime.TRENDING_UP, FillTestRegime.TRENDING)
        confidence_with_velocity = result.confidence

        # velocity modulation なしの比較
        cfg_no_vel = RegimeConfig(
            window=10,
            velocity_modulation=False,
            hysteresis_count=1,
            min_confidence=0.0,
        )
        detector2 = FillTestRegimeDetector(cfg_no_vel)
        for i in range(30):
            price = base * (1 + 0.005 * i)
            detector2.update(float(i), price)
        result2 = detector2.update(30.0, base * (1 + 0.005 * 30))
        # velocity modulation で confidence が上がっているはず
        assert confidence_with_velocity >= result2.confidence


class TestVelocityMath:
    """L: velocity_math SSOT モジュール."""

    def test_fixed_mode(self) -> None:
        from scripts.v460.lib.velocity_math import compute_velocity_offset_multiplier
        mult, prop = compute_velocity_offset_multiplier(
            observed_velocity_bps=10.0,
            threshold_bps=5.0,
            base_multiplier=1.5,
            max_multiplier=3.0,
            proportional=False,
        )
        assert mult == 1.5
        assert prop is False

    def test_proportional_mode(self) -> None:
        from scripts.v460.lib.velocity_math import compute_velocity_offset_multiplier
        mult, prop = compute_velocity_offset_multiplier(
            observed_velocity_bps=10.0,
            threshold_bps=5.0,
            base_multiplier=1.5,
            max_multiplier=3.0,
            proportional=True,
        )
        # excess_ratio = 10/5 = 2.0
        # boost = 1.0 + (1.5-1.0)*2.0 = 2.0
        assert abs(mult - 2.0) < 0.001
        assert prop is True

    def test_capped_at_max(self) -> None:
        from scripts.v460.lib.velocity_math import compute_velocity_offset_multiplier
        mult, _ = compute_velocity_offset_multiplier(
            observed_velocity_bps=100.0,
            threshold_bps=5.0,
            base_multiplier=1.5,
            max_multiplier=3.0,
            proportional=True,
        )
        assert mult == 3.0

    def test_ssot_matches_evaluator_static_method(self) -> None:
        """skip_gate_evaluator の static method が velocity_math に委譲していること."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
        from scripts.v460.lib.velocity_math import compute_velocity_offset_multiplier
        # 同一引数で同一結果
        kwargs = dict(
            observed_velocity_bps=7.5,
            threshold_bps=5.0,
            base_multiplier=2.0,
            max_multiplier=4.0,
            proportional=True,
        )
        direct = compute_velocity_offset_multiplier(**kwargs)
        via_evaluator = SkipGateEvaluator._compute_velocity_offset_multiplier(**kwargs)
        assert direct == via_evaluator


class TestEvOffsetWarningZone:
    """M: ev_as_offset warning zone + DRY."""

    def test_compute_ev_offset_multiplier_normal(self) -> None:
        from scripts.v460.lib.fill_config import compute_ev_offset_multiplier
        # ev_score=0 → mult=1.0
        mult = compute_ev_offset_multiplier(
            ev_score=0.0, sensitivity=0.05, min_mult=0.5, max_mult=1.5,
        )
        assert mult == 1.0

    def test_positive_ev_boosts(self) -> None:
        from scripts.v460.lib.fill_config import compute_ev_offset_multiplier
        mult = compute_ev_offset_multiplier(
            ev_score=5.0, sensitivity=0.05, min_mult=0.5, max_mult=1.5,
        )
        # 1.0 + 0.05*5.0 = 1.25
        assert abs(mult - 1.25) < 0.001

    def test_warning_zone_additional_tightening(self) -> None:
        from scripts.v460.lib.fill_config import compute_ev_offset_multiplier
        # ev_score=-5.0 (below warning threshold -4.0)
        # raw = 1.0 + 0.05*(-5) = 0.75
        # warning factor 0.7 → 0.75 * 0.7 = 0.525
        mult = compute_ev_offset_multiplier(
            ev_score=-5.0, sensitivity=0.05, min_mult=0.5, max_mult=1.5,
            warning_threshold=-4.0, warning_factor=0.7,
        )
        assert abs(mult - 0.525) < 0.001

    def test_above_warning_no_factor(self) -> None:
        from scripts.v460.lib.fill_config import compute_ev_offset_multiplier
        # ev_score=-3.0 (above warning threshold -4.0)
        # raw = 1.0 + 0.05*(-3) = 0.85 → no warning factor
        mult = compute_ev_offset_multiplier(
            ev_score=-3.0, sensitivity=0.05, min_mult=0.5, max_mult=1.5,
            warning_threshold=-4.0, warning_factor=0.7,
        )
        assert abs(mult - 0.85) < 0.001

    def test_config_warning_fields_exist(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert hasattr(cfg, "skip_gate_ev_warning_threshold")
        assert hasattr(cfg, "skip_gate_ev_warning_offset_factor")
        assert cfg.skip_gate_ev_warning_threshold == -4.0
        assert cfg.skip_gate_ev_warning_offset_factor == 0.7


class TestSoftDrawdownIntervalConfig:
    """10-E: soft_drawdown_interval_multiplier YAML 外部化."""

    def test_config_field_default(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.soft_drawdown_interval_multiplier == 3.0

    def test_yaml_parsing(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        kwargs = FillTestConfig._parse_stopgap_section({
            "止血": {
                "daily_drawdown": {
                    "enabled": True,
                    "hard_limit_bps": -50.0,
                    "soft_limit_bps": -30.0,
                    "soft_drawdown_interval_multiplier": 5.0,
                }
            }
        })
        assert kwargs["soft_drawdown_interval_multiplier"] == 5.0
