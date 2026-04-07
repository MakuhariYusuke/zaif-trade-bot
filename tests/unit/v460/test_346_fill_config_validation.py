"""346# fill_config_validation.py テスト.

329# で分離された FillTestConfig バリデーション関数の境界テスト。
各バリデーションルールの正常系 (デフォルト通過) と異常系 (ValueError) を検証。
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import cast

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_validation import validate_fill_config


@pytest.fixture()  # type: ignore[untyped-decorator]
def cfg() -> FillTestConfig:
    """デフォルト値で構築した FillTestConfig (バリデーション済み)."""
    return FillTestConfig()


# ============================================================
# 正常系: デフォルト値はバリデーションを通過する
# ============================================================

class TestDefaultConfigPasses:
    """デフォルト FillTestConfig は全バリデーションを通過すること."""

    def test_default_passes(self, cfg: FillTestConfig) -> None:
        validate_fill_config(cfg)  # 例外なし


# ============================================================
# 境界バリデーション: 異常値 → ValueError
# ============================================================

class TestBalanceShrinkDivisor:
    def test_below_min(self, cfg: FillTestConfig) -> None:
        cfg.balance_shrink_divisor = 0
        with pytest.raises(ValueError, match="balance_shrink_divisor"):
            validate_fill_config(cfg)


class TestOffsetRatios:
    def test_max_not_gt_min(self, cfg: FillTestConfig) -> None:
        cfg.max_offset_ratio = cfg.min_offset_ratio
        with pytest.raises(ValueError, match="max_offset_ratio"):
            validate_fill_config(cfg)

    def test_max_less_than_min(self, cfg: FillTestConfig) -> None:
        cfg.max_offset_ratio = cfg.min_offset_ratio - 0.01
        with pytest.raises(ValueError, match="max_offset_ratio"):
            validate_fill_config(cfg)


class TestPreflightParams:
    def test_pause_threshold_zero(self, cfg: FillTestConfig) -> None:
        cfg.preflight_pause_threshold = 0
        with pytest.raises(ValueError, match="preflight_pause_threshold"):
            validate_fill_config(cfg)

    def test_max_pauses_negative(self, cfg: FillTestConfig) -> None:
        cfg.preflight_max_pauses = -1
        with pytest.raises(ValueError, match="preflight_max_pauses"):
            validate_fill_config(cfg)

    def test_pause_sec_negative(self, cfg: FillTestConfig) -> None:
        cfg.preflight_pause_sec = -1
        with pytest.raises(ValueError, match="preflight_pause_sec"):
            validate_fill_config(cfg)


class TestSkipGateCalibrator:
    def test_min_samples_zero(self, cfg: FillTestConfig) -> None:
        cfg.skip_gate_calibrator_min_samples = 0
        with pytest.raises(ValueError, match="skip_gate_calibrator_min_samples"):
            validate_fill_config(cfg)

    def test_refit_interval_zero(self, cfg: FillTestConfig) -> None:
        cfg.skip_gate_calibrator_refit_interval = 0
        with pytest.raises(ValueError, match="skip_gate_calibrator_refit_interval"):
            validate_fill_config(cfg)


class TestRegimeDicts:
    def test_timeout_mult_zero(self, cfg: FillTestConfig) -> None:
        cfg.regime_timeout_multipliers = {"trending": 0}
        with pytest.raises(ValueError, match="regime_timeout_multipliers"):
            validate_fill_config(cfg)

    def test_lot_mult_negative(self, cfg: FillTestConfig) -> None:
        cfg.regime_lot_multipliers = {"ranging": -1}
        with pytest.raises(ValueError, match="regime_lot_multipliers"):
            validate_fill_config(cfg)

    def test_reprice_adj_too_large(self, cfg: FillTestConfig) -> None:
        cfg.regime_reprice_adjustments = {"trending": 11}
        with pytest.raises(ValueError, match="regime_reprice_adjustments"):
            validate_fill_config(cfg)

    def test_reprice_adj_boundary_pass(self, cfg: FillTestConfig) -> None:
        cfg.regime_reprice_adjustments = {"trending": 10}
        validate_fill_config(cfg)  # 境界値は OK

    def test_timeout_override_side_map_must_be_dict(self, cfg: FillTestConfig) -> None:
        cfg.regime_timeout_overrides = cast(
            dict[str, dict[str, float]],
            {"strong_up": 20.0},
        )
        with pytest.raises(ValueError, match="regime_timeout_overrides\\['strong_up'\\]"):
            validate_fill_config(cfg)

    def test_timeout_override_side_name_is_validated(self, cfg: FillTestConfig) -> None:
        cfg.regime_timeout_overrides = {"strong_up": {"both": 20.0}}
        with pytest.raises(ValueError, match="contains unsupported side 'both'"):
            validate_fill_config(cfg)

    def test_timeout_override_value_must_be_positive(self, cfg: FillTestConfig) -> None:
        cfg.regime_timeout_overrides = {"strong_up": {"sell": 0.0}}
        with pytest.raises(ValueError, match="must be > 0"):
            validate_fill_config(cfg)

    def test_skip_budget_window_must_be_positive(self, cfg: FillTestConfig) -> None:
        cfg.skip_gate_budget_window_min = 0
        with pytest.raises(ValueError, match="skip_gate_budget_window_min"):
            validate_fill_config(cfg)

    def test_skip_budget_limit_must_be_positive(self, cfg: FillTestConfig) -> None:
        cfg.skip_gate_budget_limits = {"default": 0}
        with pytest.raises(ValueError, match="skip_gate_budget_limits\\['default'\\]"):
            validate_fill_config(cfg)

    def test_skip_budget_side_name_is_validated(self, cfg: FillTestConfig) -> None:
        cfg.skip_gate_budget_limits = {"trending_up": {"both": 3}}
        with pytest.raises(ValueError, match="contains unsupported side 'both'"):
            validate_fill_config(cfg)

    def test_entry_gate_max_consecutive_blocks_must_be_positive(self, cfg: FillTestConfig) -> None:
        cfg.entry_gate_max_consecutive_blocks = 0
        with pytest.raises(ValueError, match="entry_gate_max_consecutive_blocks"):
            validate_fill_config(cfg)

    def test_entry_gate_max_block_rate_must_be_in_unit_interval(self, cfg: FillTestConfig) -> None:
        cfg.entry_gate_max_block_rate = 1.2
        with pytest.raises(ValueError, match="entry_gate_max_block_rate"):
            validate_fill_config(cfg)

    def test_entry_gate_min_eval_for_rate_has_lower_bound(self, cfg: FillTestConfig) -> None:
        cfg.entry_gate_min_eval_for_rate = 4
        with pytest.raises(ValueError, match="entry_gate_min_eval_for_rate"):
            validate_fill_config(cfg)

    def test_entry_gate_staleness_threshold_has_lower_bound(self, cfg: FillTestConfig) -> None:
        cfg.entry_gate_staleness_threshold_sec = 30.0
        with pytest.raises(ValueError, match="entry_gate_staleness_threshold_sec"):
            validate_fill_config(cfg)

    def test_entry_gate_buy_suppress_threshold_has_upper_bound(self, cfg: FillTestConfig) -> None:
        cfg.entry_gate_buy_suppress_ev_threshold = 0.1
        with pytest.raises(ValueError, match="entry_gate_buy_suppress_ev_threshold"):
            validate_fill_config(cfg)

    def test_entry_gate_buy_suppress_threshold_has_lower_bound(self, cfg: FillTestConfig) -> None:
        cfg.entry_gate_buy_suppress_ev_threshold = -5.1
        with pytest.raises(ValueError, match="entry_gate_buy_suppress_ev_threshold"):
            validate_fill_config(cfg)

    def test_spread_as_guard_threshold_must_be_positive(self, cfg: FillTestConfig) -> None:
        cfg.spread_as_guard_spread_threshold_bps = 0.0
        with pytest.raises(ValueError, match="spread_as_guard_spread_threshold_bps"):
            validate_fill_config(cfg)

    def test_spread_as_guard_threshold_has_upper_bound(self, cfg: FillTestConfig) -> None:
        cfg.spread_as_guard_spread_threshold_bps = 1500.0
        with pytest.raises(ValueError, match="spread_as_guard_spread_threshold_bps"):
            validate_fill_config(cfg)

    def test_regime_guard_multiplier_must_be_positive(self, cfg: FillTestConfig) -> None:
        cfg.regime_guard_spread_as_penalty_multipliers = {"ranging": 0.0}
        with pytest.raises(ValueError, match="regime_guard_spread_as_penalty_multipliers"):
            validate_fill_config(cfg)

    def test_spread_as_guard_redesign_cap_must_exceed_floor(self, cfg: FillTestConfig) -> None:
        cfg.spread_as_guard_inverse_penalty_floor_bps = 0.5
        cfg.spread_as_guard_inverse_penalty_cap_bps = 0.25
        with pytest.raises(ValueError, match="spread_as_guard_inverse_penalty_cap_bps"):
            validate_fill_config(cfg)

    def test_sell_ranging_offset_has_upper_bound(self, cfg: FillTestConfig) -> None:
        cfg.skip_gate_sell_ranging_offset = 2.5
        with pytest.raises(ValueError, match="skip_gate_sell_ranging_offset"):
            validate_fill_config(cfg)

    def test_sell_trending_up_offset_has_upper_bound(self, cfg: FillTestConfig) -> None:
        cfg.skip_gate_sell_trending_up_offset = 2.5
        with pytest.raises(ValueError, match="skip_gate_sell_trending_up_offset"):
            validate_fill_config(cfg)

    def test_sell_trending_down_offset_has_upper_bound(self, cfg: FillTestConfig) -> None:
        cfg.skip_gate_sell_trending_down_offset = 2.5
        with pytest.raises(ValueError, match="skip_gate_sell_trending_down_offset"):
            validate_fill_config(cfg)


class TestConfidenceLot:
    def test_floor_above_one(self, cfg: FillTestConfig) -> None:
        cfg.confidence_lot_floor = 1.01
        with pytest.raises(ValueError, match="confidence_lot_floor"):
            validate_fill_config(cfg)

    def test_floor_negative(self, cfg: FillTestConfig) -> None:
        cfg.confidence_lot_floor = -0.01
        with pytest.raises(ValueError, match="confidence_lot_floor"):
            validate_fill_config(cfg)

    def test_scale_negative(self, cfg: FillTestConfig) -> None:
        cfg.confidence_lot_scale = -1
        with pytest.raises(ValueError, match="confidence_lot_scale"):
            validate_fill_config(cfg)

    def test_invalid_mode(self, cfg: FillTestConfig) -> None:
        cfg.confidence_lot_mode = "invalid"
        with pytest.raises(ValueError, match="confidence_lot_mode"):
            validate_fill_config(cfg)

    def test_enabled_with_pnl_mode(self, cfg: FillTestConfig) -> None:
        cfg.enable_confidence_lot = True
        cfg.confidence_lot_mode = "pnl"
        with pytest.raises(ValueError, match="confidence_lot_mode must be 'as'"):
            validate_fill_config(cfg)


class TestSellGuardInvBypass:
    def test_above_one(self, cfg: FillTestConfig) -> None:
        cfg.sell_guard_inv_bypass_threshold = 1.01
        with pytest.raises(ValueError, match="sell_guard_inv_bypass_threshold"):
            validate_fill_config(cfg)

    def test_negative(self, cfg: FillTestConfig) -> None:
        cfg.sell_guard_inv_bypass_threshold = -0.01
        with pytest.raises(ValueError, match="sell_guard_inv_bypass_threshold"):
            validate_fill_config(cfg)


class TestDailyDrawdown:
    def test_soft_less_than_hard(self, cfg: FillTestConfig) -> None:
        cfg.daily_drawdown_soft_limit_bps = -60
        cfg.daily_drawdown_hard_limit_bps = -50
        with pytest.raises(ValueError, match="daily_drawdown_soft_limit_bps"):
            validate_fill_config(cfg)


class TestDynamicKillWindow:
    def test_sell_zero(self, cfg: FillTestConfig) -> None:
        cfg.sell_dynamic_kill_window = 0
        with pytest.raises(ValueError, match="sell_dynamic_kill_window"):
            validate_fill_config(cfg)

    def test_buy_zero(self, cfg: FillTestConfig) -> None:
        cfg.buy_dynamic_kill_window = 0
        with pytest.raises(ValueError, match="buy_dynamic_kill_window"):
            validate_fill_config(cfg)


class TestInvDecayTau:
    def test_negative(self, cfg: FillTestConfig) -> None:
        cfg.inv_decay_tau_sec = -1
        with pytest.raises(ValueError, match="inv_decay_tau_sec"):
            validate_fill_config(cfg)


class TestSellOffsetFloorInvDiscount:
    def test_above_one(self, cfg: FillTestConfig) -> None:
        cfg.sell_offset_floor_inv_discount = 1.01
        with pytest.raises(ValueError, match="sell_offset_floor_inv_discount"):
            validate_fill_config(cfg)


class TestTimingParams:
    def test_order_timeout_zero(self, cfg: FillTestConfig) -> None:
        cfg.order_timeout_sec = 0
        with pytest.raises(ValueError, match="order_timeout_sec must be > 0"):
            validate_fill_config(cfg)

    def test_max_cycle_sleep_negative(self, cfg: FillTestConfig) -> None:
        cfg.max_cycle_sleep_sec = -1
        with pytest.raises(ValueError, match="max_cycle_sleep_sec"):
            validate_fill_config(cfg)


class TestLossBoostDecayTau:
    def test_zero(self, cfg: FillTestConfig) -> None:
        cfg.loss_boost_decay_tau_sec = 0
        with pytest.raises(ValueError, match="loss_boost_decay_tau_sec"):
            validate_fill_config(cfg)


class TestLossCapRatio:
    def test_zero(self, cfg: FillTestConfig) -> None:
        cfg.loss_cap_ratio = 0
        with pytest.raises(ValueError, match="loss_cap_ratio"):
            validate_fill_config(cfg)

    def test_soft_negative(self, cfg: FillTestConfig) -> None:
        cfg.soft_loss_cap_ratio = -1
        with pytest.raises(ValueError, match="soft_loss_cap_ratio"):
            validate_fill_config(cfg)


class TestVelocityEmaAlpha:
    def test_zero(self, cfg: FillTestConfig) -> None:
        cfg.velocity_ema_alpha = 0.0
        with pytest.raises(ValueError, match="velocity_ema_alpha"):
            validate_fill_config(cfg)

    def test_above_one(self, cfg: FillTestConfig) -> None:
        cfg.velocity_ema_alpha = 1.01
        with pytest.raises(ValueError, match="velocity_ema_alpha"):
            validate_fill_config(cfg)

    def test_boundary_one_passes(self, cfg: FillTestConfig) -> None:
        cfg.velocity_ema_alpha = 1.0
        validate_fill_config(cfg)


class TestRangingObi:
    def test_factor_above_one(self, cfg: FillTestConfig) -> None:
        cfg.ranging_obi_asymmetry_factor = 1.01
        with pytest.raises(ValueError, match="ranging_obi_asymmetry_factor"):
            validate_fill_config(cfg)

    def test_threshold_negative(self, cfg: FillTestConfig) -> None:
        cfg.ranging_obi_threshold = -1
        with pytest.raises(ValueError, match="ranging_obi_threshold"):
            validate_fill_config(cfg)

    def test_mode_invalid(self, cfg: FillTestConfig) -> None:
        cfg.ranging_obi_mode = "bad"
        with pytest.raises(ValueError, match="ranging_obi_mode"):
            validate_fill_config(cfg)


class TestFFDParams:
    def test_deadzone_above_100(self, cfg: FillTestConfig) -> None:
        cfg.ffd_l2_deadzone_bps = 101
        with pytest.raises(ValueError, match="ffd_l2_deadzone_bps"):
            validate_fill_config(cfg)

    def test_boost_release_streak_zero(self, cfg: FillTestConfig) -> None:
        cfg.ffd_boost_release_streak = 0
        with pytest.raises(ValueError, match="ffd_boost_release_streak"):
            validate_fill_config(cfg)

    def test_boost_release_streak_above_20(self, cfg: FillTestConfig) -> None:
        cfg.ffd_boost_release_streak = 21
        with pytest.raises(ValueError, match="ffd_boost_release_streak"):
            validate_fill_config(cfg)


class TestDegradedLiquidation:
    def test_lot_mult_too_low(self, cfg: FillTestConfig) -> None:
        cfg.degraded_liquidation_lot_mult = 0.009
        with pytest.raises(ValueError, match="degraded_liquidation_lot_mult"):
            validate_fill_config(cfg)

    def test_offset_mult_below_one(self, cfg: FillTestConfig) -> None:
        cfg.degraded_liquidation_offset_mult = 0.99
        with pytest.raises(ValueError, match="degraded_liquidation_offset_mult"):
            validate_fill_config(cfg)

    def test_duty_cycle_one(self, cfg: FillTestConfig) -> None:
        cfg.degraded_liquidation_duty_cycle = 1
        with pytest.raises(ValueError, match="degraded_liquidation_duty_cycle"):
            validate_fill_config(cfg)


class TestDDCooldownRelease:
    def test_lot_scale_too_low(self, cfg: FillTestConfig) -> None:
        cfg.dd_cooldown_release_lot_scale = 0.009
        with pytest.raises(ValueError, match="dd_cooldown_release_lot_scale"):
            validate_fill_config(cfg)

    def test_sec_negative(self, cfg: FillTestConfig) -> None:
        cfg.dd_cooldown_release_sec = -1
        with pytest.raises(ValueError, match="dd_cooldown_release_sec"):
            validate_fill_config(cfg)

    def test_rearm_budget_positive(self, cfg: FillTestConfig) -> None:
        cfg.dd_cooldown_rearm_budget_bps = 0.01
        with pytest.raises(ValueError, match="dd_cooldown_rearm_budget_bps"):
            validate_fill_config(cfg)


class TestStructuralConsistency:
    def test_max_cycle_sleep_lt_halt_cap(self, cfg: FillTestConfig) -> None:
        cfg.max_cycle_sleep_sec = 1
        cfg.cycle_interval_sec = 10
        cfg.halt_sleep_multiplier = 2
        with pytest.raises(ValueError, match="max_cycle_sleep_sec.*must be >="):
            validate_fill_config(cfg)

    def test_order_timeout_gt_cycle_interval(self, cfg: FillTestConfig) -> None:
        cfg.order_timeout_sec = cfg.cycle_interval_sec + 1
        with pytest.raises(ValueError, match="order_timeout_sec.*must be <="):
            validate_fill_config(cfg)

    def test_lock_stale_heartbeat_too_short(self, cfg: FillTestConfig) -> None:
        cfg.lock_stale_heartbeat_sec = cfg.lock_heartbeat_period_sec * 2
        with pytest.raises(ValueError, match="lock_stale_heartbeat_sec.*must be >="):
            validate_fill_config(cfg)


class TestPerSideDDDeadlock:
    def test_halt_zero_no_ie_no_longer_raises(self, cfg: FillTestConfig) -> None:
        """522#/598#: IE は legacy read-only。halt_cycles=0 + IE無効でもエラーなし。"""
        cfg.per_side_dd_enabled = True
        cfg.per_side_dd_halt_cycles = 0
        cfg.inventory_escape_enabled = False
        validate_fill_config(cfg)  # 例外なし

    def test_halt_zero_with_ie_passes(self, cfg: FillTestConfig) -> None:
        cfg.per_side_dd_enabled = True
        cfg.per_side_dd_halt_cycles = 0
        cfg.inventory_escape_enabled = True
        validate_fill_config(cfg)


class TestKyleLambdaImbalanceDep:
    def test_kyle_without_imbalance_no_longer_warns(self, cfg: FillTestConfig) -> None:
        """665# 修正: depth cache は imbalance_enabled 無関係に更新されるため警告不要."""
        cfg.kyle_lambda_enabled = True
        cfg.imbalance_enabled = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_fill_config(cfg)
            assert not any("Kyle" in str(x.message) or "kyle" in str(x.message) for x in w)


class TestSigmaFloorVolRatio:
    def test_sigma_floor_negative(self, cfg: FillTestConfig) -> None:
        cfg.sigma_floor = -0.01
        with pytest.raises(ValueError, match="sigma_floor"):
            validate_fill_config(cfg)

    def test_sigma_floor_zero(self, cfg: FillTestConfig) -> None:
        """488# P0-3: sigma_floor=0 は AS 計算で σ²=0 除算を引き起こす."""
        cfg.sigma_floor = 0.0
        with pytest.raises(ValueError, match="sigma_floor"):
            validate_fill_config(cfg)

    def test_vol_ratio_floor_zero(self, cfg: FillTestConfig) -> None:
        cfg.vol_ratio_floor = 0
        with pytest.raises(ValueError, match="vol_ratio_floor"):
            validate_fill_config(cfg)


class TestMiscParams:
    def test_quiescence_sleep_negative(self, cfg: FillTestConfig) -> None:
        cfg.quiescence_sleep_sec = -1
        with pytest.raises(ValueError, match="quiescence_sleep_sec"):
            validate_fill_config(cfg)

    def test_halt_persist_interval_zero(self, cfg: FillTestConfig) -> None:
        cfg.halt_persist_interval = 0
        with pytest.raises(ValueError, match="halt_persist_interval"):
            validate_fill_config(cfg)

    def test_phantom_detection_mult_zero(self, cfg: FillTestConfig) -> None:
        cfg.phantom_detection_sleep_multiplier = 0
        with pytest.raises(ValueError, match="phantom_detection_sleep_multiplier"):
            validate_fill_config(cfg)

    def test_fallback_duration_zero(self, cfg: FillTestConfig) -> None:
        cfg.fallback_duration_sec = 0
        with pytest.raises(ValueError, match="fallback_duration_sec"):
            validate_fill_config(cfg)

    def test_unknown_regime_max_consecutive_zero(self, cfg: FillTestConfig) -> None:
        cfg.unknown_regime_max_consecutive = 0
        with pytest.raises(ValueError, match="unknown_regime_max_consecutive"):
            validate_fill_config(cfg)

    def test_inventory_skewing_window_negative(self, cfg: FillTestConfig) -> None:
        cfg.inventory_skewing_window = -1
        with pytest.raises(ValueError, match="inventory_skewing_window"):
            validate_fill_config(cfg)

    def test_loss_cooldown_interval_mult_below_one(self, cfg: FillTestConfig) -> None:
        cfg.loss_cooldown_interval_mult = 0.99
        with pytest.raises(ValueError, match="loss_cooldown_interval_mult"):
            validate_fill_config(cfg)

    def test_one_sided_interval_mult_zero(self, cfg: FillTestConfig) -> None:
        cfg.one_sided_consecutive_interval_mult = 0
        with pytest.raises(ValueError, match="one_sided_consecutive_interval_mult"):
            validate_fill_config(cfg)

    def test_one_sided_limit_negative(self, cfg: FillTestConfig) -> None:
        cfg.one_sided_consecutive_limit = -1
        with pytest.raises(ValueError, match="one_sided_consecutive_limit"):
            validate_fill_config(cfg)

    def test_low_vol_boost_min_below_one(self, cfg: FillTestConfig) -> None:
        cfg.low_vol_boost_min = 0.99
        with pytest.raises(ValueError, match="low_vol_boost_min"):
            validate_fill_config(cfg)

    def test_low_vol_boost_min_gt_boost(self, cfg: FillTestConfig) -> None:
        cfg.low_vol_boost_min = cfg.low_vol_offset_boost + 0.01
        with pytest.raises(ValueError, match="low_vol_boost_min.*must be <="):
            validate_fill_config(cfg)

    def test_soft_drawdown_interval_mult_zero(self, cfg: FillTestConfig) -> None:
        cfg.soft_drawdown_interval_multiplier = 0
        with pytest.raises(ValueError, match="soft_drawdown_interval_multiplier"):
            validate_fill_config(cfg)

    def test_stop_condition_check_interval_zero(self, cfg: FillTestConfig) -> None:
        cfg.stop_condition_check_interval = 0
        with pytest.raises(ValueError, match="stop_condition_check_interval"):
            validate_fill_config(cfg)

    def test_quiescence_gate_blocks_negative(self, cfg: FillTestConfig) -> None:
        cfg.quiescence_gate_blocks_threshold = -1
        with pytest.raises(ValueError, match="quiescence_gate_blocks_threshold"):
            validate_fill_config(cfg)


# ============================================================
# 491# P1-6: VPIN 連続ランプ閾値整合
# ============================================================

class TestVpinContinuousThreshold:
    """vpin_continuous_min < vpin_threshold の整合チェック."""

    def test_min_eq_threshold_raises(self, cfg: FillTestConfig) -> None:
        cfg.vg_vpin_continuous_enabled = True
        cfg.vg_vpin_continuous_min = 0.60
        cfg.volatility_guard_vpin_threshold = 0.60
        with pytest.raises(ValueError, match="vg_vpin_continuous_min"):
            validate_fill_config(cfg)

    def test_min_gt_threshold_raises(self, cfg: FillTestConfig) -> None:
        cfg.vg_vpin_continuous_enabled = True
        cfg.vg_vpin_continuous_min = 0.70
        cfg.volatility_guard_vpin_threshold = 0.60
        with pytest.raises(ValueError, match="vg_vpin_continuous_min"):
            validate_fill_config(cfg)

    def test_valid_range_passes(self, cfg: FillTestConfig) -> None:
        cfg.vg_vpin_continuous_enabled = True
        cfg.vg_vpin_continuous_min = 0.40
        cfg.volatility_guard_vpin_threshold = 0.80
        validate_fill_config(cfg)
