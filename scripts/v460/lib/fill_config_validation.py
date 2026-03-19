"""
329# fill_config_validation — FillTestConfig バリデーション.

328# God Object 分割 Step 2: fill_config.py から __post_init__ バリデーション
ロジックを分離。FillTestConfig の一貫性・値域チェックを一元管理する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig


def validate_fill_config(config: FillTestConfig) -> None:
    """103# バリデーション: YAML 誤設定による本番クラッシュ防止.

    Args:
        config: バリデーション対象の FillTestConfig インスタンス

    Raises:
        ValueError: 値域制約違反時
    """
    # 373# CRITICAL: order_quantity / min_order_btc ゼロ除算防止
    # balance_checker._check_sell() で int(max_base / min_order_btc) を行うため
    # 0 以下だと ZeroDivisionError → 注文不能 or 無限ロット計算が発生する。
    if config.order_quantity <= 0:
        raise ValueError(
            f"order_quantity must be > 0, got {config.order_quantity}"
        )
    if config.min_order_btc <= 0:
        raise ValueError(
            f"min_order_btc must be > 0, got {config.min_order_btc}"
        )
    if config.balance_shrink_divisor < 1:
        raise ValueError(
            f"balance_shrink_divisor must be >= 1, got {config.balance_shrink_divisor}"
        )
    if config.max_offset_ratio <= config.min_offset_ratio:
        raise ValueError(
            f"max_offset_ratio ({config.max_offset_ratio}) must be > "
            f"min_offset_ratio ({config.min_offset_ratio})"
        )
    # 139# §8-#6: 新規パラメータの境界バリデーション
    if config.preflight_pause_threshold < 1:
        raise ValueError(
            f"preflight_pause_threshold must be >= 1, got {config.preflight_pause_threshold}"
        )
    if config.preflight_max_pauses < 0:
        raise ValueError(
            f"preflight_max_pauses must be >= 0, got {config.preflight_max_pauses}"
        )
    if config.preflight_pause_sec < 0:
        raise ValueError(
            f"preflight_pause_sec must be >= 0, got {config.preflight_pause_sec}"
        )
    if config.skip_gate_calibrator_min_samples < 1:
        raise ValueError(
            f"skip_gate_calibrator_min_samples must be >= 1, got {config.skip_gate_calibrator_min_samples}"
        )
    if config.skip_gate_calibrator_refit_interval < 1:
        raise ValueError(
            f"skip_gate_calibrator_refit_interval must be >= 1, got {config.skip_gate_calibrator_refit_interval}"
        )
    # 145# §8-#6: レジーム設定の値域バリデーション
    for k, v in config.regime_timeout_multipliers.items():
        if v <= 0:
            raise ValueError(
                f"regime_timeout_multipliers['{k}'] must be > 0, got {v}"
            )
    for k, v in config.regime_lot_multipliers.items():
        if v <= 0:
            raise ValueError(
                f"regime_lot_multipliers['{k}'] must be > 0, got {v}"
            )
    _MAX_REPRICE_ADJ = 10
    for k, v in config.regime_reprice_adjustments.items():
        if abs(v) > _MAX_REPRICE_ADJ:
            raise ValueError(
                f"regime_reprice_adjustments['{k}'] abs value must be <= {_MAX_REPRICE_ADJ}, got {v}"
            )
    # 151# P3-03: confidence_lot バリデーション (§10 #2 対応)
    if not (0.0 <= config.confidence_lot_floor <= 1.0):
        raise ValueError(
            f"confidence_lot_floor must be in [0, 1], got {config.confidence_lot_floor}"
        )
    if config.confidence_lot_scale < 0:
        raise ValueError(
            f"confidence_lot_scale must be >= 0, got {config.confidence_lot_scale}"
        )
    if config.confidence_lot_mode not in ("as", "pnl"):
        raise ValueError(
            f"confidence_lot_mode must be 'as' or 'pnl', got '{config.confidence_lot_mode}'"
        )
    # §13 #1: enable=True + mode!=as は設定乖離 → fail-fast
    if config.enable_confidence_lot and config.confidence_lot_mode != "as":
        raise ValueError(
            f"confidence_lot_mode must be 'as' when enabled, "
            f"got '{config.confidence_lot_mode}' (mode='pnl' is frozen)"
        )
    # 173# sell_guard_inv_bypass_threshold バリデーション
    if not (0.0 <= config.sell_guard_inv_bypass_threshold <= 1.0):
        raise ValueError(
            f"sell_guard_inv_bypass_threshold must be in [0, 1], "
            f"got {config.sell_guard_inv_bypass_threshold}"
        )
    # 174# daily_drawdown soft/hard limit 順序バリデーション
    if config.daily_drawdown_soft_limit_bps < config.daily_drawdown_hard_limit_bps:
        raise ValueError(
            f"daily_drawdown_soft_limit_bps ({config.daily_drawdown_soft_limit_bps}) "
            f"must be >= daily_drawdown_hard_limit_bps ({config.daily_drawdown_hard_limit_bps}). "
            f"soft=-30, hard=-50 のように soft は hard より緩い値であること"
        )
    # 174# inventory_skewing_window / sell_dynamic_kill_window 境界
    if config.inventory_skewing_window < 0:
        raise ValueError(
            f"inventory_skewing_window must be >= 0, got {config.inventory_skewing_window}"
        )
    # 228# C2: inv_decay_tau_sec は非負
    if config.inv_decay_tau_sec < 0:
        raise ValueError(
            f"inv_decay_tau_sec must be >= 0, got {config.inv_decay_tau_sec}"
        )
    if config.sell_dynamic_kill_window < 1:
        raise ValueError(
            f"sell_dynamic_kill_window must be >= 1, got {config.sell_dynamic_kill_window}"
        )
    if config.buy_dynamic_kill_window < 1:
        raise ValueError(
            f"buy_dynamic_kill_window must be >= 1, got {config.buy_dynamic_kill_window}"
        )
    # 174# sell_offset_floor_inv_discount 値域
    if not (0.0 <= config.sell_offset_floor_inv_discount <= 1.0):
        raise ValueError(
            f"sell_offset_floor_inv_discount must be in [0, 1], "
            f"got {config.sell_offset_floor_inv_discount}"
        )
    # 201# review: 200# 新規フィールドのバリデーション
    if config.soft_drawdown_interval_multiplier <= 0:
        raise ValueError(
            f"soft_drawdown_interval_multiplier must be > 0, "
            f"got {config.soft_drawdown_interval_multiplier}"
        )
    if config.low_vol_boost_min < 1.0:
        raise ValueError(
            f"low_vol_boost_min must be >= 1.0, got {config.low_vol_boost_min}"
        )
    if config.low_vol_boost_min > config.low_vol_offset_boost:
        raise ValueError(
            f"low_vol_boost_min ({config.low_vol_boost_min}) must be <= "
            f"low_vol_offset_boost ({config.low_vol_offset_boost})"
        )
    # 348# balance_forced 撤廃: balance_forced_cooldown_sec バリデーションを削除
    # 202# A: loss_cooldown_interval_mult は 1.0 以上
    if config.loss_cooldown_interval_mult < 1.0:
        raise ValueError(
            f"loss_cooldown_interval_mult must be >= 1.0, "
            f"got {config.loss_cooldown_interval_mult}"
        )
    # 209# M-2: one-sided 制限パラメータのバリデーション
    if config.one_sided_consecutive_interval_mult <= 0:
        raise ValueError(
            f"one_sided_consecutive_interval_mult must be > 0, "
            f"got {config.one_sided_consecutive_interval_mult}"
        )
    if config.one_sided_consecutive_limit < 0:
        raise ValueError(
            f"one_sided_consecutive_limit must be >= 0, "
            f"got {config.one_sided_consecutive_limit}"
        )
    # 209# H5: コアタイミングパラメータのバリデーション
    for _timing_name in ("order_timeout_sec", "poll_interval_sec", "cycle_interval_sec"):
        if getattr(config, _timing_name) <= 0:
            raise ValueError(f"{_timing_name} must be > 0, got {getattr(config, _timing_name)}")
    if config.max_cycle_sleep_sec < 0:
        raise ValueError(
            f"max_cycle_sleep_sec must be >= 0, got {config.max_cycle_sleep_sec}"
        )
    # 243# quiescence バリデーション
    if config.quiescence_sleep_sec < 0:
        raise ValueError(
            f"quiescence_sleep_sec must be >= 0, got {config.quiescence_sleep_sec}"
        )
    if config.quiescence_gate_blocks_threshold < 0:
        raise ValueError(
            f"quiescence_gate_blocks_threshold must be >= 0, "
            f"got {config.quiescence_gate_blocks_threshold}"
        )
    # 227# M1: 追加バリデーション
    if config.loss_boost_decay_tau_sec <= 0:
        raise ValueError(
            f"loss_boost_decay_tau_sec must be > 0, got {config.loss_boost_decay_tau_sec}"
        )
    # 327# loss_cap_ratio ゼロ除算防止 — 被除数として使用されるため > 0 必須
    if config.loss_cap_ratio <= 0:
        raise ValueError(
            f"loss_cap_ratio must be > 0, got {config.loss_cap_ratio}"
        )
    if config.soft_loss_cap_ratio < 0:
        raise ValueError(
            f"soft_loss_cap_ratio must be >= 0, got {config.soft_loss_cap_ratio}"
        )
    if not (0.0 <= config.ranging_obi_asymmetry_factor <= 1.0):
        raise ValueError(
            f"ranging_obi_asymmetry_factor must be in [0, 1], "
            f"got {config.ranging_obi_asymmetry_factor}"
        )
    if config.ranging_obi_threshold < 0:
        raise ValueError(
            f"ranging_obi_threshold must be >= 0, got {config.ranging_obi_threshold}"
        )
    if not (0.0 < config.velocity_ema_alpha <= 1.0):
        raise ValueError(
            f"velocity_ema_alpha must be in (0, 1], got {config.velocity_ema_alpha}"
        )
    # 230# FFD 新規パラメータのバリデーション
    if not (0.0 <= config.ffd_l2_deadzone_bps <= 100.0):
        raise ValueError(
            f"ffd_l2_deadzone_bps must be in [0, 100], got {config.ffd_l2_deadzone_bps}"
        )
    if not (1 <= config.ffd_boost_release_streak <= 20):
        raise ValueError(
            f"ffd_boost_release_streak must be in [1, 20], got {config.ffd_boost_release_streak}"
        )
    # 249# 246# パラメータ境界バリデーション
    if not (0.01 <= config.degraded_liquidation_lot_mult <= 1.0):
        raise ValueError(
            f"degraded_liquidation_lot_mult must be in [0.01, 1.0], "
            f"got {config.degraded_liquidation_lot_mult}"
        )
    if config.degraded_liquidation_offset_mult < 1.0:
        raise ValueError(
            f"degraded_liquidation_offset_mult must be >= 1.0, "
            f"got {config.degraded_liquidation_offset_mult}"
        )
    if config.degraded_liquidation_duty_cycle < 2:
        raise ValueError(
            f"degraded_liquidation_duty_cycle must be >= 2, "
            f"got {config.degraded_liquidation_duty_cycle}"
        )
    if not (0.01 <= config.dd_cooldown_release_lot_scale <= 1.0):
        raise ValueError(
            f"dd_cooldown_release_lot_scale must be in [0.01, 1.0], "
            f"got {config.dd_cooldown_release_lot_scale}"
        )
    if config.dd_cooldown_release_sec < 0:
        raise ValueError(
            f"dd_cooldown_release_sec must be >= 0, "
            f"got {config.dd_cooldown_release_sec}"
        )
    if config.dd_cooldown_rearm_budget_bps > 0:
        raise ValueError(
            f"dd_cooldown_rearm_budget_bps must be <= 0, "
            f"got {config.dd_cooldown_rearm_budget_bps}"
        )
    # 277# 構造的整合性バリデーション — config 間の暗黙的依存を明示的に検証
    # max_cycle_sleep_sec は halt_sleep_multiplier × cycle_interval 以上であるべき
    _halt_cap = config.cycle_interval_sec * config.halt_sleep_multiplier
    if config.max_cycle_sleep_sec > 0 and config.max_cycle_sleep_sec < _halt_cap:
        raise ValueError(
            f"max_cycle_sleep_sec ({config.max_cycle_sleep_sec}) must be >= "
            f"cycle_interval_sec × halt_sleep_multiplier ({_halt_cap}). "
            f"halt sleep がキャップされると DD halt の回復待機が短縮される"
        )
    # order_timeout が cycle_interval 以上だと次サイクルが遅延
    if config.order_timeout_sec > config.cycle_interval_sec:
        raise ValueError(
            f"order_timeout_sec ({config.order_timeout_sec}) must be <= "
            f"cycle_interval_sec ({config.cycle_interval_sec}). "
            f"タイムアウトがサイクル間隔を超えると次サイクルの開始が遅延する"
        )
    # lock_stale_heartbeat は lock_heartbeat_period の 3 倍以上
    _min_stale = config.lock_heartbeat_period_sec * 3
    if config.lock_stale_heartbeat_sec < _min_stale:
        raise ValueError(
            f"lock_stale_heartbeat_sec ({config.lock_stale_heartbeat_sec}) must be >= "
            f"lock_heartbeat_period_sec × 3 ({_min_stale}). "
            f"stale 閾値が短すぎると正常 heartbeat でも stale 判定される"
        )
    # halt_persist_interval / stop_condition_check_interval 正値
    if config.halt_persist_interval < 1:
        raise ValueError(
            f"halt_persist_interval must be >= 1, got {config.halt_persist_interval}"
        )
    if config.stop_condition_check_interval < 1:
        raise ValueError(
            f"stop_condition_check_interval must be >= 1, "
            f"got {config.stop_condition_check_interval}"
        )
    if config.phantom_detection_sleep_multiplier <= 0:
        raise ValueError(
            f"phantom_detection_sleep_multiplier must be > 0, "
            f"got {config.phantom_detection_sleep_multiplier}"
        )
    if config.fallback_duration_sec <= 0:
        raise ValueError(
            f"fallback_duration_sec must be > 0, "
            f"got {config.fallback_duration_sec}"
        )
    if config.unknown_regime_max_consecutive < 1:
        raise ValueError(
            f"unknown_regime_max_consecutive must be >= 1, "
            f"got {config.unknown_regime_max_consecutive}"
        )
    # 285# 283# P0-2: per-side halt + IE 相互制約
    # 348# balance_forced 撤廃: デッドロックリスクは低減したが IE 必須バリデーションは安全策として保持
    if (
        config.per_side_dd_enabled
        and config.per_side_dd_halt_cycles == 0
        and not config.inventory_escape_enabled
    ):
        raise ValueError(
            "per_side_dd_halt_cycles=0 (永続封鎖) と "
            "inventory_escape_enabled=False の組合せは禁止。"
            "per-side halt の永久デッドロックが発生する "
            "(282# 実証済)。halt_cycles >= 1 にするか "
            "inventory_escape_enabled=True にしてください"
        )
    # 330# B4: kyle_lambda / amihud_illiq は compute_imbalance で depth が
    # 更新されるため、imbalance_enabled=False だと depth が永久 0 のまま
    # サイレント無効になる。設定ミスを早期検出。
    if (
        (config.kyle_lambda_enabled or config.amihud_illiq_enabled)
        and not config.imbalance_enabled
    ):
        import warnings
        warnings.warn(
            "kyle_lambda_enabled / amihud_illiq_enabled が True ですが "
            "imbalance_enabled=False のため depth キャッシュが更新されず、"
            "Kyle λ / Amihud ILLIQ が常にスキップされます。"
            "imbalance_enabled=True を推奨します。",
            stacklevel=3,
        )

    # 331# M-1: sigma_floor / vol_ratio_floor 値域チェック
    # 488# P0-3: sigma_floor=0 は AS 計算で σ²=0 除算を引き起こすため禁止
    if config.sigma_floor <= 0:
        raise ValueError(
            f"sigma_floor must be > 0, got {config.sigma_floor}"
        )
    if config.vol_ratio_floor <= 0:
        raise ValueError(
            f"vol_ratio_floor must be > 0, got {config.vol_ratio_floor}"
        )

    # 374# Phase 3.1: sidecar フィールドバリデーション
    # 375#/376# hard ceiling: max_boost_bps ≤ 0.20 (median spread 超過は自殺的)
    _SIDECAR_MAX_BOOST_CEILING: float = 0.20
    if config.sidecar_max_boost_bps < 0:
        raise ValueError(
            f"sidecar_max_boost_bps must be >= 0, "
            f"got {config.sidecar_max_boost_bps}"
        )
    if config.sidecar_max_boost_bps > _SIDECAR_MAX_BOOST_CEILING:
        raise ValueError(
            f"sidecar_max_boost_bps must be <= {_SIDECAR_MAX_BOOST_CEILING} "
            f"(375#/376# hard ceiling), got {config.sidecar_max_boost_bps}"
        )
    if not (0.0 <= config.sidecar_dead_zone < 1.0):
        raise ValueError(
            f"sidecar_dead_zone must be in [0.0, 1.0), "
            f"got {config.sidecar_dead_zone}"
        )
    from scripts.v460.lib.sidecar_types import _VALID_SHAPING

    if config.sidecar_shaping not in _VALID_SHAPING:
        raise ValueError(
            f"sidecar_shaping must be one of {sorted(_VALID_SHAPING)}, "
            f"got {config.sidecar_shaping!r}"
        )

    # ---- 452# Micro-timeout バリデーション ----
    if config.micro_timeout_wait_sec <= 0:
        raise ValueError(
            f"micro_timeout_wait_sec must be > 0, "
            f"got {config.micro_timeout_wait_sec}"
        )
    if (
        config.micro_timeout_wait_sec_sell is not None
        and config.micro_timeout_wait_sec_sell <= 0
    ):
        raise ValueError(
            f"micro_timeout_wait_sec_sell must be > 0 or None, "
            f"got {config.micro_timeout_wait_sec_sell}"
        )
    if config.micro_timeout_max_requote < 1:
        raise ValueError(
            f"micro_timeout_max_requote must be >= 1, "
            f"got {config.micro_timeout_max_requote}"
        )
    if config.micro_timeout_requote_cooloff_sec < 0:
        raise ValueError(
            f"micro_timeout_requote_cooloff_sec must be >= 0, "
            f"got {config.micro_timeout_requote_cooloff_sec}"
        )
    # 構造的整合性: サブサイクル合計が cycle_interval を超えないか警告
    if config.micro_timeout_enabled:
        _mt_total = config.micro_timeout_max_requote * (
            config.micro_timeout_wait_sec
            + config.micro_timeout_requote_cooloff_sec
        )
        if _mt_total > config.cycle_interval_sec:
            import warnings
            warnings.warn(
                f"micro_timeout 合計時間 ({_mt_total:.0f}s = "
                f"{config.micro_timeout_max_requote} × "
                f"({config.micro_timeout_wait_sec} + "
                f"{config.micro_timeout_requote_cooloff_sec})) が "
                f"cycle_interval_sec ({config.cycle_interval_sec}s) を超過。"
                f"サイクル遅延が発生します。",
                stacklevel=3,
            )
        # stale_order との二重制御警告
        if config.stale_order_enabled:
            import warnings
            warnings.warn(
                "micro_timeout_enabled と stale_order_enabled が同時有効。"
                "OrderMonitor 内の stale reprice と micro-timeout の "
                "二重価格制御が競合する可能性があります。",
                stacklevel=3,
            )

    # --- 491# P1-6: VPIN 連続ランプ閾値整合 ---
    if config.vg_vpin_continuous_enabled:
        if config.vg_vpin_continuous_min >= config.volatility_guard_vpin_threshold:
            raise ValueError(
                f"vg_vpin_continuous_min ({config.vg_vpin_continuous_min}) must be < "
                f"volatility_guard_vpin_threshold ({config.volatility_guard_vpin_threshold}). "
                f"連続ランプの下限が閾値以上では勾配が生成されません。"
            )
