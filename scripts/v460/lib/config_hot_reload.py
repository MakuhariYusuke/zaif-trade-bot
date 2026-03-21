"""169# Config Hot-Reload — YAML 変更をプロセス再起動なしで反映.

SkipGate モデル hot-reload (126#) と同パターン:
  - mtime ベースのファイル変更検知 (polling, configurable interval)
  - 安全なフィールドのみ差分更新 (構造体再構築が必要なものは対象外 or 明示的再構築)
  - 失敗時は旧設定を維持 (防御的設計)

Usage (FillLoopOrchestratorMixin 内)::

    self._config_reloader = ConfigHotReloader(config, yaml_path, yaml_cfg)
    # 各サイクル末尾:
    self._config_reloader.maybe_reload(self)
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig
    from scripts.v460.lib.maker_price import MakerPriceCalculator
    from scripts.v460.lib.time_filter import TimeFilter

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _fill_config_field_names() -> tuple[str, ...]:
    """FillTestConfig field names resolved once for repeated hot-reload checks."""
    from scripts.v460.lib.fill_config import FillTestConfig

    return tuple(f.name for f in dataclasses.fields(FillTestConfig))


class _HotReloadableRunner(Protocol):
    """ConfigHotReloader が runner に要求する最小インタフェース.

    FillTestRunner の circular import を避けつつ型安全を確保する
    構造的サブタイピング (PEP 544).
    """

    _time_filter: TimeFilter
    _maker_price: MakerPriceCalculator
    _fast_fill_defense: object  # 210# D: FFD sync に必要
    _git_sha: str

    def _rebuild_sell_kill_mgr(self) -> None: ...
    def _rebuild_buy_kill_mgr(self) -> None: ...
    def _rebuild_daily_drawdown_guard(self) -> None: ...
    def _rebuild_fast_fill_defense(self) -> None: ...
    def _rebuild_cycle_strategy(self) -> None: ...  # 179#


@lru_cache(maxsize=1)
def _resolve_time_filter_cls() -> type[TimeFilter]:
    """Resolve TimeFilter lazily and cache the class object."""
    from scripts.v460.lib.time_filter import TimeFilter as _TF
    return _TF  # type: ignore[no-any-return]  # lazy import


# ======================================================================
# 安全にホットリロード可能なフィールドの定義
# ======================================================================

# ランタイム中に直接代入で反映できるフィールド
# (構造体の再構築が不要、または再構築を明示的に行うもの)
_HOT_RELOADABLE_FIELDS: frozenset[str] = frozenset({
    # --- offset / price 関連 ---
    "spread_offset_ratio",
    "spread_offset_ratio_buy",
    "spread_offset_ratio_sell",
    "min_offset_jpy",
    "max_offset_ratio",
    "min_offset_ratio",
    # --- regime offset 関連 ---
    "regime_trending_offset_boost",
    "regime_trending_offset_boost_buy",
    "regime_trending_offset_boost_sell",
    # 176# B: 方向×サイド別 offset boost
    "trending_up_buy_offset_boost",
    "trending_up_sell_offset_boost",
    "trending_down_buy_offset_boost",
    "trending_down_sell_offset_boost",
    "regime_high_vol_offset_boost",
    "regime_ranging_offset_discount",
    "low_vol_offset_boost_enabled",
    "low_vol_offset_boost",
    "low_vol_threshold",
    "skip_ranging_buy_low_vol",
    # --- time_filter ---
    "enable_time_filter",
    "skip_utc_hours",
    "skip_utc_hours_buy",
    "skip_utc_hours_sell",
    "regime_adaptive_enabled",
    "regime_adaptive_extra_buy",
    "regime_adaptive_extra_sell",
    # --- SkipGate 閾値 (モデル自体は別途 hot-reload) ---
    "skip_gate_enabled",
    "skip_gate_buy_enabled",
    "skip_gate_sell_enabled",
    "skip_gate_as_threshold",
    "skip_gate_as_threshold_buy",
    "skip_gate_as_threshold_sell",
    "skip_gate_pnl_threshold",
    "skip_gate_max_skip_rate",
    "skip_gate_adaptive_threshold",
    "skip_gate_target_skip_rate_buy",
    "skip_gate_target_skip_rate_sell",
    "skip_gate_hour_offsets",
    # 183# narrow spread adverse guard
    "skip_gate_narrow_spread_threshold_jpy",
    "skip_gate_narrow_spread_offset",
    # 187# clamp YAML外部化
    "skip_gate_offset_floor",
    "skip_gate_offset_ceil",
    # 188# C-1: ev_weighted SkipGate
    "skip_gate_ev_weighted_enabled",
    "skip_gate_ev_w30",
    "skip_gate_ev_w120",
    # 190# A/B: ev_weighted 連続 skip 安全弁 + 片側 balance threshold 緩和
    "skip_gate_ev_max_consecutive_skip",
    "skip_gate_ev_one_sided_threshold_shift",
    # 193# EV soft offset
    "skip_gate_ev_as_offset_enabled",
    "skip_gate_ev_offset_sensitivity",
    "skip_gate_ev_offset_min_mult",
    "skip_gate_ev_offset_max_mult",
    "skip_gate_ev_emergency_skip_threshold",
    # 189# D: MacroRegime
    "enable_macro_regime",
    "macro_regime_conflict_action",
    # 458# macro sell/buy boost + timeout
    "macro_sell_boost_weak_up",
    "macro_sell_boost_strong_up",
    "macro_buy_boost_weak_down",
    "macro_buy_boost_strong_down",
    "macro_sell_timeout_weak_up",
    "macro_sell_timeout_strong_up",
    # --- dynamic kill ---
    "sell_dynamic_kill_enabled",
    "sell_dynamic_kill_window",
    "sell_dynamic_kill_threshold_bps",
    "sell_dynamic_kill_resume_window",
    "sell_dynamic_kill_regime_thresholds",
    "buy_dynamic_kill_enabled",
    "buy_dynamic_kill_window",
    "buy_dynamic_kill_threshold_bps",
    "buy_dynamic_kill_resume_window",
    "buy_dynamic_kill_regime_thresholds",
    # --- lot 関連 ---
    "order_quantity",
    "max_lot",
    "regime_lot_multipliers",
    "enable_confidence_lot",
    "confidence_lot_scale",
    "confidence_lot_floor",
    # --- stale / reprice ---
    "stale_order_enabled",
    "stale_check_after_sec",
    "stale_drift_bps",
    "stale_max_reprice",
    "stale_reprice_tighten",
    "stale_reprice_min_delta_jpy",     # 292# reprice deadband
    "stale_reprice_skip_gate_offset",
    # 175# stale side 別フィールド
    "stale_check_after_sec_buy",
    "stale_check_after_sec_sell",
    "stale_drift_bps_buy",
    "stale_drift_bps_sell",
    "stale_max_reprice_buy",
    "stale_max_reprice_sell",
    # --- narrow spread ---
    "narrow_spread_pause_enabled",
    "narrow_spread_pause_bps",
    "narrow_spread_pause_max_consecutive",
    # --- fast fill defense ---
    "fast_fill_defense_enabled",
    "fast_fill_threshold_sec",
    "fast_fill_threshold_sec_buy",     # 174# side 別
    "fast_fill_threshold_sec_sell",    # 174# side 別
    "fast_fill_offset_boost",
    "fast_fill_offset_boost_buy",      # 174# side 別
    "fast_fill_offset_boost_sell",     # 174# side 別
    # --- 日次ドローダウン ---
    "daily_drawdown_enabled",
    "daily_drawdown_hard_limit_bps",
    "daily_drawdown_soft_limit_bps",
    # --- safety ---
    "loss_cap_jpy",
    "loss_cap_ratio",
    "soft_loss_cap_ratio",
    # --- sell/buy guard ---
    "skip_sell_unknown_regime",
    "skip_buy_unknown_regime",
    "skip_sell_trending",
    "skip_sell_trending_up_only",
    # 252# Sell Asymmetric Mode: high_vol でも sell skip (Glosten-Milgrom)
    "sell_asymmetric_high_vol_enabled",
    # 196# trending sell soft offset
    "trending_sell_as_offset_enabled",
    "trending_sell_offset_boost_factor",
    # 253# 削除済み: balance_forced_apply_trending_offset (234# dead config)
    # 348# balance_forced 撤廃: skip_balance_forced, balance_forced_*, forced_buy_delay_* 削除
    "max_consecutive_trending_sell_skip",
    "sell_guard_inv_bypass_threshold",  # 171# Guard Paradox 対策
    # --- velocity skip ---
    "sell_velocity_skip_enabled",
    "sell_velocity_skip_threshold_bps",
    "buy_velocity_skip_enabled",
    "buy_velocity_skip_threshold_bps",
    # 195/196# velocity soft offset
    "velocity_skip_as_offset_enabled",
    "velocity_offset_boost_factor",
    "velocity_offset_proportional",
    "velocity_offset_max_mult",
    # --- VG ---
    "volatility_guard_enabled",
    "volatility_guard_velocity_threshold_bps",
    "volatility_guard_offset_boost_factor",
    # --- cycle timing ---
    "cycle_interval_sec",
    "order_timeout_sec",
    "order_timeout_sec_sell",
    "post_fill_wait_sec",       # 174# base (sell フォールバック先)
    "post_fill_wait_sec_sell",
    # --- misc ---
    "as_deadzone_bps",
    "min_spread_jpy",
    "e3_sampling_ratio",
    "progress_log_interval",
    # --- 173# maker_price 単純値読み取り ---
    "sell_offset_floor",
    "sell_offset_floor_inv_discount",
    "sell_max_spread_jpy",
    "unknown_buy_offset_boost",
    "fallback_stale_sec",
    # --- 215# P0-B: 防御パラメータ (202#-210# 追加分) ---
    # 202# A: 単一サイクル大損失クールダウン
    "loss_cooldown_threshold_bps",
    "loss_cooldown_interval_mult",
    # 207# §3: 大損後 offset 防御拡大
    "loss_boost_offset_mult",
    # 207# §1: Toxic Fill 同一サイド拒否
    "toxic_fill_veto_threshold_bps",
    "toxic_fill_veto_cycles",
    # 209# M-3: 片側連続実行制限
    "one_sided_consecutive_limit",
    "one_sided_consecutive_interval_mult",
    # 205# §9.5: 片側 DD ガード
    "per_side_dd_enabled",
    "per_side_dd_hard_limit_bps",
    "per_side_dd_halt_cycles",
    # 205# §9.4: 時間帯 Hard Skip
    "hard_skip_utc_hours",
    # 209# M-4: max cycle sleep cap
    "max_cycle_sleep_sec",
    # 212# §3.2: Soft DD interval multiplier
    "soft_drawdown_interval_multiplier",
    # ══════════════════════════════════════════════════════════════════
    # 295# 一括追加: 運用パラメータの包括的 Hot-Reload 対応
    # ══════════════════════════════════════════════════════════════════
    # --- AS (Avellaneda-Stoikov / delta-star) ---
    "as_reservation_enabled",
    "as_reservation_gamma",
    "as_reservation_tau_sec",
    "as_tau_dynamic_enabled",
    "as_tau_dynamic_min_sec",
    "as_tau_dynamic_max_sec",
    "as_delta_star_enabled",
    "as_delta_star_fill_rate_k",
    # as_deadzone_bps は上部 offset セクションで登録済み
    # --- Amihud illiquidity ---
    "amihud_illiq_enabled",
    "amihud_illiq_baseline",
    "amihud_illiq_max_mult",
    # --- Kyle lambda ---
    "kyle_lambda_enabled",
    "kyle_lambda_impact_mult",
    "kyle_lambda_max_add_ratio",
    # --- balance / inventory ---
    "balance_freeze_cycles",
    "balance_margin_ratio",
    "balance_shrink_consecutive",
    "balance_shrink_divisor",
    "one_sided_balance_rescue_offset",
    # --- inventory skewing ---
    "inventory_skewing_enabled",
    "inventory_skewing_window",
    "inventory_skewing_max_factor",
    "inventory_skewing_neutral_band",
    "inv_skew_regime_gate_enabled",
    "inv_decay_tau_sec",
    # --- degraded liquidation ---
    "degraded_liquidation_enabled",
    "degraded_liquidation_lot_mult",
    "degraded_liquidation_offset_mult",
    "degraded_liquidation_duty_cycle",
    # --- inventory escape (522# 撤廃済みだが後方互換で残置) ---
    # --- DD cooldown / recovery ---
    "dd_cooldown_release_sec",
    "dd_cooldown_release_lot_scale",
    "dd_cooldown_rearm_budget_bps",
    "dd_day_reset_utc_offset_hours",
    "per_side_dd_recovery_cycles",
    "per_side_dd_recovery_lot_scale",
    "per_side_dd_reanchor_budget_bps",
    "recovery_trending_penalty",
    "recovery_high_vol_penalty",
    # 303# B: DD soft lot side 分離
    "daily_drawdown_soft_lot_side_aware",
    # 303# C: none regime Passive MM
    "none_regime_passive_mm_enabled",
    "none_regime_fixed_offset_bps",
    # 305# Parkinson σ 推定器
    "sigma_parkinson_enabled",
    "sigma_parkinson_window_sec",
    # --- one-sided escalation ---
    "one_sided_escalation_cooldown_offset",
    "one_sided_escalation_cooldown_cycles",
    "one_sided_escalation_freeze_offset",
    "one_sided_escalation_freeze_cycles",
    # --- dual kill quiescence ---
    "dual_kill_quiescence_enabled",
    "quiescence_gate_blocks_threshold",
    "quiescence_sleep_sec",
    # --- MCB (Market Circuit Breaker) ---
    "mcb_enabled",
    "mcb_caution_sigma",
    "mcb_warning_sigma",
    "mcb_halt_sigma",
    "mcb_warning_interval_mult",
    "mcb_warning_offset_mult",
    "mcb_halt_cooldown_sec",
    # --- SAD (Spread Anomaly Detector) ---
    "sad_enabled",
    "sad_wide_ratio",
    "sad_dry_ratio",
    "sad_frozen_ratio",
    "sad_baseline_window_sec",
    # --- buy AS guard ---
    "buy_as_guard_enabled",
    "buy_as_guard_velocity_threshold_bps",
    "buy_as_guard_offset_mult",
    "buy_as_guard_max_offset_ratio",
    # --- SkipGate EV warning ---
    "skip_gate_ev_warning_threshold",
    "skip_gate_ev_warning_offset_factor",
    # --- SkipGate adaptive ---
    "skip_gate_adaptive_ceiling",
    "skip_gate_adaptive_floor",
    "skip_gate_adaptive_min_samples",
    "skip_gate_adaptive_step",
    "skip_gate_adaptive_window",
    "skip_gate_calibrator_min_samples",
    "skip_gate_calibrator_refit_interval",
    "skip_gate_regime_thresholds",
    "skip_gate_score_calibration",
    "skip_gate_ob_depth",
    "skip_gate_recent_trades_limit",
    "skip_gate_use_ob_features",
    # --- imbalance / OBI ---
    "imbalance_enabled",
    "imbalance_threshold",
    "imbalance_skip_threshold",
    "imbalance_offset_boost",
    "imbalance_depth",
    "ranging_obi_threshold",
    "ranging_obi_asymmetry_factor",
    # --- narrow spread boost ---
    "narrow_spread_boost",
    "narrow_spread_boost_buy",
    "narrow_spread_boost_sell",
    "narrow_spread_bps",
    "narrow_spread_pause_sec",
    # --- wide spread ---
    "wide_spread_bps",
    "wide_spread_ratio",
    # --- spread adaptive ---
    "spread_adaptive_enabled",
    # --- early exit ---
    "early_exit_enabled",
    "early_exit_threshold_bps",
    "early_exit_monitor_interval_sec",
    "early_exit_rapid_interval_sec",
    # --- E3 measurement ---
    "e3_60s_multiplier",
    "e3_120s_multiplier",
    # --- dynamic kill advanced ---
    "sell_dynamic_kill_max_duration_sec",
    "sell_dynamic_kill_max_force_probes",
    "sell_dynamic_kill_max_stale_cycles",
    "sell_dynamic_kill_toxic_stale_mult",
    "buy_dynamic_kill_max_duration_sec",
    "buy_dynamic_kill_max_force_probes",
    "buy_dynamic_kill_max_stale_cycles",
    "buy_dynamic_kill_toxic_stale_mult",
    "buy_dynamic_kill_inv_relaxation_enabled",
    "buy_dynamic_kill_inv_relaxation_scale",
    "buy_dynamic_kill_inv_relaxation_max_bps",
    # 344# 342#D: EWMA α + 342#B inv_relaxation max_bps
    "sell_dynamic_kill_ewma_alpha",
    "buy_dynamic_kill_ewma_alpha",
    # 353# EWMA 時間減衰
    "sell_dynamic_kill_ewma_time_decay_tau_sec",
    "buy_dynamic_kill_ewma_time_decay_tau_sec",
    "sell_dynamic_kill_inv_relaxation_enabled",
    "sell_dynamic_kill_inv_relaxation_scale",
    "sell_dynamic_kill_inv_relaxation_max_bps",
    # --- VG advanced ---
    "volatility_guard_vpin_threshold",
    "volatility_guard_velocity_window_sec",
    "vg_vpin_continuous_enabled",
    "vg_vpin_continuous_min",
    "vg_vpin_buy_extra_mult",
    "vg_inv_skew_damping_enabled",
    # --- loss / PnL ---
    "loss_boost_decay_tau_sec",
    "loss_cap_auto",
    "loss_cap_update_interval",
    "loss_cap_warning_ratio",
    "min_loss_cap_jpy",
    "pnl_fee_deduction_enabled",
    "soft_loss_cap_lot_divisor",
    "recent_pnl_window",
    # --- lot / adapt ---
    "enable_dynamic_lot",
    "confidence_lot_mode",
    "enable_auto_adapt",
    "adapt_interval_cycles",
    "adapt_min_side_samples",
    "adapt_recency_window",
    "min_adapt_samples",
    "lot_adapt_interval_cycles",
    # --- low vol boost ---
    "low_vol_boost_proportional",
    "low_vol_boost_min",
    # --- ranging buy ---
    "ranging_buy_low_vol_as_offset",
    # --- unknown regime ---
    "unknown_regime_max_consecutive",
    # --- smart side ---
    "smart_side_enabled",
    "smart_side_mode",
    "smart_side_max_consecutive",
    # --- FFD advanced ---
    "ffd_boost_release_streak",
    "ffd_l2_deadzone_bps",
    # --- stale cooldown ---
    "stale_cooldown_sec",
    # --- preflight ---
    "preflight_pause_enabled",
    "preflight_pause_threshold",
    "preflight_pause_sec",
    "preflight_max_pauses",
    "max_preflight_skip",
    # --- misc operational ---
    # 348# balance_forced 撤廃: forced_fill_pnl_downweight, forced_buy/sell_kpi_tracking 削除
    # 343# skip_gate/kill 連携
    "skip_gate_kill_release_grace_cycles",
    "skip_gate_kill_release_offset",
    "dust_sweep_enabled",
    "phantom_detection_sleep_multiplier",
    "max_086_consecutive_wait",
    "fallback_duration_sec",
    "velocity_ema_alpha",
    "mid_trend_validity_sec",
    # --- regime multipliers (not structural) ---
    "regime_warmup_multiplier",
    "regime_high_vol_multiplier",
    "regime_timeout_multipliers",
    "regime_reprice_adjustments",
    # 306# Microprice side selection
    "microprice_side_enabled",
    "microprice_side_threshold",
    # 306# Dynamic cycle interval
    "dynamic_cycle_interval_enabled",
    "dynamic_cycle_interval_min_sec",
    "dynamic_cycle_interval_max_sec",
    "dynamic_cycle_interval_sigma_ref",
    # 306# Queue position estimation
    "queue_position_tracking_enabled",
    "queue_position_early_cancel_prob",
    # 306# Offset stage recording
    "offset_stage_recording_enabled",
    # 306# Offset ceiling (300# 構造的矛盾 #2)
    "offset_ceiling_ratio",
    # 320# サイド別 ceiling (321# hot-reload 追加)
    "offset_ceiling_ratio_buy",
    "offset_ceiling_ratio_sell",
    # 421# P0: Execution Final Clamp
    "execution_final_clamp_enabled",
    "execution_final_clamp_hard_skip_mult",
    # 467# deep-night ceiling 緩和
    "hour_ceiling_mult",
    # --- 491# Composite Risk Score ---
    "composite_risk_enabled",
    "composite_risk_threshold",
    "composite_risk_weight_unknown_regime",
    "composite_risk_weight_ranging_low_vol",
    "composite_risk_weight_trending_sell",
    "composite_risk_weight_velocity",
    # --- 374# Phase 3.1: SAC Sidecar Proportional Boost ---
    "sidecar_enabled",
    "sidecar_max_boost_bps",
    "sidecar_dead_zone",
    "sidecar_shaping",
    "sidecar_use_v2",
    # ══════════════════════════════════════════════════════════════════
    # 498# 横展開: micro-timeout / recovery_skew / cross-venue 閾値 等
    # ══════════════════════════════════════════════════════════════════
    # --- micro-timeout (fill_cycle_executor 毎サイクル読み) ---
    "micro_timeout_enabled",
    "micro_timeout_wait_sec",
    "micro_timeout_wait_sec_sell",
    "micro_timeout_max_requote",
    "micro_timeout_requote_cooloff_sec",
    "micro_timeout_cancel_on_cv_flip",
    # --- recovery_skew (522# 撤廃済みだが後方互換で残置) ---
    # --- cross-venue 閾値 (maker_risk_guards 毎サイクル読み) ---
    # ※ cross_venue_lead_lag_enabled / cross_venue_reference_exchange は
    #   WebSocket 初期化に関わるため除外
    "cross_venue_lead_lag_max_age_sec",
    "cross_venue_lead_lag_spread_bps_threshold",
    "cross_venue_lead_lag_velocity_bps_threshold",
    "cross_venue_lead_lag_offset_boost",
    "cross_venue_lead_lag_veto_enabled",
    "cross_venue_lead_lag_veto_threshold_bps",
    "cross_venue_reference_ob_depth",
    "cross_venue_microprice_enabled",
    "cross_venue_depth_imbalance_enabled",
    "cross_venue_depth_imbalance_boost",
    "cross_venue_depth_imbalance_threshold",
    "cross_venue_ema_alpha",
    "cross_venue_min_confidence",
    "cross_venue_confidence_reference_spread_bps",
    "cross_venue_confidence_floor",
    # 506# basis correction
    "cross_venue_basis_correction_enabled",
    "cross_venue_basis_ema_alpha",
    # 512# favorable-side tightening
    "cross_venue_favorable_tighten_enabled",
    "cross_venue_favorable_tighten_mult",
    # 506# sell age cap
    "sell_age_cap_sec",
    # --- macro regime 閾値 (MacroRegimeDetector 毎 update 読み) ---
    "macro_regime_bucket_sec",
    "macro_regime_slope_threshold",
    "macro_regime_strong_threshold",
    # --- regime detector パラメータ (config 参照で毎サイクル読み) ---
    "regime_window",
    "regime_trend_threshold_pct",
    "regime_hysteresis_count",
    "regime_min_confidence",
    "regime_mid_confidence_lo",
    "regime_mid_confidence_hi",
    "regime_mid_confidence_offset_boost",
    "regime_ranging_offset_discount_buy",
    "regime_ranging_offset_discount_sell",
    # --- microprice パラメータ ---
    "microprice_depth",
    "microprice_min_qty",
    "microprice_side_min_spread_bps",
    "microprice_side_regime_gate",
    # --- sell/unknown offset boost (maker_price 毎サイクル読み) ---
    "sell_hour_offset_boost",
    "unknown_sell_offset_boost",
    # --- ranging sell soft mode (459# 対称) ---
    "skip_ranging_sell_low_vol",
    "ranging_sell_low_vol_as_offset",
    # --- VPIN vol-sync パラメータ ---
    "vpin_vol_sync_enabled",
    "vpin_vol_sync_bucket_btc",
    "vpin_vol_sync_n_buckets",
    # --- Bayesian regime パラメータ ---
    "bayesian_regime_enabled",
    "bayesian_regime_stickiness",
    "bayesian_regime_emission_lr",
    # --- GLFT dynamic k ---
    "glft_dynamic_k_enabled",
    "glft_dynamic_k_min_samples",
    # --- sigma clustering 閾値 ---
    "sigma_clustering_low_threshold",
    "sigma_clustering_high_threshold",
    "sigma_clustering_extreme_threshold",
    "sigma_floor",
    "vol_ratio_floor",
    # --- health monitor 閾値 (ランタイム参照) ---
    "hm_rss_warn_mb",
    "hm_rss_critical_mb",
    "hm_disk_free_warn_gb",
    "hm_gc_interval_cycles",
    "halt_sleep_multiplier",
})

# 構造体再構築が必要なコンポーネントのマッピング
# field_prefix -> コールバック名
_COMPONENT_REBUILD_PREFIXES: dict[str, str] = {
    "sell_dynamic_kill_": "_rebuild_sell_kill_mgr",
    "buy_dynamic_kill_": "_rebuild_buy_kill_mgr",
    "daily_drawdown_": "_rebuild_daily_drawdown_guard",
    "per_side_dd_": "_rebuild_daily_drawdown_guard",  # 215# P0-B: 片側 DD も同再構築
    "fast_fill_": "_rebuild_fast_fill_defense",
}


class ConfigHotReloader:
    """YAML config hot-reload manager.

    サイクル間の自然なリロードポイントで呼び出され、
    YAML ファイルの mtime を確認し、変更があれば安全なフィールドのみ差分更新。
    """

    def __init__(
        self,
        config: FillTestConfig,
        yaml_path: str | Path | None,
        yaml_cfg: dict[str, object],
        check_interval_sec: float = 120.0,
    ) -> None:
        self._config = config
        self._yaml_path: Path | None = (
            Path(yaml_path) if yaml_path is not None else None
        )
        self._yaml_cfg = yaml_cfg
        self._check_interval_sec = check_interval_sec
        self._last_check_time: float = time.time()
        self._last_mtime: float = self._get_mtime()
        self._reload_count: int = 0
        self._last_reload_time: float = 0.0

    @property
    def reload_count(self) -> int:
        return self._reload_count

    def _get_mtime(self) -> float:
        """YAML ファイルの最終更新時刻を取得."""
        if self._yaml_path is None:
            return 0.0
        try:
            return os.path.getmtime(self._yaml_path)
        except OSError:
            return 0.0

    def maybe_reload(self, runner: _HotReloadableRunner) -> bool:
        """mtime check → 変更検出時にリロード実行.

        Args:
            runner: FillTestRunner インスタンス (コンポーネント再構築用)

        Returns:
            True if reload was performed.
        """
        now = time.time()
        if now - self._last_check_time < self._check_interval_sec:
            return False

        self._last_check_time = now
        current_mtime = self._get_mtime()

        if current_mtime <= self._last_mtime:
            return False

        # mtime changed → reload
        logger.info(
            f"[config_hot_reload] YAML change detected "
            f"(mtime {self._last_mtime:.0f} → {current_mtime:.0f}), "
            f"reloading config..."
        )
        self._last_mtime = current_mtime

        try:
            return self._do_reload(runner)
        except Exception as e:
            logger.error(
                f"[config_hot_reload] Reload FAILED, keeping old config: {e}",
                exc_info=True,
            )
            return False

    def _do_reload(self, runner: _HotReloadableRunner) -> bool:
        """実際のリロード処理."""
        from scripts.v460.lib.config_loader import load_fill_test_config

        if self._yaml_path is None:
            return False

        # 新 YAML 読込 + FillTestConfig 構築 (バリデーション含む)
        new_yaml_cfg = load_fill_test_config(self._yaml_path)
        new_config = type(self._config).from_yaml(new_yaml_cfg)

        # 差分検出 & 適用
        changed_fields: list[str] = []
        skipped_fields: list[str] = []
        rebuild_needed: set[str] = set()
        current_values = vars(self._config)
        new_values = vars(new_config)

        for field_name in _fill_config_field_names():
            old_val = current_values[field_name]
            new_val = new_values[field_name]
            if old_val == new_val:
                continue

            if field_name in _HOT_RELOADABLE_FIELDS:
                current_values[field_name] = new_val
                changed_fields.append(field_name)
                logger.info(
                    f"[config_hot_reload]   {field_name}: {old_val!r} → {new_val!r}"
                )

                # コンポーネント再構築が必要か判定
                for prefix, callback_name in _COMPONENT_REBUILD_PREFIXES.items():
                    if field_name.startswith(prefix):
                        rebuild_needed.add(callback_name)
            else:
                skipped_fields.append(field_name)

        if not changed_fields:
            logger.info("[config_hot_reload] No hot-reloadable fields changed")
            if skipped_fields:
                logger.warning(
                    f"[config_hot_reload] Non-reloadable fields changed "
                    f"(restart required): {skipped_fields}"
                )
            return False

        # コンポーネント再構築
        for callback_name in rebuild_needed:
            try:
                callback = getattr(runner, callback_name, None)
                if callback is not None:
                    callback()
                    logger.info(
                        f"[config_hot_reload]   component rebuilt: {callback_name}"
                    )
                    # 210# H2: FFD 再構築後に MakerPriceCalculator の参照を同期
                    # _rebuild_fast_fill_defense() は runner._fast_fill_defense を
                    # 新インスタンスに差し替えるが、_maker_price 側は旧参照を保持したまま
                    # になるため、明示的に同期する。
                    if callback_name == "_rebuild_fast_fill_defense":
                        # 261# P2-6: _HotReloadableRunner Protocol で宣言済み → 直接参照
                        _ffd = runner._fast_fill_defense
                        if _ffd is not None:
                            runner._maker_price.update_fast_fill_defense(_ffd)
                            logger.info(
                                "[config_hot_reload]   MakerPriceCalculator._fast_fill_defense synced"
                            )
            except Exception as e:
                logger.error(
                    f"[config_hot_reload]   component rebuild FAILED: {callback_name}: {e}",
                    exc_info=True,
                )

        # TimeFilter 再構築 (config から直接読み取るため再構築が必要)
        if any(f.startswith(("enable_time_filter", "skip_utc_hours", "regime_adaptive_")) for f in changed_fields):
            try:
                runner._time_filter = _resolve_time_filter_cls()(self._config)
                logger.info("[config_hot_reload]   TimeFilter rebuilt")
            except Exception as e:
                logger.error(f"[config_hot_reload]   TimeFilter rebuild FAILED: {e}")

        # MakerPriceCalculator の base offset 更新
        if any(f.startswith("spread_offset_ratio") for f in changed_fields):
            runner._maker_price.base_offset_ratio = self._config.spread_offset_ratio
            runner._maker_price.base_offset_ratio_buy = self._config.spread_offset_ratio_buy
            runner._maker_price.base_offset_ratio_sell = self._config.spread_offset_ratio_sell
            logger.info("[config_hot_reload]   MakerPriceCalculator offsets updated")

        # git SHA の再取得
        try:
            from ztb.utils.git_utils import get_git_sha
            new_sha = get_git_sha()
            if new_sha != runner._git_sha:
                old_sha = runner._git_sha
                runner._git_sha = new_sha
                logger.info(
                    f"[config_hot_reload]   git SHA updated: {old_sha} → {new_sha}"
                )
        except Exception as e:
            logger.warning(f"[config_hot_reload]   git SHA update failed: {e}")

        # 179# regime_policy セクション変更 → CycleStrategy 再構築
        # NOTE: _yaml_cfg.update() の前に比較する (更新後は old==new になるため)
        _old_rp = self._yaml_cfg.get("regime_policy", {})
        _new_rp = new_yaml_cfg.get("regime_policy", {})
        _rp_changed = _old_rp != _new_rp

        # YAML cfg も更新 (AdaptationEngine 等が参照)
        self._yaml_cfg.update(new_yaml_cfg)

        # 467# config_hash 更新 (config drift 追跡)
        try:
            from scripts.v460.lib.manifest import compute_config_hash
            runner._config_hash = compute_config_hash(self._yaml_cfg)
        except Exception:
            pass  # hash 失敗は非致命的

        if _rp_changed:
            try:
                runner._rebuild_cycle_strategy()
                logger.info("[config_hot_reload]   CycleStrategy rebuilt (regime_policy changed)")
                changed_fields.append("regime_policy")
            except Exception as e:
                logger.error(
                    f"[config_hot_reload]   CycleStrategy rebuild FAILED: {e}",
                    exc_info=True,
                )

        self._reload_count += 1
        self._last_reload_time = time.time()

        logger.info(
            f"[config_hot_reload] Reload #{self._reload_count} complete: "
            f"{len(changed_fields)} fields updated"
        )
        if skipped_fields:
            logger.warning(
                f"[config_hot_reload] Non-reloadable fields changed "
                f"(restart required): {skipped_fields}"
            )

        return True
