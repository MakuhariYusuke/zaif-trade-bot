"""
329# fill_config_parser — YAML→FillTestConfig パーサー.

328# God Object 分割 Step 3: fill_config.py から YAML パーサーを分離。
5 つのセクションパーサー + from_yaml エントリポイントを管理する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig


# ================================================================
# 354# DRY ヘルパ — sell/buy 対称パーサー共通化
# ================================================================

def _parse_dynamic_kill_block(
    section: dict, prefix: str, kwargs: dict,
) -> None:
    """sell_dynamic_kill / buy_dynamic_kill の共通パース.

    Args:
        section: YAML の ``sell_dynamic_kill`` or ``buy_dynamic_kill`` dict.
        prefix: ``"sell_dynamic_kill"`` or ``"buy_dynamic_kill"``.
        kwargs: 結果を追加する kwargs dict (in-place 更新).
    """
    if section.get("enabled") is not None:
        kwargs[f"{prefix}_enabled"] = section["enabled"]
    for yk, ck in {
        "window": f"{prefix}_window",
        "threshold_bps": f"{prefix}_threshold_bps",
        "resume_window": f"{prefix}_resume_window",
    }.items():
        if yk in section:
            kwargs[ck] = section[yk]
    # 139# regime_thresholds
    if "regime_thresholds" in section:
        kwargs[f"{prefix}_regime_thresholds"] = section["regime_thresholds"]
    # 243# toxic_stale_multiplier
    if "toxic_stale_multiplier" in section:
        kwargs[f"{prefix}_toxic_stale_mult"] = int(section["toxic_stale_multiplier"])
    # 269# probe / force-release
    if "max_stale_kill_cycles" in section:
        kwargs[f"{prefix}_max_stale_cycles"] = int(section["max_stale_kill_cycles"])
    if "max_force_release_probes" in section:
        kwargs[f"{prefix}_max_force_probes"] = int(section["max_force_release_probes"])
    # 273# kill 時間上限
    if "max_kill_duration_sec" in section:
        kwargs[f"{prefix}_max_duration_sec"] = float(section["max_kill_duration_sec"])
    # 344# EWMA α
    if "ewma_alpha" in section:
        kwargs[f"{prefix}_ewma_alpha"] = float(section["ewma_alpha"])
    # 353# EWMA 時間減衰
    if "ewma_time_decay_tau_sec" in section:
        kwargs[f"{prefix}_ewma_time_decay_tau_sec"] = float(
            section["ewma_time_decay_tau_sec"]
        )


def _parse_inv_relaxation_block(
    section: dict, prefix: str, kwargs: dict,
) -> None:
    """buy/sell dynamic_kill inv_relaxation の共通パース.

    Args:
        section: YAML の ``*_dynamic_kill_inv_relaxation`` dict.
        prefix: ``"buy_dynamic_kill_inv_relaxation"``
                or ``"sell_dynamic_kill_inv_relaxation"``.
        kwargs: 結果を追加する kwargs dict (in-place 更新).
    """
    if section.get("enabled") is not None:
        kwargs[f"{prefix}_enabled"] = bool(section["enabled"])
    if "scale" in section:
        kwargs[f"{prefix}_scale"] = float(section["scale"])
    if "max_bps" in section:
        kwargs[f"{prefix}_max_bps"] = float(section["max_bps"])


# ================================================================
# セクションパーサー (163# God Object 分割 — fill_config.py より移動)
# WARNING: 下記関数は parse_fill_config_yaml() から呼ばれる補助関数。
#          新設定キーは対応するセクションパーサーに追加すること。
# ================================================================


def _parse_trading_features(yaml_cfg: dict) -> dict:
    """049#/054# E3/side_offset/FFD/imbalance/smart_side/early_exit/spread_adaptive."""
    kwargs: dict = {}
    # 049# E3 サンプリング
    e3 = yaml_cfg.get("e3", {})
    if "sampling_ratio" in e3:
        kwargs["e3_sampling_ratio"] = e3["sampling_ratio"]

    # 049# side 別 offset
    side_offset = yaml_cfg.get("side_offset", {})
    if "buy" in side_offset:
        kwargs["spread_offset_ratio_buy"] = side_offset["buy"]
    if "sell" in side_offset:
        kwargs["spread_offset_ratio_sell"] = side_offset["sell"]

    # 049# 即約定防御
    ffd = yaml_cfg.get("fast_fill_defense", {})
    if ffd.get("enabled") is not None:
        kwargs["fast_fill_defense_enabled"] = ffd["enabled"]
    if "threshold_sec" in ffd:
        kwargs["fast_fill_threshold_sec"] = ffd["threshold_sec"]
    if "offset_boost" in ffd:
        kwargs["fast_fill_offset_boost"] = ffd["offset_boost"]
    # 093# side 別 fast_fill_defense
    if "threshold_sec_buy" in ffd:
        kwargs["fast_fill_threshold_sec_buy"] = ffd["threshold_sec_buy"]
    if "threshold_sec_sell" in ffd:
        kwargs["fast_fill_threshold_sec_sell"] = ffd["threshold_sec_sell"]
    if "offset_boost_buy" in ffd:
        kwargs["fast_fill_offset_boost_buy"] = ffd["offset_boost_buy"]
    if "offset_boost_sell" in ffd:
        kwargs["fast_fill_offset_boost_sell"] = ffd["offset_boost_sell"]
    # 230# H-1/H-2: Layer 2 deadzone + boost release streak
    if "l2_deadzone_bps" in ffd:
        kwargs["ffd_l2_deadzone_bps"] = float(ffd["l2_deadzone_bps"])
    if "boost_release_streak" in ffd:
        kwargs["ffd_boost_release_streak"] = int(ffd["boost_release_streak"])

    # 054# S1: Orderbook Imbalance
    imb = yaml_cfg.get("imbalance", {})
    if imb.get("enabled") is not None:
        kwargs["imbalance_enabled"] = imb["enabled"]
    imb_map = {
        "depth": "imbalance_depth",
        "threshold": "imbalance_threshold",
        "offset_boost": "imbalance_offset_boost",
        "skip_threshold": "imbalance_skip_threshold",
    }
    for yaml_key, config_key in imb_map.items():
        if yaml_key in imb:
            kwargs[config_key] = imb[yaml_key]

    # 054# S2: Smart Side
    ss = yaml_cfg.get("smart_side", {})
    if ss.get("enabled") is not None:
        kwargs["smart_side_enabled"] = ss["enabled"]
    if "mode" in ss:
        kwargs["smart_side_mode"] = ss["mode"]
    if "max_consecutive_same" in ss:
        kwargs["smart_side_max_consecutive"] = ss["max_consecutive_same"]

    # 054# S3: Early Exit (テール損失カット)
    ee = yaml_cfg.get("early_exit", {})
    if ee.get("enabled") is not None:
        kwargs["early_exit_enabled"] = ee["enabled"]
    ee_map = {
        "threshold_bps": "early_exit_threshold_bps",
        "monitoring_interval_sec": "early_exit_monitor_interval_sec",
        "rapid_exit_interval_sec": "early_exit_rapid_interval_sec",
    }
    for yaml_key, config_key in ee_map.items():
        if yaml_key in ee:
            kwargs[config_key] = ee[yaml_key]

    # 054# S4: Spread Adaptive Offset
    sa = yaml_cfg.get("spread_adaptive", {})
    if sa.get("enabled") is not None:
        kwargs["spread_adaptive_enabled"] = sa["enabled"]
    sa_map = {
        "narrow_spread_bps": "narrow_spread_bps",
        "narrow_spread_boost": "narrow_spread_boost",
        "narrow_spread_boost_buy": "narrow_spread_boost_buy",    # 093#
        "narrow_spread_boost_sell": "narrow_spread_boost_sell",  # 093#
        "wide_spread_bps": "wide_spread_bps",
        "wide_spread_ratio": "wide_spread_ratio",
    }
    for yaml_key, config_key in sa_map.items():
        if yaml_key in sa:
            kwargs[config_key] = sa[yaml_key]

    return kwargs


def _parse_cross_venue_section(yaml_cfg: dict) -> dict:
    """439# cross-venue lead-lag guard YAML mapping."""
    kwargs: dict[str, object] = {}
    cv = yaml_cfg.get("cross_venue_lead_lag", {})
    if not isinstance(cv, dict):
        return kwargs

    if cv.get("enabled") is not None:
        kwargs["cross_venue_lead_lag_enabled"] = cv["enabled"]

    cv_map = {
        "reference_exchange": "cross_venue_reference_exchange",
        "max_age_sec": "cross_venue_lead_lag_max_age_sec",
        "spread_bps_threshold": "cross_venue_lead_lag_spread_bps_threshold",
        "velocity_bps_threshold": "cross_venue_lead_lag_velocity_bps_threshold",
        "offset_boost": "cross_venue_lead_lag_offset_boost",
        "veto_enabled": "cross_venue_lead_lag_veto_enabled",
        "veto_threshold_bps": "cross_venue_lead_lag_veto_threshold_bps",
        # 442# 板深度拡張
        "reference_ob_depth": "cross_venue_reference_ob_depth",
        "microprice_enabled": "cross_venue_microprice_enabled",
        "depth_imbalance_enabled": "cross_venue_depth_imbalance_enabled",
        "depth_imbalance_boost": "cross_venue_depth_imbalance_boost",
    }
    for yaml_key, config_key in cv_map.items():
        if yaml_key in cv:
            kwargs[config_key] = cv[yaml_key]

    return kwargs


def _parse_skip_gate_section(yaml_cfg: dict) -> dict:
    """062# S5: SkipGate ML フィルター YAML マッピング."""
    kwargs: dict = {}
    # 062# S5: SkipGate ML フィルター
    sg = yaml_cfg.get("skip_gate", {})
    if sg.get("enabled") is not None:
        kwargs["skip_gate_enabled"] = sg["enabled"]
    sg_map = {
        "mode": "skip_gate_mode",
        "model_path": "skip_gate_model_path",
        # 141# P1-01: side 別モデルパス
        "model_path_buy": "skip_gate_model_path_buy",
        "model_path_sell": "skip_gate_model_path_sell",
        # 188# C-1: ev_weighted SkipGate
        "model_path_buy_long": "skip_gate_model_path_buy_long",
        "model_path_sell_short": "skip_gate_model_path_sell_short",
        "ev_weighted_enabled": "skip_gate_ev_weighted_enabled",
        "ev_w30": "skip_gate_ev_w30",
        "ev_w120": "skip_gate_ev_w120",
        # 190# A/B: ev_weighted 安全弁 + 片側 balance threshold 緩和
        "ev_max_consecutive_skip": "skip_gate_ev_max_consecutive_skip",
        "ev_one_sided_threshold_shift": "skip_gate_ev_one_sided_threshold_shift",
        # 193#: ev_weighted → offset 修飾子モード
        "ev_as_offset_enabled": "skip_gate_ev_as_offset_enabled",
        "ev_offset_sensitivity": "skip_gate_ev_offset_sensitivity",
        "ev_offset_min_mult": "skip_gate_ev_offset_min_mult",
        "ev_offset_max_mult": "skip_gate_ev_offset_max_mult",
        "ev_emergency_skip_threshold": "skip_gate_ev_emergency_skip_threshold",
        # 200# M: ev warning zone
        "ev_warning_threshold": "skip_gate_ev_warning_threshold",
        "ev_warning_offset_factor": "skip_gate_ev_warning_offset_factor",
        "as_threshold": "skip_gate_as_threshold",
        "pnl_threshold": "skip_gate_pnl_threshold",
        "max_skip_rate": "skip_gate_max_skip_rate",
        # 118# A3: side 別有効/無効
        "buy_enabled": "skip_gate_buy_enabled",
        "sell_enabled": "skip_gate_sell_enabled",
        # 068# §3.3: side 別閾値
        "as_threshold_buy": "skip_gate_as_threshold_buy",
        "as_threshold_sell": "skip_gate_as_threshold_sell",
        # 072# OB トグル
        "use_ob_features": "skip_gate_use_ob_features",
        # 088# 動的閾値較正
        "adaptive_threshold": "skip_gate_adaptive_threshold",
        "target_skip_rate_buy": "skip_gate_target_skip_rate_buy",
        "target_skip_rate_sell": "skip_gate_target_skip_rate_sell",
        "adaptive_window": "skip_gate_adaptive_window",
        "adaptive_min_samples": "skip_gate_adaptive_min_samples",
        "adaptive_step": "skip_gate_adaptive_step",
        "adaptive_floor": "skip_gate_adaptive_floor",
        "adaptive_ceiling": "skip_gate_adaptive_ceiling",
        # 124# Rule: unknown regime sell skip
        "skip_sell_unknown_regime": "skip_sell_unknown_regime",
        # 130# unknown buy offset boost
        "unknown_buy_offset_boost": "unknown_buy_offset_boost",
        # 440# unknown sell offset boost
        "unknown_sell_offset_boost": "unknown_sell_offset_boost",
        # 165# AS-R1: velocity-based skip
        "sell_velocity_skip_enabled": "sell_velocity_skip_enabled",
        "sell_velocity_skip_threshold_bps": "sell_velocity_skip_threshold_bps",
        "buy_velocity_skip_enabled": "buy_velocity_skip_enabled",
        "buy_velocity_skip_threshold_bps": "buy_velocity_skip_threshold_bps",
        # 195# velocity_skip ソフト化
        "velocity_skip_as_offset_enabled": "velocity_skip_as_offset_enabled",
        "velocity_offset_boost_factor": "velocity_offset_boost_factor",
        # 196# velocity offset 段階的 boost
        "velocity_offset_proportional": "velocity_offset_proportional",
        "velocity_offset_max_mult": "velocity_offset_max_mult",
        # 141# P1-04: regime thresholds
        "regime_thresholds": "skip_gate_regime_thresholds",
        # 138# P1-03: score calibration
        "score_calibration": "skip_gate_score_calibration",
        "calibrator_path": "skip_gate_calibrator_path",
        "calibrator_min_samples": "skip_gate_calibrator_min_samples",
        "calibrator_refit_interval": "skip_gate_calibrator_refit_interval",
        # 183# narrow spread adverse guard
        "skip_gate_narrow_spread_threshold_jpy": "skip_gate_narrow_spread_threshold_jpy",
        "skip_gate_narrow_spread_offset": "skip_gate_narrow_spread_offset",
        # 187# clamp YAML外部化
        "offset_floor": "skip_gate_offset_floor",
        "offset_ceil": "skip_gate_offset_ceil",
        # 343# skip_gate/kill 連携
        "kill_release_grace_cycles": "skip_gate_kill_release_grace_cycles",
        "kill_release_offset": "skip_gate_kill_release_offset",
    }
    for yaml_key, config_key in sg_map.items():
        if yaml_key in sg and sg[yaml_key] is not None:
            kwargs[config_key] = sg[yaml_key]

    # 158# P1-6: hour_offsets (UTC hour → offset bps)
    hour_offsets_raw = sg.get("hour_offsets", {})
    if hour_offsets_raw:
        kwargs["skip_gate_hour_offsets"] = {
            int(k): float(v) for k, v in hour_offsets_raw.items()
        }

    # 205# §9.4: hard_skip_utc_hours (取引完全停止する UTC 時間帯)
    hard_skip_raw = sg.get("hard_skip_utc_hours", [])
    if hard_skip_raw:
        kwargs["hard_skip_utc_hours"] = [int(h) for h in hard_skip_raw]

    return kwargs


def _parse_stale_vg_section(yaml_cfg: dict) -> dict:
    """094#/096#/107# Stale order + VG + sell_guard."""
    kwargs: dict = {}
    # 094# stale order 検出 & cancel-replace
    so = yaml_cfg.get("stale_order", {})
    if so.get("enabled") is not None:
        kwargs["stale_order_enabled"] = so["enabled"]
    so_map = {
        "check_after_sec": "stale_check_after_sec",
        "drift_bps": "stale_drift_bps",
        "max_reprice": "stale_max_reprice",
        "cooldown_sec": "stale_cooldown_sec",
        # 096# side-specific
        "check_after_sec_buy": "stale_check_after_sec_buy",
        "check_after_sec_sell": "stale_check_after_sec_sell",
        "drift_bps_buy": "stale_drift_bps_buy",
        "drift_bps_sell": "stale_drift_bps_sell",
        "max_reprice_buy": "stale_max_reprice_buy",
        "max_reprice_sell": "stale_max_reprice_sell",
        # 158# P1-2: reprice offset tightening
        "reprice_tighten": "stale_reprice_tighten",
        "reprice_min_delta_jpy": "stale_reprice_min_delta_jpy",
        "reprice_skip_gate_offset": "stale_reprice_skip_gate_offset",
    }
    for yaml_key, config_key in so_map.items():
        if yaml_key in so:
            kwargs[config_key] = so[yaml_key]

    # 096# adaptation recency window
    adapt = yaml_cfg.get("adaptation", {})
    if adapt.get("recency_window") is not None:
        kwargs["adapt_recency_window"] = adapt["recency_window"]
    # 103# adaptation.min_samples → min_adapt_samples マッピング
    if "min_samples" in adapt:
        kwargs["min_adapt_samples"] = adapt["min_samples"]

    # 107# Volatility Guard
    vg = yaml_cfg.get("volatility_guard", {})
    if vg.get("enabled") is not None:
        kwargs["volatility_guard_enabled"] = vg["enabled"]
    vg_map = {
        "velocity_window_sec": "volatility_guard_velocity_window_sec",
        "velocity_threshold_bps": "volatility_guard_velocity_threshold_bps",
        "vpin_threshold": "volatility_guard_vpin_threshold",
        "offset_boost_factor": "volatility_guard_offset_boost_factor",
        "inv_skew_damping_enabled": "vg_inv_skew_damping_enabled",
        # 269# VPIN continuous modulator YAML 配線
        "vpin_continuous_enabled": "vg_vpin_continuous_enabled",
        "vpin_continuous_min": "vg_vpin_continuous_min",
        # 353# VPIN 非対称 buy boost
        "vpin_buy_extra_mult": "vg_vpin_buy_extra_mult",
    }
    for yaml_key, config_key in vg_map.items():
        if yaml_key in vg:
            kwargs[config_key] = vg[yaml_key]

    # 211# P1-B: Micro Circuit Breaker
    mcb = yaml_cfg.get("micro_circuit_breaker", {})
    if mcb.get("enabled") is not None:
        kwargs["mcb_enabled"] = mcb["enabled"]
    mcb_map = {
        "caution_sigma": "mcb_caution_sigma",
        "warning_sigma": "mcb_warning_sigma",
        "halt_sigma": "mcb_halt_sigma",
        "halt_cooldown_sec": "mcb_halt_cooldown_sec",
        "warning_offset_mult": "mcb_warning_offset_mult",
        "warning_interval_mult": "mcb_warning_interval_mult",
    }
    for yaml_key, config_key in mcb_map.items():
        if yaml_key in mcb:
            kwargs[config_key] = mcb[yaml_key]

    # 211# P1-C: Spread Anomaly Detector
    sad = yaml_cfg.get("spread_anomaly_detector", {})
    if sad.get("enabled") is not None:
        kwargs["sad_enabled"] = sad["enabled"]
    sad_map = {
        "wide_ratio": "sad_wide_ratio",
        "dry_ratio": "sad_dry_ratio",
        "frozen_ratio": "sad_frozen_ratio",
        "baseline_window_sec": "sad_baseline_window_sec",
    }
    for yaml_key, config_key in sad_map.items():
        if yaml_key in sad:
            kwargs[config_key] = sad[yaml_key]

    # 269# 市場理論 YAML 配線 (258#/264#/266#)
    # AS Reservation Price (Avellaneda-Stoikov)
    as_res = yaml_cfg.get("as_reservation", {})
    if as_res.get("enabled") is not None:
        kwargs["as_reservation_enabled"] = bool(as_res["enabled"])
    for yk, ck in {
        "gamma": "as_reservation_gamma",
        "tau_sec": "as_reservation_tau_sec",
    }.items():
        if yk in as_res:
            kwargs[ck] = float(as_res[yk])
    # GLFT τ動的化 (266#)
    if as_res.get("tau_dynamic_enabled") is not None:
        kwargs["as_tau_dynamic_enabled"] = bool(as_res["tau_dynamic_enabled"])
    for yk, ck in {
        "tau_dynamic_min_sec": "as_tau_dynamic_min_sec",
        "tau_dynamic_max_sec": "as_tau_dynamic_max_sec",
    }.items():
        if yk in as_res:
            kwargs[ck] = float(as_res[yk])
    # AS δ* (266#)
    if as_res.get("delta_star_enabled") is not None:
        kwargs["as_delta_star_enabled"] = bool(as_res["delta_star_enabled"])
    if "delta_star_fill_rate_k" in as_res:
        kwargs["as_delta_star_fill_rate_k"] = float(as_res["delta_star_fill_rate_k"])
    # Kyle λ (266#)
    kyle = yaml_cfg.get("kyle_lambda", {})
    if kyle.get("enabled") is not None:
        kwargs["kyle_lambda_enabled"] = bool(kyle["enabled"])
    for yk, ck in {
        "impact_mult": "kyle_lambda_impact_mult",
        "max_add_ratio": "kyle_lambda_max_add_ratio",
    }.items():
        if yk in kyle:
            kwargs[ck] = float(kyle[yk])
    # Amihud ILLIQ (266#)
    amihud = yaml_cfg.get("amihud_illiq", {})
    if amihud.get("enabled") is not None:
        kwargs["amihud_illiq_enabled"] = bool(amihud["enabled"])
    for yk, ck in {
        "baseline": "amihud_illiq_baseline",
        "max_mult": "amihud_illiq_max_mult",
    }.items():
        if yk in amihud:
            kwargs[ck] = float(amihud[yk])

    # 286# Buy-side AS Guard (Glosten-Milgrom 1985)
    bag = yaml_cfg.get("buy_as_guard", {})
    if bag.get("enabled") is not None:
        kwargs["buy_as_guard_enabled"] = bool(bag["enabled"])
    for yk, ck in {
        "velocity_threshold_bps": "buy_as_guard_velocity_threshold_bps",
        "offset_mult": "buy_as_guard_offset_mult",
        "max_offset_ratio": "buy_as_guard_max_offset_ratio",
    }.items():
        if yk in bag:
            kwargs[ck] = float(bag[yk])

    # 088# sell 専用ハードガード
    sell_guard = yaml_cfg.get("sell_guard", {})
    if sell_guard.get("max_spread_jpy") is not None:
        kwargs["sell_max_spread_jpy"] = sell_guard["max_spread_jpy"]
    if sell_guard.get("offset_floor") is not None:
        kwargs["sell_offset_floor"] = sell_guard["offset_floor"]
    # 175# sell_offset_floor_inv_discount YAML バインド
    if sell_guard.get("offset_floor_inv_discount") is not None:
        kwargs["sell_offset_floor_inv_discount"] = float(
            sell_guard["offset_floor_inv_discount"]
        )

    return kwargs


def _parse_stopgap_section(yaml_cfg: dict) -> dict:
    """133# 止血施策 + dynamic kill + narrow spread + inventory skewing."""
    kwargs: dict = {}
    # 133# P0-08/09/10: 止血施策
    止血: dict = yaml_cfg.get("止血", yaml_cfg.get("loss_control", {}))
    # 348# balance_forced 撤廃: skip_balance_forced, balance_forced_* のパースを削除
    if 止血.get("skip_buy_unknown_regime") is not None:
        kwargs["skip_buy_unknown_regime"] = 止血["skip_buy_unknown_regime"]
    # 155# §9: trending sell 抑制
    if 止血.get("skip_sell_trending") is not None:
        kwargs["skip_sell_trending"] = 止血["skip_sell_trending"]
    # 156# D-4: trending 方向別分解
    if 止血.get("skip_sell_trending_up_only") is not None:
        kwargs["skip_sell_trending_up_only"] = 止血["skip_sell_trending_up_only"]
    # 251# Sell Asymmetric Mode: high_vol でも sell skip
    if 止血.get("sell_asymmetric_high_vol_enabled") is not None:
        kwargs["sell_asymmetric_high_vol_enabled"] = 止血["sell_asymmetric_high_vol_enabled"]
    # 196# trending sell ソフト化
    if 止血.get("trending_sell_as_offset_enabled") is not None:
        kwargs["trending_sell_as_offset_enabled"] = 止血["trending_sell_as_offset_enabled"]
    if 止血.get("trending_sell_offset_boost_factor") is not None:
        kwargs["trending_sell_offset_boost_factor"] = float(止血["trending_sell_offset_boost_factor"])
    # 253# 削除済み: balance_forced_apply_trending_offset (234# dead config)
    # 158# §20-B: 連続 trending sell skip 安全弁
    if 止血.get("max_consecutive_trending_sell_skip") is not None:
        kwargs["max_consecutive_trending_sell_skip"] = 止血["max_consecutive_trending_sell_skip"]
    # 171# Guard Paradox 対策: 在庫偏重時の sell ガードバイパス閾値
    if 止血.get("sell_guard_inv_bypass_threshold") is not None:
        kwargs["sell_guard_inv_bypass_threshold"] = float(止血["sell_guard_inv_bypass_threshold"])
    sell_kill = 止血.get("sell_dynamic_kill", {})
    _parse_dynamic_kill_block(sell_kill, "sell_dynamic_kill", kwargs)

    # 157# §19: buy 動的 kill
    buy_kill = 止血.get("buy_dynamic_kill", {})
    _parse_dynamic_kill_block(buy_kill, "buy_dynamic_kill", kwargs)

    # 286# 283# P1-4: 在庫連動 buy_dynamic_kill 緩和 (Ho & Stoll 1981)
    _parse_inv_relaxation_block(
        止血.get("buy_dynamic_kill_inv_relaxation", {}),
        "buy_dynamic_kill_inv_relaxation",
        kwargs,
    )

    # 337# 在庫連動 sell_dynamic_kill 緩和 (Ho & Stoll 1981 対称性)
    _parse_inv_relaxation_block(
        止血.get("sell_dynamic_kill_inv_relaxation", {}),
        "sell_dynamic_kill_inv_relaxation",
        kwargs,
    )

    # 348# balance_forced 撤廃: forced KPI/delay/downweight のパースを削除

    # 249# dual_kill_quiescence
    _dkq = 止血.get("dual_kill_quiescence_enabled")
    if _dkq is not None:
        kwargs["dual_kill_quiescence_enabled"] = bool(_dkq)

    # 137# P1-08: narrow spread pause
    narrow_pause = 止血.get("narrow_spread_pause", {})
    if narrow_pause.get("enabled") is not None:
        kwargs["narrow_spread_pause_enabled"] = narrow_pause["enabled"]
    for yk, ck in {
        "threshold_bps": "narrow_spread_pause_bps",
        "pause_sec": "narrow_spread_pause_sec",
        "max_consecutive": "narrow_spread_pause_max_consecutive",
    }.items():
        if yk in narrow_pause:
            kwargs[ck] = narrow_pause[yk]


    # 162# Inventory Skewing: 在庫偏重による非対称クオート
    inv_skew = 止血.get("inventory_skewing", {})
    if inv_skew.get("enabled") is not None:
        kwargs["inventory_skewing_enabled"] = inv_skew["enabled"]
    for yk, ck in {
        "window": "inventory_skewing_window",
        "max_factor": "inventory_skewing_max_factor",
        "neutral_band": "inventory_skewing_neutral_band",
        "decay_tau_sec": "inv_decay_tau_sec",  # 228# C2
    }.items():
        if yk in inv_skew:
            kwargs[ck] = inv_skew[yk]
    # 249# regime gate
    if "regime_gate_enabled" in inv_skew:
        kwargs["inv_skew_regime_gate_enabled"] = bool(inv_skew["regime_gate_enabled"])

    # 168# §4.1 #3: 日次ドローダウンガード
    dd_guard = 止血.get("daily_drawdown", {})
    if dd_guard.get("enabled") is not None:
        kwargs["daily_drawdown_enabled"] = dd_guard["enabled"]
    if "hard_limit_bps" in dd_guard:
        kwargs["daily_drawdown_hard_limit_bps"] = float(dd_guard["hard_limit_bps"])
    if "soft_limit_bps" in dd_guard:
        kwargs["daily_drawdown_soft_limit_bps"] = float(dd_guard["soft_limit_bps"])
    # 200# 10-A/10-E: soft_drawdown_interval_multiplier YAML 外部化
    if "soft_drawdown_interval_multiplier" in dd_guard:
        kwargs["soft_drawdown_interval_multiplier"] = float(dd_guard["soft_drawdown_interval_multiplier"])
    # 348# balance_forced 撤廃: balance_forced_cooldown_sec のパースを削除
    # 202# A: 単一サイクル大損失クールダウン
    if "loss_cooldown_threshold_bps" in 止血:
        kwargs["loss_cooldown_threshold_bps"] = float(止血["loss_cooldown_threshold_bps"])
    if "loss_cooldown_interval_mult" in 止血:
        kwargs["loss_cooldown_interval_mult"] = float(止血["loss_cooldown_interval_mult"])
    # 205# §9.2: Toxic Fill 同一サイド拒否
    if "toxic_fill_veto_threshold_bps" in 止血:
        kwargs["toxic_fill_veto_threshold_bps"] = float(止血["toxic_fill_veto_threshold_bps"])
    if "toxic_fill_veto_cycles" in 止血:
        kwargs["toxic_fill_veto_cycles"] = int(止血["toxic_fill_veto_cycles"])
    # 204# I: loss_boost_offset_mult + 226# T1: 指数減衰 τ
    if "loss_boost_offset_mult" in 止血:
        kwargs["loss_boost_offset_mult"] = float(止血["loss_boost_offset_mult"])
    if "loss_boost_decay_tau_sec" in 止血:
        kwargs["loss_boost_decay_tau_sec"] = float(止血["loss_boost_decay_tau_sec"])
    # 202# B: 片側残高枯渇時の rescue offset
    if "one_sided_balance_rescue_offset" in 止血:
        kwargs["one_sided_balance_rescue_offset"] = 止血["one_sided_balance_rescue_offset"]
    # 207# §4: one-sided 連続実行制限
    if "one_sided_consecutive_limit" in 止血:
        kwargs["one_sided_consecutive_limit"] = int(止血["one_sided_consecutive_limit"])
    if "one_sided_consecutive_interval_mult" in 止血:
        kwargs["one_sided_consecutive_interval_mult"] = float(止血["one_sided_consecutive_interval_mult"])
    # 234# one-sided エスカレーション
    if "one_sided_escalation_cooldown_offset" in 止血:
        kwargs["one_sided_escalation_cooldown_offset"] = int(止血["one_sided_escalation_cooldown_offset"])
    if "one_sided_escalation_cooldown_cycles" in 止血:
        kwargs["one_sided_escalation_cooldown_cycles"] = int(止血["one_sided_escalation_cooldown_cycles"])
    if "one_sided_escalation_freeze_offset" in 止血:
        kwargs["one_sided_escalation_freeze_offset"] = int(止血["one_sided_escalation_freeze_offset"])
    if "one_sided_escalation_freeze_cycles" in 止血:
        kwargs["one_sided_escalation_freeze_cycles"] = int(止血["one_sided_escalation_freeze_cycles"])
    # 234# 縮退清算モード
    if "degraded_liquidation_enabled" in 止血:
        kwargs["degraded_liquidation_enabled"] = bool(止血["degraded_liquidation_enabled"])
    if "degraded_liquidation_lot_mult" in 止血:
        kwargs["degraded_liquidation_lot_mult"] = float(止血["degraded_liquidation_lot_mult"])
    if "degraded_liquidation_offset_mult" in 止血:
        kwargs["degraded_liquidation_offset_mult"] = float(止血["degraded_liquidation_offset_mult"])
    if "degraded_liquidation_duty_cycle" in 止血:
        kwargs["degraded_liquidation_duty_cycle"] = int(止血["degraded_liquidation_duty_cycle"])
    # 269# Inventory Escape Mode
    if "inventory_escape_enabled" in 止血:
        kwargs["inventory_escape_enabled"] = bool(止血["inventory_escape_enabled"])
    if "inventory_escape_duty_cycle" in 止血:
        kwargs["inventory_escape_duty_cycle"] = int(止血["inventory_escape_duty_cycle"])
    # 205# §9.5: 片側 DD Halt (daily_drawdown サブキー)
    if dd_guard.get("per_side_enabled") is not None:
        kwargs["per_side_dd_enabled"] = dd_guard["per_side_enabled"]
    if "per_side_hard_limit_bps" in dd_guard:
        kwargs["per_side_dd_hard_limit_bps"] = float(dd_guard["per_side_hard_limit_bps"])
    if "per_side_halt_cycles" in dd_guard:
        kwargs["per_side_dd_halt_cycles"] = int(dd_guard["per_side_halt_cycles"])
    # 224# B1: halt解除後ソフトリカバリ
    if "per_side_recovery_cycles" in dd_guard:
        kwargs["per_side_dd_recovery_cycles"] = int(dd_guard["per_side_recovery_cycles"])
    if "per_side_recovery_lot_scale" in dd_guard:
        kwargs["per_side_dd_recovery_lot_scale"] = float(dd_guard["per_side_recovery_lot_scale"])
    # 269# per-side halt PnL リアンカー
    if "per_side_reanchor_budget_bps" in dd_guard:
        kwargs["per_side_dd_reanchor_budget_bps"] = float(dd_guard["per_side_reanchor_budget_bps"])
    # 225# regime-aware recovery ペナルティ
    if "recovery_trending_penalty" in dd_guard:
        kwargs["recovery_trending_penalty"] = float(dd_guard["recovery_trending_penalty"])
    if "recovery_high_vol_penalty" in dd_guard:
        kwargs["recovery_high_vol_penalty"] = float(dd_guard["recovery_high_vol_penalty"])
    # 246# DD halt cooldown release
    if "cooldown_release_sec" in dd_guard:
        kwargs["dd_cooldown_release_sec"] = float(dd_guard["cooldown_release_sec"])
    if "cooldown_release_lot_scale" in dd_guard:
        kwargs["dd_cooldown_release_lot_scale"] = float(dd_guard["cooldown_release_lot_scale"])
    # 249# DD cooldown re-arm
    if "cooldown_rearm_budget_bps" in dd_guard:
        kwargs["dd_cooldown_rearm_budget_bps"] = float(dd_guard["cooldown_rearm_budget_bps"])
    # 268# DD day reset timezone
    if "day_reset_utc_offset_hours" in dd_guard:
        kwargs["dd_day_reset_utc_offset_hours"] = float(dd_guard["day_reset_utc_offset_hours"])

    return kwargs


def _parse_infra_section(yaml_cfg: dict) -> dict:
    """137#/102#/158# PnL fee + preflight + tuning + resilience + A/B test."""
    kwargs: dict = {}
    # 163#: 止血セクション参照 (pnl_fee_deduction, preflight_pause 等のサブキー用)
    止血 = yaml_cfg.get("止血", yaml_cfg.get("loss_control", {}))
    # 137# P1-11: PnL fee 控除
    fee_cfg = 止血.get("pnl_fee_deduction", {})
    if fee_cfg.get("enabled") is not None:
        kwargs["pnl_fee_deduction_enabled"] = fee_cfg["enabled"]
    if "maker_fee_bps" in fee_cfg:
        kwargs["maker_fee_bps"] = fee_cfg["maker_fee_bps"]
    if "taker_fee_bps" in fee_cfg:
        kwargs["taker_fee_bps"] = fee_cfg["taker_fee_bps"]

    # 138# P1-10: preflight pause (dead-cycle 防止)
    pf_pause = 止血.get("preflight_pause", {})
    if pf_pause.get("enabled") is not None:
        kwargs["preflight_pause_enabled"] = pf_pause["enabled"]
    for yk, ck in {
        "threshold": "preflight_pause_threshold",
        "pause_sec": "preflight_pause_sec",
        "max_pauses": "preflight_max_pauses",
    }.items():
        if yk in pf_pause:
            kwargs[ck] = pf_pause[yk]

    # 102# YAML 化: 散在マジックナンバーの設定外部化
    tuning = yaml_cfg.get("tuning", {})
    tuning_map = {
        "max_offset_ratio": "max_offset_ratio",
        "min_offset_ratio": "min_offset_ratio",
        "loss_cap_update_interval": "loss_cap_update_interval",
        "min_loss_cap_jpy": "min_loss_cap_jpy",
        "mid_trend_validity_sec": "mid_trend_validity_sec",
        # 227# C3: velocity EMA smoothing
        "velocity_ema_alpha": "velocity_ema_alpha",
        "balance_margin_ratio": "balance_margin_ratio",
        "balance_shrink_consecutive": "balance_shrink_consecutive",
        "balance_shrink_divisor": "balance_shrink_divisor",
        "skip_gate_recent_trades_limit": "skip_gate_recent_trades_limit",
        "status_unknown_retry_delays": "status_unknown_retry_delays",
        "rate_limit_min_backoff_sec": "rate_limit_min_backoff_sec",
        "save_retry_backoff_sec": "save_retry_backoff_sec",
        "regime_warmup_multiplier": "regime_warmup_multiplier",
        "e3_60s_multiplier": "e3_60s_multiplier",
        "e3_120s_multiplier": "e3_120s_multiplier",
        "adapt_min_side_samples": "adapt_min_side_samples",
        "batch_flush_interval_sec": "batch_flush_interval_sec",
        "heartbeat_interval_sec": "heartbeat_interval_sec",
        # 121# 追加外部化
        "min_order_btc": "min_order_btc",
        "dust_sweep_enabled": "dust_sweep_enabled",  # 128#
        "lock_acquire_retries": "lock_acquire_retries",
        "skip_gate_ob_depth": "skip_gate_ob_depth",
        "retry_backoff_base": "retry_backoff_base",
        "soft_loss_cap_lot_divisor": "soft_loss_cap_lot_divisor",
        "file_log_level": "file_log_level",
        "insufficient_funds_patterns": "insufficient_funds_patterns",
        # 148# §9 #2: heartbeat 設定を YAML から調整可能に
        "lock_heartbeat_period_sec": "lock_heartbeat_period_sec",
        "lock_stale_heartbeat_sec": "lock_stale_heartbeat_sec",
        # 158# YAML 外部化: tuning 追加
        "hot_reload_check_interval_sec": "hot_reload_check_interval_sec",
        "records_cache_ttl_sec": "records_cache_ttl_sec",
        "trades_recorder_fetch_limit": "trades_recorder_fetch_limit",
        "balance_freeze_cycles": "balance_freeze_cycles",
    }
    for yaml_key, config_key in tuning_map.items():
        if yaml_key in tuning:
            kwargs[config_key] = tuning[yaml_key]

    # 158# YAML 外部化: resilience セクション (CircuitBreaker / HealthMonitor)
    resilience = yaml_cfg.get("resilience", {})
    cb = resilience.get("circuit_breaker", {})
    cb_map = {
        "failure_threshold": "cb_failure_threshold",
        "recovery_timeout": "cb_recovery_timeout",
        "success_threshold": "cb_success_threshold",
        "timeout": "cb_timeout",
    }
    for yaml_key, config_key in cb_map.items():
        if yaml_key in cb:
            kwargs[config_key] = cb[yaml_key]
    hm = resilience.get("health_monitor", {})
    hm_map = {
        "rss_warn_mb": "hm_rss_warn_mb",
        "rss_critical_mb": "hm_rss_critical_mb",
        "disk_free_warn_gb": "hm_disk_free_warn_gb",
        "gc_interval_cycles": "hm_gc_interval_cycles",
        "check_interval_sec": "hm_check_interval_sec",
    }
    for yaml_key, config_key in hm_map.items():
        if yaml_key in hm:
            kwargs[config_key] = hm[yaml_key]

    # 158# P1-5: A/B テスト variant 識別子
    ab_test = yaml_cfg.get("ab_test", {})
    if ab_test.get("variant"):
        kwargs["ab_test_variant"] = str(ab_test["variant"])

    return kwargs


def parse_fill_config_yaml(yaml_cfg: dict) -> FillTestConfig:
    """YAML dict から FillTestConfig を構築.

    YAML のフラットキー + ネスト (adaptation / lot_sizing / safety) を
    dataclass フィールドにマッピングする.

    329# Step 3: fill_config.py の from_yaml() から分離。
    FillTestConfig.from_yaml() はこの関数に委譲する。
    """
    from scripts.v460.lib.fill_config import FillTestConfig as _FillTestConfig

    kwargs: dict = {}

    # フラットキー (YAML キー == dataclass フィールド名)
    flat_keys = {
        "symbol", "order_quantity", "cycle_interval_sec", "max_cycle_sleep_sec",
        "halt_sleep_multiplier",  # 276#
        "phantom_detection_sleep_multiplier",  # 277#
        "halt_persist_interval",  # 277#
        "stop_condition_check_interval",  # 277#
        "fallback_duration_sec",  # 277#
        "unknown_regime_max_consecutive",  # 277#
        "quiescence_gate_blocks_threshold", "quiescence_sleep_sec",  # 243#
        "order_timeout_sec",
        "order_timeout_sec_sell",
        "poll_interval_sec", "post_fill_wait_sec", "post_fill_wait_sec_sell",
        "results_dir",
        "max_preflight_skip", "start_side",
        "spread_offset_ratio", "min_offset_jpy",
        "max_order_retries", "retry_delay_sec",
        "as_deadzone_bps", "min_spread_jpy",
        "batch_size", "max_save_retries", "save_fail_threshold",
        "progress_log_interval",
        "log_max_bytes", "log_backup_count",
        "fallback_stale_sec",  # 156# §16
    }
    for key in flat_keys:
        if key in yaml_cfg:
            kwargs[key] = yaml_cfg[key]

    # adaptation セクション → 方策 A
    adapt = yaml_cfg.get("adaptation", {})
    if adapt.get("enabled") is not None:
        kwargs["enable_auto_adapt"] = adapt["enabled"]
    if "interval_cycles" in adapt:
        kwargs["adapt_interval_cycles"] = adapt["interval_cycles"]

    # lot_sizing セクション → 方策 B
    lot = yaml_cfg.get("lot_sizing", {})
    if lot.get("enabled") is not None:
        kwargs["enable_dynamic_lot"] = lot["enabled"]
    if "interval_cycles" in lot:
        kwargs["lot_adapt_interval_cycles"] = lot["interval_cycles"]
    if "max_lot" in lot:
        kwargs["max_lot"] = lot["max_lot"]
    if "recent_pnl_window" in lot:
        kwargs["recent_pnl_window"] = lot["recent_pnl_window"]

    # 151# P3-03: confidence_lot セクション (AS 確率連動ロットサイジング)
    cl = yaml_cfg.get("confidence_lot", {})
    if cl.get("enabled") is not None:
        kwargs["enable_confidence_lot"] = cl["enabled"]
    cl_map = {
        "scale": "confidence_lot_scale",
        "floor": "confidence_lot_floor",
        "mode": "confidence_lot_mode",
    }
    for yaml_key, config_key in cl_map.items():
        if yaml_key in cl:
            kwargs[config_key] = cl[yaml_key]

    # regime セクション → レジーム検知 (035# §4)
    regime = yaml_cfg.get("regime", {})
    if regime.get("enabled") is not None:
        kwargs["enable_regime"] = regime["enabled"]
    regime_map = {
        "window": "regime_window",
        "trend_threshold_pct": "regime_trend_threshold_pct",
        "high_vol_multiplier": "regime_high_vol_multiplier",
        "hysteresis_count": "regime_hysteresis_count",
        "min_confidence": "regime_min_confidence",
        "trending_offset_boost": "regime_trending_offset_boost",
        "trending_offset_boost_buy": "regime_trending_offset_boost_buy",    # 157# §19
        "trending_offset_boost_sell": "regime_trending_offset_boost_sell",   # 157# §19
        # 176# B: 方向×サイド別 offset boost
        "trending_up_buy_offset_boost": "trending_up_buy_offset_boost",
        "trending_up_sell_offset_boost": "trending_up_sell_offset_boost",
        "trending_down_buy_offset_boost": "trending_down_buy_offset_boost",
        "trending_down_sell_offset_boost": "trending_down_sell_offset_boost",
        "high_vol_offset_boost": "regime_high_vol_offset_boost",       # 143# R-1a
        "ranging_offset_discount": "regime_ranging_offset_discount",   # 143# R-1a
        # 440# ranging offset buy/sell 非対称化
        "ranging_offset_discount_buy": "regime_ranging_offset_discount_buy",
        "ranging_offset_discount_sell": "regime_ranging_offset_discount_sell",
        # 227# C1: Ranging × OBI 方向別非対称 offset
        "ranging_obi_asymmetry_factor": "ranging_obi_asymmetry_factor",
        "ranging_obi_threshold": "ranging_obi_threshold",
        # 397# mid-confidence paradox guard
        "mid_confidence_offset_boost": "regime_mid_confidence_offset_boost",
        "mid_confidence_lo": "regime_mid_confidence_lo",
        "mid_confidence_hi": "regime_mid_confidence_hi",
        "low_vol_offset_boost_enabled": "low_vol_offset_boost_enabled", # 168#
        "low_vol_offset_boost": "low_vol_offset_boost",               # 168#
        "low_vol_threshold": "low_vol_threshold",                     # 168#
        # 200# C: low_vol proportional boost
        "low_vol_boost_proportional": "low_vol_boost_proportional",
        "low_vol_boost_min": "low_vol_boost_min",
        "skip_ranging_buy_low_vol": "skip_ranging_buy_low_vol",       # 169# B1'
        "ranging_buy_low_vol_as_offset": "ranging_buy_low_vol_as_offset", # 195# B1' ソフト化
    }
    for yaml_key, config_key in regime_map.items():
        if yaml_key in regime:
            kwargs[config_key] = regime[yaml_key]
    # 189# D: macro_regime サブセクション
    macro_cfg = regime.get("macro", {})
    if isinstance(macro_cfg, dict):
        macro_map = {
            "enabled": "enable_macro_regime",
            "bucket_sec": "macro_regime_bucket_sec",
            "slope_threshold": "macro_regime_slope_threshold",
            "strong_threshold": "macro_regime_strong_threshold",
            "conflict_action": "macro_regime_conflict_action",
        }
        for yaml_key, config_key in macro_map.items():
            if yaml_key in macro_cfg:
                kwargs[config_key] = macro_cfg[yaml_key]
    # 143# R-1b: レジーム別 lot 倍率
    if "lot_multipliers" in regime and isinstance(regime["lot_multipliers"], dict):
        kwargs["regime_lot_multipliers"] = {
            str(k): float(v) for k, v in regime["lot_multipliers"].items()
        }
    # 144# R-1c: レジーム別 reprice 上限調整
    if "reprice_adjustments" in regime and isinstance(regime["reprice_adjustments"], dict):
        kwargs["regime_reprice_adjustments"] = {
            str(k): int(v) for k, v in regime["reprice_adjustments"].items()
        }
    # 144# R-1d: レジーム別 timeout 倍率
    if "timeout_multipliers" in regime and isinstance(regime["timeout_multipliers"], dict):
        kwargs["regime_timeout_multipliers"] = {
            str(k): float(v) for k, v in regime["timeout_multipliers"].items()
        }

    # safety セクション → 損失キャップ
    safety = yaml_cfg.get("safety", {})
    if "loss_cap_jpy" in safety:
        kwargs["loss_cap_jpy"] = safety["loss_cap_jpy"]
    if "loss_cap_warning_ratio" in safety:
        kwargs["loss_cap_warning_ratio"] = safety["loss_cap_warning_ratio"]
    # 041# 動的 loss_cap
    if safety.get("loss_cap_auto") is not None:
        kwargs["loss_cap_auto"] = safety["loss_cap_auto"]
    if "loss_cap_ratio" in safety:
        kwargs["loss_cap_ratio"] = safety["loss_cap_ratio"]
    # 046# soft/hard 二段 loss_cap
    if "soft_loss_cap_ratio" in safety:
        kwargs["soft_loss_cap_ratio"] = safety["soft_loss_cap_ratio"]

    # 041# 時間帯フィルター
    tf = yaml_cfg.get("time_filter", {})
    if tf.get("enabled") is not None:
        kwargs["enable_time_filter"] = tf["enabled"]
    if "skip_utc_hours" in tf:
        kwargs["skip_utc_hours"] = tf["skip_utc_hours"]
    # 073# side 別時間帯フィルター
    if "skip_utc_hours_buy" in tf:
        kwargs["skip_utc_hours_buy"] = tf["skip_utc_hours_buy"]
    if "skip_utc_hours_sell" in tf:
        kwargs["skip_utc_hours_sell"] = tf["skip_utc_hours_sell"]
    # 110# 086# デッドロック修正
    if "max_086_consecutive_wait" in tf:
        kwargs["max_086_consecutive_wait"] = tf["max_086_consecutive_wait"]

    # 163# regime 連動動的ゲーティング
    if tf.get("regime_adaptive_enabled"):
        kwargs["regime_adaptive_enabled"] = True
    if "regime_adaptive_extra_buy" in tf:
        kwargs["regime_adaptive_extra_buy"] = tf["regime_adaptive_extra_buy"]
    if "regime_adaptive_extra_sell" in tf:
        kwargs["regime_adaptive_extra_sell"] = tf["regime_adaptive_extra_sell"]

    # 163# ステージ抽出: trading features
    kwargs.update(_parse_trading_features(yaml_cfg))

    # 439# cross-venue lead-lag guard
    kwargs.update(_parse_cross_venue_section(yaml_cfg))

    # 163# ステージ抽出: skip_gate ML filter
    kwargs.update(_parse_skip_gate_section(yaml_cfg))

    # 163# ステージ抽出: stale order + VG + sell_guard
    kwargs.update(_parse_stale_vg_section(yaml_cfg))

    # 163# ステージ抽出: 止血 + dynamic kill + narrow spread + inventory skewing
    kwargs.update(_parse_stopgap_section(yaml_cfg))

    # 163# ステージ抽出: PnL fee + preflight + tuning + resilience + A/B test
    kwargs.update(_parse_infra_section(yaml_cfg))

    # ---- 306# 新機能セクション ----
    # Parkinson σ
    sp = yaml_cfg.get("sigma_parkinson", {})
    if sp.get("enabled") is not None:
        kwargs["sigma_parkinson_enabled"] = sp["enabled"]
    if "window_sec" in sp:
        kwargs["sigma_parkinson_window_sec"] = float(sp["window_sec"])
    # 330# σ / vol_ratio floor (sigma_parkinson セクション配下)
    if "sigma_floor" in sp:
        kwargs["sigma_floor"] = float(sp["sigma_floor"])
    if "vol_ratio_floor" in sp:
        kwargs["vol_ratio_floor"] = float(sp["vol_ratio_floor"])
    # none レジーム Passive MM
    nr = yaml_cfg.get("none_regime", {})
    if nr.get("passive_mm_enabled") is not None:
        kwargs["none_regime_passive_mm_enabled"] = nr["passive_mm_enabled"]
    if "fixed_offset_bps" in nr:
        kwargs["none_regime_fixed_offset_bps"] = float(nr["fixed_offset_bps"])
    # DD Guard soft lot side-aware
    if "daily_drawdown_soft_lot_side_aware" in yaml_cfg:
        kwargs["daily_drawdown_soft_lot_side_aware"] = yaml_cfg["daily_drawdown_soft_lot_side_aware"]
    # Microprice side selection
    ms = yaml_cfg.get("microprice_side", {})
    if ms.get("enabled") is not None:
        kwargs["microprice_side_enabled"] = ms["enabled"]
    if "threshold_bps" in ms:
        kwargs["microprice_side_threshold"] = float(ms["threshold_bps"])
    # 310# C: L2 guardrails
    if "min_spread_bps" in ms:
        kwargs["microprice_side_min_spread_bps"] = float(ms["min_spread_bps"])
    if "regime_gate" in ms:
        kwargs["microprice_side_regime_gate"] = [str(r) for r in ms["regime_gate"]]
    # Dynamic cycle interval
    dci = yaml_cfg.get("dynamic_cycle_interval", {})
    if dci.get("enabled") is not None:
        kwargs["dynamic_cycle_interval_enabled"] = dci["enabled"]
    for yk, ck in {
        "min_sec": "dynamic_cycle_interval_min_sec",
        "max_sec": "dynamic_cycle_interval_max_sec",
        "sigma_ref": "dynamic_cycle_interval_sigma_ref",
    }.items():
        if yk in dci:
            kwargs[ck] = float(dci[yk])
    # Queue position estimation
    qp = yaml_cfg.get("queue_position", {})
    if qp.get("tracking_enabled") is not None:
        kwargs["queue_position_tracking_enabled"] = qp["tracking_enabled"]
    if "early_cancel_prob" in qp:
        kwargs["queue_position_early_cancel_prob"] = float(qp["early_cancel_prob"])
    # Offset stage recording
    osr = yaml_cfg.get("offset_stage_recording", {})
    if osr.get("enabled") is not None:
        kwargs["offset_stage_recording_enabled"] = osr["enabled"]
    # 310# A: Sell AS Time-of-Day Offset Boost
    shob = yaml_cfg.get("sell_hour_offset_boost", {})
    if shob:
        kwargs["sell_hour_offset_boost"] = {
            int(k): float(v) for k, v in shob.items()
        }
    # Offset ceiling
    if "offset_ceiling_ratio" in yaml_cfg:
        kwargs["offset_ceiling_ratio"] = float(yaml_cfg["offset_ceiling_ratio"])
    # 321# CRITICAL fix: 320# で追加したサイド別 ceiling が未パースだった
    if "offset_ceiling_ratio_buy" in yaml_cfg:
        kwargs["offset_ceiling_ratio_buy"] = float(yaml_cfg["offset_ceiling_ratio_buy"])
    if "offset_ceiling_ratio_sell" in yaml_cfg:
        kwargs["offset_ceiling_ratio_sell"] = float(yaml_cfg["offset_ceiling_ratio_sell"])
    # ---- 421# P0: Execution Final Clamp ----
    if "execution_final_clamp_enabled" in yaml_cfg:
        kwargs["execution_final_clamp_enabled"] = bool(yaml_cfg["execution_final_clamp_enabled"])
    if "execution_final_clamp_hard_skip_mult" in yaml_cfg:
        kwargs["execution_final_clamp_hard_skip_mult"] = float(
            yaml_cfg["execution_final_clamp_hard_skip_mult"]
        )

    # ---- 366# 市場理論システム M2-M5 ----
    # M2: Bayesian Regime Filter
    br = yaml_cfg.get("bayesian_regime", {})
    if br.get("enabled") is not None:
        kwargs["bayesian_regime_enabled"] = br["enabled"]
    if "stickiness" in br:
        kwargs["bayesian_regime_stickiness"] = float(br["stickiness"])
    if "emission_lr" in br:
        kwargs["bayesian_regime_emission_lr"] = float(br["emission_lr"])
    # M3: σ-Clustering
    sc = yaml_cfg.get("sigma_clustering", {})
    if sc.get("enabled") is not None:
        kwargs["sigma_clustering_enabled"] = sc["enabled"]
    for yk, ck in {
        "low_threshold": "sigma_clustering_low_threshold",
        "high_threshold": "sigma_clustering_high_threshold",
        "extreme_threshold": "sigma_clustering_extreme_threshold",
    }.items():
        if yk in sc:
            kwargs[ck] = float(sc[yk])
    # M4: GLFT dynamic k
    glft = yaml_cfg.get("glft_dynamic_k", {})
    if glft.get("enabled") is not None:
        kwargs["glft_dynamic_k_enabled"] = glft["enabled"]
    if "min_samples" in glft:
        kwargs["glft_dynamic_k_min_samples"] = int(glft["min_samples"])
    # M5: Volume-Sync VPIN
    vvs = yaml_cfg.get("vpin_vol_sync", {})
    if vvs.get("enabled") is not None:
        kwargs["vpin_vol_sync_enabled"] = vvs["enabled"]
    if "bucket_btc" in vvs:
        kwargs["vpin_vol_sync_bucket_btc"] = float(vvs["bucket_btc"])
    if "n_buckets" in vvs:
        kwargs["vpin_vol_sync_n_buckets"] = int(vvs["n_buckets"])

    # ---- 374# Phase 3.1: SAC Sidecar Proportional Boost ----
    sc = yaml_cfg.get("sidecar", {})
    if sc.get("enabled") is not None:
        kwargs["sidecar_enabled"] = bool(sc["enabled"])
    if "max_boost_bps" in sc:
        kwargs["sidecar_max_boost_bps"] = float(sc["max_boost_bps"])
    if "dead_zone" in sc:
        kwargs["sidecar_dead_zone"] = float(sc["dead_zone"])
    if "shaping" in sc:
        kwargs["sidecar_shaping"] = str(sc["shaping"])
    if "use_v2" in sc:
        kwargs["sidecar_use_v2"] = bool(sc["use_v2"])

    return _FillTestConfig(**kwargs)
