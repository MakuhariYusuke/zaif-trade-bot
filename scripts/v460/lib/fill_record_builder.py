"""323# Mixin: FillRecordBuilderMixin -- FillRecord 構築ヘルパー.

fill_cycle_executor.py からの God Object 分割 (323# 分割計画)。
FillRecord のフィールド構築・組立・EV 計算・decision path 導出を担当。

WARNING -- AI Coding Agent / 人間開発者への注意:
    このファイルは Mixin クラスであり、単独でインスタンス化しないこと。
    FillTestRunner.__init__ で生成される属性に依存する。
    責務: FillRecord 構築のみ。
    run_single_cycle / ループ制御ロジックを追加しないこと。
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from scripts.v460.lib.constants import BPS_FACTOR as _BPS_FACTOR
from scripts.v460.lib.cross_venue_lead_lag import build_cross_venue_fill_fields
from scripts.v460.lib.fill_config import (
    PnlMeasurement as _PnlMeasurement,
)

if TYPE_CHECKING:
    from scripts.v460.lib.cycle_strategy import DefaultCycleStrategy
    from scripts.v460.lib.fill_config import FillTestConfig
    from scripts.v460.lib.maker_price import MakerPriceCalculator
    from ztb.metrics.fill_quality import FillRecord
    from ztb.trading.risk.fast_fill_defense import FastFillDefense

logger = logging.getLogger(__name__)


class FillRecordBuilderMixin:
    """FillRecord 構築ヘルパー (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: FillRecord フィールド構築, EV 加重計算, decision path 導出
      NG: 発注, 監視, ループ制御, SkipGate 評価
    MAX LINES: 450
    ────────────────────────────────────────────────────
    323# fill_cycle_executor.py からの抽出
    """

    config: FillTestConfig
    _maker_price: MakerPriceCalculator
    _fast_fill_defense: FastFillDefense
    _cycle_strategy: DefaultCycleStrategy | None = None
    _run_id: str = ""
    _git_sha: str = ""
    _config_hash: str = ""
    _BPS_FACTOR: float = _BPS_FACTOR

    def _resolve_fill_cancel_reason(
        self,
        *,
        filled: bool,
        queue_wait: float,
        cancel_reason_poll: str | None,
        effective_timeout: float | None,
    ) -> str | None:
        """約定結果に応じた cancel_reason を一元解決."""
        if cancel_reason_poll:
            return cancel_reason_poll
        if filled:
            return None
        timeout_limit = effective_timeout or self.config.order_timeout_sec
        return "timeout" if queue_wait >= timeout_limit else "unknown"

    def _compute_fill_spread_bps(
        self,
        *,
        spread_at_order: float | None,
        mid_at_fill: float | None,
    ) -> float | None:
        """FillRecord 用 spread_bps を安全に算出."""
        if spread_at_order is None or mid_at_fill is None or mid_at_fill <= 0:
            return None
        return spread_at_order / mid_at_fill * self._BPS_FACTOR

    def _build_fill_measurement_fields(
        self,
        *,
        fill_price: float | None,
        filled: bool,
        queue_wait: float,
        cancel_reason_poll: str | None,
        effective_timeout: float | None,
        timeout_reason: str | None,
        pnl: _PnlMeasurement,
    ) -> dict[str, object]:
        """FillRecord の約定/計測系フィールドを構築."""
        return {
            "fill_price": fill_price,
            "filled": filled,
            "cancelled": not filled,
            "queue_wait_sec": queue_wait,
            "mid_at_fill": pnl.mid_at_fill,
            "mid_30s_after": pnl.mid_30s_after,
            "mid_60s_after": pnl.mid_60s_after,
            "mid_120s_after": pnl.mid_120s_after,
            "post_fill_30s_pnl": pnl.post_fill_pnl,
            "post_fill_60s_pnl": pnl.post_fill_60s_pnl,
            "post_fill_120s_pnl": pnl.post_fill_120s_pnl,
            "spread_capture_bps": pnl.spread_capture_bps,
            "adverse_selection_cost_bps": pnl.adverse_selection_cost_bps,
            "adverse_selected": pnl.adverse_selected,
            "adverse_selected_raw": pnl.adverse_selected_raw,
            "cancel_reason": self._resolve_fill_cancel_reason(
                filled=filled,
                queue_wait=queue_wait,
                cancel_reason_poll=cancel_reason_poll,
                effective_timeout=effective_timeout,
            ),
            "actual_measurement_sec": pnl.actual_measurement_sec if filled else None,
            "early_exit_triggered": pnl.early_exit_triggered if filled else None,
            "pnl_at_exit_bps": pnl.pnl_at_exit_bps if filled else None,
            "timeout_applied_sec": effective_timeout,
            "timeout_reason": timeout_reason,
        }

    def _build_fill_market_fields(
        self,
        *,
        side: str,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        reprice_count: int,
        reprice_drift_bps: float | None,
        sg_skipped: bool | None,
        sg_bypassed: bool | None,
        sg_score: float | None,
        sg_reason: str | None,
        sg_model_used: str | None,
        sg_as_prob: float | None,
        sg_threshold_used: float | None,
        sg_hour_offset: float | None,
        sg_velocity_bps: float | None,
        trend_5s_guard_triggered: bool,
        trend_5s_guard_action: str | None,
        trend_5s_at_order: float | None,
        regime_str: str | None,
        regime_conf: float | None,
        regime_stab: int | None,
        regime_trend_pct: float | None,
        regime_vol_ratio: float | None,
        confidence_factor: float,
        regime_lot: float,
        order_lot: float,
        cancel_failed_likely_filled: bool,
        mid_at_fill: float | None,
        ev_score_pretrade: float | None = None,
        ev_offset_mult_applied: float | None = None,
        decision_path: str | None = None,
        decision_trace_id: str | None = None,
        sidecar_offset_bps: float | None = None,
        sidecar_bias: float | None = None,
        # 487# P0: sidecar attribution 可観測性
        sidecar_confidence: float | None = None,
        sidecar_model_version: str | None = None,
        sidecar_signal_status: str | None = None,
        ppo_sidecar_action: str | None = None,
        ppo_sidecar_confidence: float | None = None,
        ppo_sidecar_action_margin: float | None = None,
        ppo_sidecar_model_version: str | None = None,
        ppo_sidecar_signal_status: str | None = None,
        ppo_sidecar_override_active: bool | None = None,
        sg_budget_regime: str | None = None,
        sg_budget_remaining: int | None = None,
        sg_budget_exhausted: bool | None = None,
    ) -> dict[str, object]:
        """FillRecord の市場観測/実行メタ系フィールドを構築."""
        fields: dict[str, object] = {
            "spread_at_order": spread_at_order,
            "spread_offset_ratio": effective_offset_ratio,
            "regime": regime_str,
            "regime_confidence": regime_conf,
            "regime_stability": regime_stab,
            "regime_trend_pct": regime_trend_pct,
            "regime_volatility_ratio": regime_vol_ratio,
            "orderbook_imbalance": self._maker_price._last_imbalance,
            "bid_depth_total": self._maker_price._last_bid_depth,
            "ask_depth_total": self._maker_price._last_ask_depth,
            "mid_price_trend_5s": self._maker_price._last_mid_trend_bps,
            "trend_5s_guard_triggered": trend_5s_guard_triggered or None,
            "trend_5s_guard_action": trend_5s_guard_action,
            "trend_5s_at_order": trend_5s_at_order,
            "spread_bps": self._compute_fill_spread_bps(
                spread_at_order=spread_at_order,
                mid_at_fill=mid_at_fill,
            ),
            "effective_offset_used": effective_offset_ratio,
            "skip_gate_skipped": sg_skipped,
            "skip_gate_bypassed": sg_bypassed,
            "skip_gate_score": sg_score,
            "skip_gate_reason": sg_reason,
            "skip_gate_model_used": sg_model_used,
            "skip_gate_as_prob": sg_as_prob,
            "skip_gate_threshold_used": sg_threshold_used,
            "skip_gate_hour_offset": sg_hour_offset,
            "skip_gate_budget_regime": sg_budget_regime,
            "skip_gate_budget_remaining": sg_budget_remaining,
            "skip_gate_budget_exhausted": sg_budget_exhausted,
            "reprice_count": reprice_count,
            "reprice_drift_bps": reprice_drift_bps if reprice_count > 0 else None,
            "ffd_boost_active": self._fast_fill_defense.is_boost_active(side),
            "vg_triggered": self._maker_price.last_vg_triggered,
            "vg_velocity_bps": self._maker_price.last_vg_velocity_bps,
            "vg_vpin": self._maker_price.last_vg_vpin,
            "vg_boost_factor": self._maker_price.last_vg_boost_factor,
            "vg_reason": self._maker_price.last_vg_reason,
            "inv_skew_factor": self._maker_price.last_inv_skew_factor,
            "price_velocity_bps": sg_velocity_bps,
            "confidence_lot_factor": (
                confidence_factor if self.config.enable_confidence_lot else None
            ),
            "order_lot_regime": regime_lot,
            "order_lot_effective": order_lot,
            "confidence_lot_mode": (
                self.config.confidence_lot_mode if self.config.enable_confidence_lot else None
            ),
            "ab_test_variant": self.config.ab_test_variant or None,
            "cancel_failed_likely_filled": cancel_failed_likely_filled or None,
            # 292# P0: ev_weighted 可観測性強化 (290#/291# review)
            "ev_score_pretrade": ev_score_pretrade,
            "ev_offset_mult_applied": ev_offset_mult_applied,
            "decision_path": decision_path,
            "decision_trace_id": decision_trace_id,
            # 372# F1: SAC Sidecar offset 記録
            "sidecar_offset_bps": sidecar_offset_bps,
            "sidecar_bias": sidecar_bias,
            # 487# P0: sidecar attribution 可観測性
            "sidecar_confidence": sidecar_confidence,
            "sidecar_model_version": sidecar_model_version,
            "sidecar_signal_status": sidecar_signal_status,
            "ppo_sidecar_action": ppo_sidecar_action,
            "ppo_sidecar_confidence": ppo_sidecar_confidence,
            "ppo_sidecar_action_margin": ppo_sidecar_action_margin,
            "ppo_sidecar_model_version": ppo_sidecar_model_version,
            "ppo_sidecar_signal_status": ppo_sidecar_signal_status,
            "ppo_sidecar_override_active": ppo_sidecar_override_active,
        }
        fields.update(self._build_fill_cross_venue_fields(side=side))
        return fields

    def _build_fill_cross_venue_fields(
        self,
        *,
        side: str,
    ) -> dict[str, object]:
        """Cross-venue lead-lag の観測値を FillRecord 向けに整形する."""
        # 512# getattr → 直接参照 (型安全化)
        enabled = self.config.cross_venue_lead_lag_enabled
        hint = self._maker_price.cross_venue_lead_lag_hint
        vetoed = self._maker_price.cross_venue_lead_lag_vetoed
        fields: dict[str, object] = build_cross_venue_fill_fields(
            enabled=enabled,
            hint=hint,
            side=side,
            vetoed=vetoed,
        )
        # 448# F2: no-op 可視化フィールド追加
        # 512# getattr → 直接参照
        fields["cross_venue_lead_lag_pre_offset"] = self._maker_price._cross_venue_lead_lag_pre_offset
        fields["cross_venue_lead_lag_post_offset"] = self._maker_price._cross_venue_lead_lag_post_offset
        fields["cross_venue_lead_lag_cap_hit"] = self._maker_price._cross_venue_lead_lag_cap_hit
        # 533# veto deadlock 防止: 連続 veto 回数
        fields["cross_venue_lead_lag_veto_consecutive"] = self._maker_price._consecutive_veto_count
        # 642# cv_offset_action: widen/tighten 方向を直接記録
        _pre = self._maker_price._cross_venue_lead_lag_pre_offset
        _post = self._maker_price._cross_venue_lead_lag_post_offset
        if _pre is not None and _post is not None and _pre != _post:
            fields["cv_offset_action"] = "widen" if _post > _pre else "tighten"
        return fields

    def _build_fill_strategy_fields(
        self,
        *,
        post_fill_pnl: float | None,
        post_fill_120s_pnl: float | None,
        regime_str: str | None,
        regime_conf: float | None,
        macro_trend: str | None,
        macro_slope_5m: float | None,
        macro_slope_15m: float | None,
        macro_aligned: bool | None,
        macro_boost_applied: bool | None = None,
    ) -> dict[str, object]:
        """FillRecord の strategy/macro 系フィールドを構築."""
        ev_weighted = self._compute_ev_weighted(
            post_fill_pnl,
            post_fill_120s_pnl,
            w30=self._cycle_strategy.policy.ev_weighted_w30,
            w120=self._cycle_strategy.policy.ev_weighted_w120,
        ) if self._cycle_strategy is not None else self._compute_ev_weighted(
            post_fill_pnl,
            post_fill_120s_pnl,
        )
        return {
            "ev_weighted_pnl": ev_weighted,
            "gated_regime": (
                self._cycle_strategy.gated_regime(regime_str, regime_conf)
                if self._cycle_strategy is not None and regime_str is not None
                else None
            ),
            "effective_cycle_interval": (
                self._cycle_strategy.effective_interval(regime_str)
                if self._cycle_strategy is not None
                else None
            ),
            "macro_trend": macro_trend,
            "macro_slope_5m": macro_slope_5m,
            "macro_slope_15m": macro_slope_15m,
            "macro_aligned": macro_aligned,
            "macro_boost_applied": macro_boost_applied,
        }

    # ------------------------------------------------------------------
    # 181# EV_weighted: pnl30/pnl120 加重平均 (178# §1.3)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_ev_weighted(
        pnl30: float | None,
        pnl120: float | None,
        *,
        w30: float = 0.4,
        w120: float = 0.6,
    ) -> float | None:
        """30s/120s PnL の加重平均を計算.

        pnl120 が None (E3 サンプリング外) の場合は pnl30 単独値を返す。
        """
        if pnl30 is None:
            return None
        if pnl120 is None:
            return pnl30  # 120s 未計測時は 30s 単独
        return w30 * pnl30 + w120 * pnl120

    # ------------------------------------------------------------------
    # 188# FillRecord 構築 (run_single_cycle からの抽出)
    # ------------------------------------------------------------------
    def _build_fill_record(
        self,
        *,
        cycle_id: str,
        t_submit: float,
        side: str,
        order_price: float,
        order_lot: float,
        fill_price: float | None,
        filled: bool,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        queue_wait: float,
        cancel_reason_poll: str | None,
        reprice_count: int,
        reprice_drift_bps: float | None,
        effective_timeout: float | None,
        timeout_reason: str | None,
        cancel_failed_likely_filled: bool,
        pnl: _PnlMeasurement,
        sg_skipped: bool | None,
        sg_bypassed: bool | None,
        sg_score: float | None,
        sg_reason: str | None,
        sg_model_used: str | None,
        sg_as_prob: float | None,
        sg_threshold_used: float | None,
        sg_hour_offset: float | None,
        sg_velocity_bps: float | None,
        trend_5s_guard_triggered: bool,
        trend_5s_guard_action: str | None,
        trend_5s_at_order: float | None,
        regime_str: str | None,
        regime_conf: float | None,
        regime_stab: int | None,
        regime_trend_pct: float | None,
        regime_vol_ratio: float | None,
        confidence_factor: float,
        regime_lot: float,
        macro_trend: str | None = None,
        macro_slope_5m: float | None = None,
        macro_slope_15m: float | None = None,
        macro_aligned: bool | None = None,
        macro_boost_applied: bool | None = None,
        ev_score_pretrade: float | None = None,
        ev_offset_mult_applied: float | None = None,
        decision_path: str | None = None,
        decision_trace_id: str | None = None,
        sidecar_offset_bps: float | None = None,
        sidecar_bias: float | None = None,
        # 487# P0: sidecar attribution 可観測性
        sidecar_confidence: float | None = None,
        sidecar_model_version: str | None = None,
        sidecar_signal_status: str | None = None,
        ppo_sidecar_action: str | None = None,
        ppo_sidecar_confidence: float | None = None,
        ppo_sidecar_action_margin: float | None = None,
        ppo_sidecar_model_version: str | None = None,
        ppo_sidecar_signal_status: str | None = None,
        ppo_sidecar_override_active: bool | None = None,
        queue_depth_ahead: float | None = None,
        queue_fill_prob_est: float | None = None,
        regime_at_order: str | None = None,
        regime_observation_count: int | None = None,
        mid_at_order: float | None = None,
        execution_pre_clamp_offset: float | None = None,
        executor_offset_stages: str | None = None,
        start_git_sha: str | None = None,
        requested_side: str | None = None,
        resolved_side_reason: str | None = None,
        last_executed_side: str | None = None,
        last_attempted_side: str | None = None,
        log_cycle_no: int | None = None,
        # 573# eDRC テレメトリ
        execution_sigma: float | None = None,
        execution_adverse_ofi: float | None = None,
        execution_additive_enabled: bool | None = None,
        # 642# 可観測性
        sg_forced_pass: bool = False,
        sg_side_skip_rate: float | None = None,
        sg_budget_regime: str | None = None,
        sg_budget_remaining: int | None = None,
        sg_budget_exhausted: bool | None = None,
        execution_hard_skip_mult_used: float | None = None,
        balance_jpy_at_order: float | None = None,
        balance_btc_at_order: float | None = None,
    ) -> FillRecord:
        """188# FillRecord を組み立てる.

        run_single_cycle の末尾から抽出。self 経由のセンサー値 +
        サイクル変数を統合して 1 レコードを構築する。
        """
        from ztb.metrics.fill_quality import build_fill_record

        payload: dict[str, object] = {
            "cycle_id": cycle_id,
            "timestamp": t_submit,
            "side": side,
            "order_price": order_price,
            "order_quantity": order_lot,
            "run_id": self._run_id,
            "git_sha": self._git_sha,
            "start_git_sha": getattr(self, "_start_git_sha", None),  # 420# P1
            "config_hash": self._config_hash or None,  # 467# 設定識別子
            "pid": os.getpid(),  # 285# 283# P0-1: Split-Brain 検知用
            # 306# O1: queue position estimation
            "queue_depth_ahead": queue_depth_ahead,
            "queue_fill_prob_est": queue_fill_prob_est,
            # 306# E1: offset stage recording
            "offset_stages": self._maker_price.last_offset_stages,
            # 306# L2: microprice bias
            "microprice_bias_bps": self._maker_price.compute_microprice_bias_bps(),
            # 318# F5-3: none regime 可観測性 (307# F5)
            "regime_at_order": regime_at_order,
            "regime_observation_count": regime_observation_count,
            # 319# S-3: mid_at_order (316# S-3: spread capture 精度向上)
            "mid_at_order": mid_at_order,
            # 421# P0: Execution Final Clamp 発火記録
            "execution_pre_clamp_offset": execution_pre_clamp_offset,
            # 420# P1: Executor Offset Stages / start_git_sha / side 切替可観測性
            "executor_offset_stages": executor_offset_stages,
            "start_git_sha": start_git_sha,
            "requested_side": requested_side,
            "resolved_side_reason": resolved_side_reason,
            "last_executed_side": (
                self._side_selector.last_executed_side
                if last_executed_side is None else last_executed_side
            ),
            "last_attempted_side": (
                self._side_selector.last_attempted_side
                if last_attempted_side is None else last_attempted_side
            ),
            # 533# log_cycle_no: ログ⇔JSONL join key
            "log_cycle_no": log_cycle_no,
            # 573# eDRC テレメトリ
            "execution_sigma": execution_sigma,
            "execution_adverse_ofi": execution_adverse_ofi,
            "execution_additive_enabled": execution_additive_enabled,
            # 642# 可観測性
            "skip_gate_forced_pass": sg_forced_pass,
            "skip_gate_side_skip_rate": sg_side_skip_rate,
            "execution_hard_skip_mult_used": execution_hard_skip_mult_used,
            "balance_jpy_at_order": balance_jpy_at_order,
            "balance_btc_at_order": balance_btc_at_order,
        }
        payload.update(
            self._build_fill_measurement_fields(
                fill_price=fill_price,
                filled=filled,
                queue_wait=queue_wait,
                cancel_reason_poll=cancel_reason_poll,
                effective_timeout=effective_timeout,
                timeout_reason=timeout_reason,
                pnl=pnl,
            )
        )
        payload.update(
            self._build_fill_market_fields(
                side=side,
                spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio,
                reprice_count=reprice_count,
                reprice_drift_bps=reprice_drift_bps,
                sg_skipped=sg_skipped,
                sg_bypassed=sg_bypassed,
                sg_score=sg_score,
                sg_reason=sg_reason,
                sg_model_used=sg_model_used,
                sg_as_prob=sg_as_prob,
                sg_threshold_used=sg_threshold_used,
                sg_hour_offset=sg_hour_offset,
                sg_velocity_bps=sg_velocity_bps,
                trend_5s_guard_triggered=trend_5s_guard_triggered,
                trend_5s_guard_action=trend_5s_guard_action,
                trend_5s_at_order=trend_5s_at_order,
                regime_str=regime_str,
                regime_conf=regime_conf,
                regime_stab=regime_stab,
                regime_trend_pct=regime_trend_pct,
                regime_vol_ratio=regime_vol_ratio,
                confidence_factor=confidence_factor,
                regime_lot=regime_lot,
                order_lot=order_lot,
                cancel_failed_likely_filled=cancel_failed_likely_filled,
                mid_at_fill=pnl.mid_at_fill,
                ev_score_pretrade=ev_score_pretrade,
                ev_offset_mult_applied=ev_offset_mult_applied,
                decision_path=decision_path,
                decision_trace_id=decision_trace_id,
                sidecar_offset_bps=sidecar_offset_bps,
                sidecar_bias=sidecar_bias,
                sidecar_confidence=sidecar_confidence,
                sidecar_model_version=sidecar_model_version,
                sidecar_signal_status=sidecar_signal_status,
                ppo_sidecar_action=ppo_sidecar_action,
                ppo_sidecar_confidence=ppo_sidecar_confidence,
                ppo_sidecar_action_margin=ppo_sidecar_action_margin,
                ppo_sidecar_model_version=ppo_sidecar_model_version,
                ppo_sidecar_signal_status=ppo_sidecar_signal_status,
                ppo_sidecar_override_active=ppo_sidecar_override_active,
                sg_budget_regime=sg_budget_regime,
                sg_budget_remaining=sg_budget_remaining,
                sg_budget_exhausted=sg_budget_exhausted,
            )
        )
        payload.update(
            self._build_fill_strategy_fields(
                post_fill_pnl=pnl.post_fill_pnl,
                post_fill_120s_pnl=pnl.post_fill_120s_pnl,
                regime_str=regime_str,
                regime_conf=regime_conf,
                macro_trend=macro_trend,
                macro_slope_5m=macro_slope_5m,
                macro_slope_15m=macro_slope_15m,
                macro_aligned=macro_aligned,
                macro_boost_applied=macro_boost_applied,
            )
        )
        return build_fill_record(**payload)

    @staticmethod
    def _derive_decision_path(
        *,
        ev_score_pretrade: float | None,
        skip_gate_reason: str | None,
        ev_offset_applied: bool,
    ) -> str:
        """292# P0: FillRecord.decision_path を一元導出する."""
        if ev_score_pretrade is None:
            return "primary_only"
        if skip_gate_reason is not None and "emergency_skip" in skip_gate_reason:
            return "ev_emergency_skip"
        return "ev_offset" if ev_offset_applied else "ev_no_change"
