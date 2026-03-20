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

from scripts.v460.lib.cross_venue_lead_lag import build_cross_venue_fill_fields
from scripts.v460.lib.fill_config import (
    PnlMeasurement as _PnlMeasurement,
)

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig
    from ztb.metrics.fill_quality import FillRecord

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
        }

    def _build_fill_market_fields(
        self,
        *,
        side: str,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        reprice_count: int,
        reprice_drift_bps: float | None,
        sg_skipped: bool,
        sg_score: float,
        sg_reason: str,
        sg_model_used: str,
        sg_as_prob: float | None,
        sg_threshold_used: float | None,
        sg_hour_offset: float | None,
        sg_velocity_bps: float | None,
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
        sidecar_offset_bps: float | None = None,
        sidecar_bias: float | None = None,
        # 487# P0: sidecar attribution 可観測性
        sidecar_confidence: float | None = None,
        sidecar_model_version: str | None = None,
        sidecar_signal_status: str | None = None,
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
            "spread_bps": self._compute_fill_spread_bps(
                spread_at_order=spread_at_order,
                mid_at_fill=mid_at_fill,
            ),
            "effective_offset_used": effective_offset_ratio,
            "skip_gate_skipped": sg_skipped,
            "skip_gate_score": sg_score,
            "skip_gate_reason": sg_reason,
            "skip_gate_model_used": sg_model_used,
            "skip_gate_as_prob": sg_as_prob,
            "skip_gate_threshold_used": sg_threshold_used,
            "skip_gate_hour_offset": sg_hour_offset,
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
            # 372# F1: SAC Sidecar offset 記録
            "sidecar_offset_bps": sidecar_offset_bps,
            "sidecar_bias": sidecar_bias,
            # 487# P0: sidecar attribution 可観測性
            "sidecar_confidence": sidecar_confidence,
            "sidecar_model_version": sidecar_model_version,
            "sidecar_signal_status": sidecar_signal_status,
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
        fields = build_cross_venue_fill_fields(
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
        cancel_failed_likely_filled: bool,
        pnl: _PnlMeasurement,
        sg_skipped: bool,
        sg_score: float,
        sg_reason: str,
        sg_model_used: str,
        sg_as_prob: float | None,
        sg_threshold_used: float | None,
        sg_hour_offset: float | None,
        sg_velocity_bps: float | None,
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
        sidecar_offset_bps: float | None = None,
        sidecar_bias: float | None = None,
        # 487# P0: sidecar attribution 可観測性
        sidecar_confidence: float | None = None,
        sidecar_model_version: str | None = None,
        sidecar_signal_status: str | None = None,
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
        }
        payload.update(
            self._build_fill_measurement_fields(
                fill_price=fill_price,
                filled=filled,
                queue_wait=queue_wait,
                cancel_reason_poll=cancel_reason_poll,
                effective_timeout=effective_timeout,
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
                sg_score=sg_score,
                sg_reason=sg_reason,
                sg_model_used=sg_model_used,
                sg_as_prob=sg_as_prob,
                sg_threshold_used=sg_threshold_used,
                sg_hour_offset=sg_hour_offset,
                sg_velocity_bps=sg_velocity_bps,
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
                sidecar_offset_bps=sidecar_offset_bps,
                sidecar_bias=sidecar_bias,
                sidecar_confidence=sidecar_confidence,
                sidecar_model_version=sidecar_model_version,
                sidecar_signal_status=sidecar_signal_status,
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
        skip_gate_reason: str,
        ev_offset_applied: bool,
    ) -> str:
        """292# P0: FillRecord.decision_path を一元導出する."""
        if ev_score_pretrade is None:
            return "primary_only"
        if "emergency_skip" in skip_gate_reason:
            return "ev_emergency_skip"
        return "ev_offset" if ev_offset_applied else "ev_no_change"
