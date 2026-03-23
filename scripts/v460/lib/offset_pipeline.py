"""460# Mixin: OffsetPipelineMixin -- 発注前 offset 乗数チェーン + lot スケール.

fill_cycle_executor.py からの God Object 分割 (323# 分割計画)。
9 段の offset adjustment pipeline と lot スケールヘルパーを担当。

WARNING -- AI Coding Agent / 人間開発者への注意:
    このファイルは Mixin クラスであり、単独でインスタンス化しないこと。
    FillTestRunner.__init__ で生成される属性に依存する。
    責務: offset pipeline 実行 (9 段乗数 + final clamp) + lot スケール
    FillRecord 構築 / SkipGate 評価 / 監視ロジックを追加しないこと。
"""

from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.fill_config import compute_ev_offset_multiplier
from scripts.v460.lib.hour_rules import current_utc_hour
from scripts.v460.lib.macro_regime import MacroTrend
from scripts.v460.lib.pre_order_adjustments import PreOrderAdjustmentsMixin
from ztb.trading.pricing.offset_ceiling import clamp_offset_ratio_to_ceiling

if TYPE_CHECKING:
    from ztb.metrics.fill_quality import FillRecord

logger = logging.getLogger(__name__)


@dataclass
class OffsetPipelineResult:
    """460# offset pipeline の出力をまとめるデータクラス."""

    order_price: float
    effective_offset_ratio: float
    ev_offset_applied: bool
    ev_score_pretrade: float | None
    ev_offset_mult_applied: float | None
    macro_boost_applied: bool
    execution_pre_clamp_offset: float | None
    executor_offset_stages_json: str | None
    # 573# eDRC テレメトリ
    execution_sigma: float | None = None
    execution_adverse_ofi: float | None = None
    execution_additive_enabled: bool | None = None
    early_return_record: FillRecord | None = None


class OffsetPipelineMixin(PreOrderAdjustmentsMixin):
    """Offset pipeline + lot scale ヘルパー (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: offset 乗数チェーン (EV/Vel/Trend/Tox/VG/Macro/Alert/Sidecar/Clamp)
      OK: lot スケール適用
      NG: FillRecord 構築, SkipGate, 監視, ループ制御
    MAX LINES: 360
    ────────────────────────────────────────────────────
    460# fill_cycle_executor.py からの抽出
    """

    def _apply_offset_pipeline(
        self,
        *,
        side: str,
        order_price: float,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        sg_ev_score: float | None,
        sg_velocity_offset_mult: float | None,
        sg_velocity_bps: float | None,
        trending_offset_mult: float | None,
        toxicity_offset_mult: float,
        sidecar_offset_bps: float,
        cycle_id: str,
    ) -> OffsetPipelineResult:
        """460# Offset adjustment pipeline — RMS 加法結合モデル (刷新版).

        SkipGate 通過後のリスク因子を RMS で統合し、幾何学的爆発を抑制する。
        """
        additive_enabled = self.config.experimental_additive_pipeline
        base_offset_ratio = effective_offset_ratio # 通常 0.05
        
        if not additive_enabled:
            # ── 既存の乗算チェーン (Legacy Mode) ──────────────────
            # (互換性のための既存ロジック呼び出し、または旧実装の維持)
            # ※ 今回は簡略化のため、Additive への完全移行を主眼に置くが、
            # 既存コードを if 分岐で包む形で実装する。
            return self._apply_offset_pipeline_multiplicative(
                side=side, order_price=order_price, spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio, sg_ev_score=sg_ev_score,
                sg_velocity_offset_mult=sg_velocity_offset_mult, sg_velocity_bps=sg_velocity_bps,
                trending_offset_mult=trending_offset_mult, toxicity_offset_mult=toxicity_offset_mult,
                sidecar_offset_bps=sidecar_offset_bps, cycle_id=cycle_id
            )

        # ── 真の加法パイプライン (True Additive / RMS Mode) ──
        import math
        deltas: list[float] = []
        _exec_stages: dict[str, float | None] = {}

        # Stage 1: EV Score (193#)
        if sg_ev_score is not None and self.config.skip_gate_ev_as_offset_enabled:
            ev_mult = compute_ev_offset_multiplier(
                ev_score=sg_ev_score, sensitivity=self.config.skip_gate_ev_offset_sensitivity,
                min_mult=self.config.skip_gate_ev_offset_min_mult, max_mult=self.config.skip_gate_ev_offset_max_mult,
                warning_threshold=self.config.skip_gate_ev_warning_threshold, warning_factor=self.config.skip_gate_ev_warning_offset_factor
            )
            # 加法的増分への変換: (mult - 1.0) * base
            d_ev = (ev_mult - 1.0) * base_offset_ratio
            deltas.append(d_ev)
            _exec_stages["ev"] = ev_mult

        # Stage 2: Velocity (195#)
        if sg_velocity_offset_mult is not None:
            d_vel = (sg_velocity_offset_mult - 1.0) * base_offset_ratio
            deltas.append(d_vel)
            _exec_stages["velocity"] = sg_velocity_offset_mult

        # Stage 3: Trending (196#)
        if side == "sell" and trending_offset_mult is not None:
            d_trend = (trending_offset_mult - 1.0) * base_offset_ratio
            deltas.append(d_trend)
            _exec_stages["trending"] = trending_offset_mult

        # Stage 4: Toxicity (240#)
        if toxicity_offset_mult > 1.0:
            d_tox = (toxicity_offset_mult - 1.0) * base_offset_ratio
            deltas.append(d_tox)
            _exec_stages["toxicity"] = toxicity_offset_mult

        # Stage 5: VG Sell Supplement (202#)
        if side == "sell" and not self._maker_price.last_vg_triggered and sg_velocity_bps is not None:
            if abs(sg_velocity_bps) > self.config.volatility_guard_velocity_threshold_bps:
                vg_boost = self.config.volatility_guard_offset_boost_factor
                d_vg = (vg_boost - 1.0) * base_offset_ratio
                deltas.append(d_vg)
                _exec_stages["vg_supp"] = vg_boost

        # Stage 6: Macro (458#)
        # ... (Macro ロジックの加法化、簡略化して記載)
        _lt = self._last_macro_trend
        if _lt is not None:
            m_mult = 1.0
            if side == "sell":
                if _lt == MacroTrend.STRONG_UP.value: m_mult = self.config.macro_sell_boost_strong_up
            elif side == "buy":
                if _lt == MacroTrend.STRONG_DOWN.value: m_mult = self.config.macro_buy_boost_strong_down
            if m_mult > 1.0:
                deltas.append((m_mult - 1.0) * base_offset_ratio)
                _exec_stages["macro"] = m_mult

        # RMS 統合
        rms_delta = math.sqrt(sum(d**2 for d in deltas)) if deltas else 0.0
        new_offset_ratio = base_offset_ratio + rms_delta

        # Final Clamp & eDRC
        _execution_sigma = self._maker_price.last_sigma
        _execution_adverse_ofi = self._maker_price.get_adverse_ofi(side)
        
        # ロバスト入力の使用
        _robust_sigma, _robust_ofi = self._maker_price.get_robust_inputs(side)
        _sigma_bps = _robust_sigma * 10_000
        _adverse_ofi = _robust_ofi

        _fc_ceil = self.config.resolve_offset_ceiling(
            side, utc_hour=current_utc_hour(), sigma=_sigma_bps, adverse_ofi=_adverse_ofi
        )
        
        _ceiling = clamp_offset_ratio_to_ceiling(new_offset_ratio, _fc_ceil)
        _execution_pre_clamp_offset = new_offset_ratio if _ceiling.clamped else None
        final_offset_ratio = _ceiling.updated_ratio

        # 価格の最終決定
        if final_offset_ratio != effective_offset_ratio and spread_at_order:
            order_price = self._recalc_price_with_new_offset(
                side, order_price, spread_at_order, effective_offset_ratio, final_offset_ratio
            )

        return OffsetPipelineResult(
            order_price=order_price, effective_offset_ratio=final_offset_ratio,
            ev_offset_applied="ev" in _exec_stages, ev_score_pretrade=sg_ev_score,
            ev_offset_mult_applied=_exec_stages.get("ev"), macro_boost_applied="macro" in _exec_stages,
            execution_pre_clamp_offset=_execution_pre_clamp_offset,
            executor_offset_stages_json=_json.dumps(_exec_stages, separators=(",", ":")),
            execution_sigma=_execution_sigma, execution_adverse_ofi=_execution_adverse_ofi,
            execution_additive_enabled=True
        )

    def _apply_offset_pipeline_multiplicative(
        self,
        *,
        side: str,
        order_price: float,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        sg_ev_score: float | None,
        sg_velocity_offset_mult: float | None,
        sg_velocity_bps: float | None,
        trending_offset_mult: float | None,
        toxicity_offset_mult: float,
        sidecar_offset_bps: float,
        cycle_id: str,
    ) -> OffsetPipelineResult:
        """460# 旧来の乗算型パイプライン (Multiplicative Chain)."""
        _ev_offset_applied = False
        _ev_score_pretrade: float | None = sg_ev_score
        _ev_offset_mult_applied: float | None = None
        if (
            sg_ev_score is not None
            and self.config.skip_gate_ev_as_offset_enabled
            and spread_at_order is not None
            and spread_at_order > 0
            and order_price > 0
        ):
            _ev_s = sg_ev_score
            _ev_mult = compute_ev_offset_multiplier(
                ev_score=_ev_s,
                sensitivity=self.config.skip_gate_ev_offset_sensitivity,
                min_mult=self.config.skip_gate_ev_offset_min_mult,
                max_mult=self.config.skip_gate_ev_offset_max_mult,
                warning_threshold=self.config.skip_gate_ev_warning_threshold,
                warning_factor=self.config.skip_gate_ev_warning_offset_factor,
            )
            order_price, effective_offset_ratio, _applied_mult, _delta = self._apply_offset_multiplier(
                side=side, order_price=order_price, spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio, offset_mult=_ev_mult,
                aggressive_when_multiplier_gt_one=True,
            )
            if _applied_mult is not None and _delta is not None:
                _ev_offset_applied = True
                _ev_offset_mult_applied = _applied_mult
            else:
                _ev_offset_mult_applied = _ev_mult

        _vel_offset_applied = False
        order_price, effective_offset_ratio, _vel_mult, _delta = self._apply_offset_multiplier(
            side=side, order_price=order_price, spread_at_order=spread_at_order,
            effective_offset_ratio=effective_offset_ratio, offset_mult=sg_velocity_offset_mult,
        )
        if _vel_mult is not None and _delta is not None:
            _vel_offset_applied = True

        order_price, effective_offset_ratio, _trend_mult, _delta = self._apply_offset_multiplier(
            side=side, order_price=order_price, spread_at_order=spread_at_order,
            effective_offset_ratio=effective_offset_ratio,
            offset_mult=trending_offset_mult if side == "sell" else None,
        )

        _tox_mult = None
        if toxicity_offset_mult > 1.0:
            order_price, effective_offset_ratio, _tox_mult, _tox_delta = self._apply_offset_multiplier(
                side=side, order_price=order_price, spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio, offset_mult=toxicity_offset_mult,
            )

        _vg_supp_mult = None
        if (
            side == "sell" and not self._maker_price.last_vg_triggered
            and sg_velocity_bps is not None
            and abs(sg_velocity_bps) > self.config.volatility_guard_velocity_threshold_bps
            and not _vel_offset_applied
        ):
            vg_boost = self.config.volatility_guard_offset_boost_factor
            order_price, effective_offset_ratio, _vg_supp_mult, _vg_supp_delta = self._apply_offset_multiplier(
                side=side, order_price=order_price, spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio, offset_mult=vg_boost,
            )

        _macro_offset_mult = None
        _macro_boost_applied = False
        _lt = self._last_macro_trend
        if _lt is not None:
            _m_mult = 1.0
            if side == "sell":
                if _lt == MacroTrend.STRONG_UP.value: _m_mult = self.config.macro_sell_boost_strong_up
            elif side == "buy":
                if _lt == MacroTrend.STRONG_DOWN.value: _m_mult = self.config.macro_buy_boost_strong_down
            if _m_mult > 1.0:
                order_price, effective_offset_ratio, _macro_offset_mult, _macro_delta = self._apply_offset_multiplier(
                    side=side, order_price=order_price, spread_at_order=spread_at_order,
                    effective_offset_ratio=effective_offset_ratio, offset_mult=_m_mult,
                )
                if _macro_offset_mult is not None and _macro_delta is not None:
                    _macro_boost_applied = True

        _alert_om = self._alert_offset_mult
        _a_mult = None
        if _alert_om != 1.0:
            order_price, effective_offset_ratio, _a_mult, _a_delta = self._apply_offset_multiplier(
                side=side, order_price=order_price, spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio, offset_mult=_alert_om,
            )

        if sidecar_offset_bps != 0.0 and order_price > 0:
            _sidecar_delta = round(sidecar_offset_bps / 10000.0 * order_price)
            order_price = round(order_price + _sidecar_delta) if side == "buy" else round(order_price - _sidecar_delta)

        _exec_stages = {"ev": _ev_offset_mult_applied, "velocity": _vel_mult if _vel_offset_applied else None,
                        "trending": _trend_mult, "toxicity": _tox_mult, "vg_supp": _vg_supp_mult, "alert": _a_mult}
        
        _execution_sigma = self._maker_price.last_sigma
        _execution_adverse_ofi = self._maker_price.get_adverse_ofi(side)
        
        if self.config.execution_final_clamp_enabled:
            _fc_ceil = self.config.resolve_offset_ceiling(side, utc_hour=current_utc_hour(), sigma=_execution_sigma*10000, adverse_ofi=_execution_adverse_ofi)
            _ceiling = clamp_offset_ratio_to_ceiling(effective_offset_ratio, _fc_ceil)
            effective_offset_ratio = _ceiling.updated_ratio

        return OffsetPipelineResult(
            order_price=order_price, effective_offset_ratio=effective_offset_ratio,
            ev_offset_applied=_ev_offset_applied, ev_score_pretrade=_ev_score_pretrade,
            ev_offset_mult_applied=_ev_offset_mult_applied, macro_boost_applied=_macro_boost_applied,
            execution_pre_clamp_offset=None, executor_offset_stages_json=_json.dumps(_exec_stages),
            execution_sigma=_execution_sigma, execution_adverse_ofi=_execution_adverse_ofi,
            execution_additive_enabled=False
        )

    @staticmethod
    def _scale_lot(
        lot: float, scale: float, min_lot: float, tag: str, *, warn: bool = False,
    ) -> float:
        """460# Lot に乗数を適用 (min_lot ガード + ログ)."""
        pre = lot
        lot = max(min_lot, lot * scale)
        (logger.warning if warn else logger.info)(
            f"[{tag}] lot_scale={scale:.2f}: {pre:.6f} → {lot:.6f}"
        )
        return lot
