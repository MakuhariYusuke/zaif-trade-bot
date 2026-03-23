"""460# Mixin: OffsetPipelineMixin -- 発注前 offset 乗数チェーン + lot スケール.

fill_cycle_executor.py からの God Object 分割 (323# 分割計画)。
9 段の offset adjustment pipeline と lot スケールヘルパーを担当。

WARNING -- AI Coding Agent / 人間開発者への注意:
    このファイルは Mixin クラスであり、単独でインスタンス化しないこと。
    FillTestRunner.__init__ で生成される属性に依存する。
    責務: offset pipeline 実行 (9 段乗数 + final clamp) + lot スケール
    FillRecord 構築 / SkipGate 評価 / 監視ロジックを追加しないこと。

582# Task 2.2: Toxicity & Liquidity Buffer Separation
    experimental_additive_pipeline=True 時、9 段乗算を RMS 加法モデルへ切替。
    Toxicity (velocity/trending/toxicity/vg_supp/alert) と
    Liquidity (ev/macro) を独立 RMS バッファとして結合。
"""

from __future__ import annotations

import json as _json
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.fill_config import compute_ev_offset_multiplier
from scripts.v460.lib.hour_rules import current_utc_hour
from scripts.v460.lib.macro_regime import MacroTrend
from scripts.v460.lib.multiplicative_pipeline import MultiplicativePipelineMixin
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
    early_return_record: FillRecord | None = None


class OffsetPipelineMixin(MultiplicativePipelineMixin):
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
        """582# A/B dispatcher: experimental flag に応じて加法 or 乗法を呼び分け."""
        if self.config.experimental_additive_pipeline:
            return self._apply_offset_pipeline_additive(
                side=side,
                order_price=order_price,
                spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio,
                sg_ev_score=sg_ev_score,
                sg_velocity_offset_mult=sg_velocity_offset_mult,
                sg_velocity_bps=sg_velocity_bps,
                trending_offset_mult=trending_offset_mult,
                toxicity_offset_mult=toxicity_offset_mult,
                sidecar_offset_bps=sidecar_offset_bps,
                cycle_id=cycle_id,
            )
        return self._apply_offset_pipeline_multiplicative(
            side=side,
            order_price=order_price,
            spread_at_order=spread_at_order,
            effective_offset_ratio=effective_offset_ratio,
            sg_ev_score=sg_ev_score,
            sg_velocity_offset_mult=sg_velocity_offset_mult,
            sg_velocity_bps=sg_velocity_bps,
            trending_offset_mult=trending_offset_mult,
            toxicity_offset_mult=toxicity_offset_mult,
            sidecar_offset_bps=sidecar_offset_bps,
            cycle_id=cycle_id,
        )

    # ── 582# True Additive Pipeline (RMS Toxicity/Liquidity Split) ──

    def _apply_offset_pipeline_additive(
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
        """582# RMS 加法 offset pipeline — Toxicity / Liquidity バッファ分離.

        各 multiplier の増分 (m - 1.0) × base_ratio を ΔR_i とし、
        Toxicity 群と Liquidity 群それぞれで RMS 結合する。
        final_offset = base + sqrt(Σ tox_ΔR²) + sqrt(Σ liq_ΔR²)
        """
        base_ratio = effective_offset_ratio

        # ── 1. 各ステージの multiplier を収集 ──
        _ev_offset_applied = False
        _ev_score_pretrade: float | None = sg_ev_score
        _ev_offset_mult_applied: float | None = None
        _macro_boost_applied = False

        _vel_mult: float | None = None
        _trend_mult: float | None = None
        _tox_mult: float | None = None
        _vg_supp_mult: float | None = None
        _a_mult: float | None = None
        _macro_m_mult: float | None = None

        # Toxicity deltas: velocity, trending, toxicity, vg_supp, alert
        tox_deltas: list[float] = []
        # Liquidity deltas: ev, macro
        liq_deltas: list[float] = []

        # 193# EV → Liquidity buffer
        if (
            sg_ev_score is not None
            and self.config.skip_gate_ev_as_offset_enabled
            and spread_at_order is not None
            and spread_at_order > 0
            and order_price > 0
        ):
            _ev_mult = compute_ev_offset_multiplier(
                ev_score=sg_ev_score,
                sensitivity=self.config.skip_gate_ev_offset_sensitivity,
                min_mult=self.config.skip_gate_ev_offset_min_mult,
                max_mult=self.config.skip_gate_ev_offset_max_mult,
                warning_threshold=self.config.skip_gate_ev_warning_threshold,
                warning_factor=self.config.skip_gate_ev_warning_offset_factor,
            )
            _ev_offset_mult_applied = _ev_mult
            if _ev_mult > 1.0:
                _ev_offset_applied = True
                liq_deltas.append(base_ratio * (_ev_mult - 1.0))
        else:
            _ev_offset_mult_applied = None

        # 195# Velocity → Toxicity buffer
        if sg_velocity_offset_mult is not None and sg_velocity_offset_mult > 1.0:
            _vel_mult = sg_velocity_offset_mult
            tox_deltas.append(base_ratio * (_vel_mult - 1.0))

        # 196# Trending (sell only) → Toxicity buffer
        if side == "sell" and trending_offset_mult is not None and trending_offset_mult > 1.0:
            _trend_mult = trending_offset_mult
            tox_deltas.append(base_ratio * (_trend_mult - 1.0))

        # 240# Toxicity Budget → Toxicity buffer
        if toxicity_offset_mult > 1.0:
            _tox_mult = toxicity_offset_mult
            tox_deltas.append(base_ratio * (_tox_mult - 1.0))

        # 202# VG sell supplement → Toxicity buffer
        _vel_already = _vel_mult is not None
        if (
            side == "sell"
            and not self._maker_price.last_vg_triggered
            and sg_velocity_bps is not None
            and abs(sg_velocity_bps) > self.config.volatility_guard_velocity_threshold_bps
            and not _vel_already
        ):
            _vg_supp_mult = self.config.volatility_guard_offset_boost_factor
            if _vg_supp_mult > 1.0:
                tox_deltas.append(base_ratio * (_vg_supp_mult - 1.0))

        # 458# Macro → Liquidity buffer
        _lt = self._last_macro_trend
        if _lt is not None:
            _m_mult = 1.0
            if side == "sell":
                if _lt == MacroTrend.STRONG_UP.value:
                    _m_mult = self.config.macro_sell_boost_strong_up
                elif _lt == MacroTrend.WEAK_UP.value:
                    _m_mult = self.config.macro_sell_boost_weak_up
            elif side == "buy":
                if _lt == MacroTrend.STRONG_DOWN.value:
                    _m_mult = self.config.macro_buy_boost_strong_down
                elif _lt == MacroTrend.WEAK_DOWN.value:
                    _m_mult = self.config.macro_buy_boost_weak_down
            if _m_mult > 1.0:
                _macro_boost_applied = True
                _macro_m_mult = _m_mult
                liq_deltas.append(base_ratio * (_m_mult - 1.0))

        # 215# Alert → Toxicity buffer
        _alert_om = self._alert_offset_mult
        if _alert_om > 1.0:
            _a_mult = _alert_om
            tox_deltas.append(base_ratio * (_a_mult - 1.0))

        # ── 2. RMS 結合 ──
        tox_rms = math.sqrt(sum(d * d for d in tox_deltas)) if tox_deltas else 0.0
        liq_rms = math.sqrt(sum(d * d for d in liq_deltas)) if liq_deltas else 0.0
        effective_offset_ratio = base_ratio + tox_rms + liq_rms

        # ── 3. Sidecar (bps 加算、RMS 外) ──
        if sidecar_offset_bps != 0.0 and order_price > 0:
            _sidecar_delta = round(sidecar_offset_bps / 10000.0 * order_price)
            if side == "buy":
                order_price = round(order_price + _sidecar_delta)
            else:
                order_price = round(order_price - _sidecar_delta)

        # ── 4. Price 再計算 (新 offset_ratio に合わせて) ──
        if spread_at_order is not None and spread_at_order > 0:
            order_price = self._recalc_price_with_new_offset(
                side, order_price, spread_at_order, base_ratio, effective_offset_ratio,
            )

        # ── 5. Stages JSON (tox/liq バッファ値を含む) ──
        _exec_stages: dict[str, float | None] = {
            "ev": _ev_offset_mult_applied,
            "velocity": _vel_mult,
            "trending": _trend_mult,
            "toxicity": _tox_mult,
            "vg_supp": _vg_supp_mult,
            "macro": _macro_m_mult,
            "alert": _a_mult,
            "tox_buffer": round(tox_rms, 6),
            "liq_buffer": round(liq_rms, 6),
        }
        _executor_offset_stages_json = _json.dumps(
            _exec_stages, separators=(",", ":"),
        )

        # ── 6. Final Clamp (共通) ──
        _execution_pre_clamp_offset: float | None = None
        if self.config.execution_final_clamp_enabled:
            _fc_ceil = self.config.resolve_offset_ceiling(
                side,
                utc_hour=current_utc_hour(),
                sigma=self._maker_price.last_sigma,
                adverse_ofi=self._maker_price.get_adverse_ofi(side),
            )
            _ceiling = clamp_offset_ratio_to_ceiling(
                effective_offset_ratio=effective_offset_ratio,
                ceiling_ratio=_fc_ceil,
            )
            if _ceiling.clamped:
                _execution_pre_clamp_offset = effective_offset_ratio
                _hs_mult = self.config.execution_final_clamp_hard_skip_mult
                if _hs_mult > 0 and effective_offset_ratio > _fc_ceil * _hs_mult:
                    logger.warning(
                        "[582# additive_clamp] HARD SKIP: %s "
                        "pre_clamp=%.4f > ceil(%.4f)×%.1f",
                        side, effective_offset_ratio, _fc_ceil, _hs_mult,
                    )
                    return OffsetPipelineResult(
                        order_price=order_price,
                        effective_offset_ratio=effective_offset_ratio,
                        ev_offset_applied=_ev_offset_applied,
                        ev_score_pretrade=_ev_score_pretrade,
                        ev_offset_mult_applied=_ev_offset_mult_applied,
                        macro_boost_applied=_macro_boost_applied,
                        execution_pre_clamp_offset=_execution_pre_clamp_offset,
                        executor_offset_stages_json=_executor_offset_stages_json,
                        early_return_record=self._make_cycle_skip_record(
                            side=side,
                            cancel_reason=CR.FINAL_CLAMP_HARD_SKIP,
                            cycle_id=cycle_id,
                            order_price=order_price,
                            spread_at_order=spread_at_order,
                            spread_offset_ratio=effective_offset_ratio,
                        ),
                    )
                # normal clamp
                if spread_at_order is not None and spread_at_order > 0:
                    order_price = self._recalc_price_with_new_offset(
                        side, order_price, spread_at_order,
                        effective_offset_ratio, _fc_ceil,
                    )
                effective_offset_ratio = _ceiling.updated_ratio

        logger.info(
            "[582# additive] %s: base=%.4f tox_rms=%.4f liq_rms=%.4f "
            "→ final=%.4f",
            side, base_ratio, tox_rms, liq_rms, effective_offset_ratio,
        )
        return OffsetPipelineResult(
            order_price=order_price,
            effective_offset_ratio=effective_offset_ratio,
            ev_offset_applied=_ev_offset_applied,
            ev_score_pretrade=_ev_score_pretrade,
            ev_offset_mult_applied=_ev_offset_mult_applied,
            macro_boost_applied=_macro_boost_applied,
            execution_pre_clamp_offset=_execution_pre_clamp_offset,
            executor_offset_stages_json=_executor_offset_stages_json,
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
