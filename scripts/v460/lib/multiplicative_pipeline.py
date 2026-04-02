"""583# Mixin: MultiplicativePipelineMixin -- 従来の乗算 offset チェーン.

Prompt 583 Task B:
    `offset_pipeline.py` の行数超過を解消するため、
    `_apply_offset_pipeline_multiplicative()` を分離する。
"""

from __future__ import annotations

import json as _json
import logging
from typing import TYPE_CHECKING

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.fill_config import compute_ev_offset_multiplier
from scripts.v460.lib.hour_rules import current_utc_hour
from scripts.v460.lib.macro_regime import MacroTrend
from scripts.v460.lib.pre_order_adjustments import PreOrderAdjustmentsMixin
from ztb.trading.pricing.offset_ceiling import clamp_offset_ratio_to_ceiling

if TYPE_CHECKING:
    from scripts.v460.lib.offset_pipeline import OffsetPipelineResult

logger = logging.getLogger(__name__)


class MultiplicativePipelineMixin(PreOrderAdjustmentsMixin):
    """従来の乗算 offset pipeline を提供する shared mixin."""

    _offset_pipeline_stage_stats: dict[str, list[int]] = {}
    _offset_pipeline_last_stats_cycle: int = 0

    def _record_offset_stage_stat(self, stage_name: str, *, applied: bool) -> None:
        stats = self._offset_pipeline_stage_stats.setdefault(stage_name, [0, 0])
        stats[0] += 1
        if applied:
            stats[1] += 1

    def _maybe_log_offset_pipeline_stats(self) -> None:
        cycle_count = getattr(self, "_cycle_count", 0)
        if cycle_count <= 0 or cycle_count % 100 != 0:
            return
        if self._offset_pipeline_last_stats_cycle == cycle_count:
            return
        self._offset_pipeline_last_stats_cycle = cycle_count

        def _pair(stage_name: str) -> tuple[int, int]:
            total, applied = self._offset_pipeline_stage_stats.get(stage_name, [0, 0])
            return applied, total

        logger.info(
            "[690# pipeline_stats] stages: "
            "ev=%d/%d vel=%d/%d toxic_veto=%d/%d trend5s=%d/%d astrail=%d/%d macro=%d/%d "
            "tox_model=%d/%d vg_supp=%d/%d alert=%d/%d clamp=%d/%d",
            *_pair("ev"),
            *_pair("velocity"),
            *_pair("toxic_veto"),
            *_pair("trend_5s_guard"),
            *_pair("as_trailing_guard"),
            *_pair("macro"),
            *_pair("toxicity"),
            *_pair("vg_supp"),
            *_pair("alert"),
            *_pair("clamp"),
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
        sg_toxic_veto_offset_mult: float | None = None,  # 657# A-4
        sg_trend_5s_guard_offset_mult: float | None = None,
        sg_as_trailing_offset_mult: float | None = None,
        trending_offset_mult: float | None,
        toxicity_offset_mult: float,
        sidecar_offset_bps: float,
        cycle_id: str,
        decision_trace_id: str | None = None,
    ) -> "OffsetPipelineResult":
        """460# Offset adjustment pipeline -- 9 段の offset 乗数チェーン + final clamp."""
        # Avoid module import cycle at class-definition time.
        from scripts.v460.lib.offset_pipeline import OffsetPipelineResult

        _ev_offset_applied = False
        _ev_score_pretrade: float | None = sg_ev_score
        _ev_offset_mult_applied: float | None = None
        if (
            self.config.offset_ev_stage_enabled
            and
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
                side=side,
                order_price=order_price,
                spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio,
                offset_mult=_ev_mult,
                aggressive_when_multiplier_gt_one=True,
            )
            if _applied_mult is not None and _delta is not None:
                _ev_offset_applied = True
                _ev_offset_mult_applied = _applied_mult
                logger.info(
                    f"[193# ev_offset] {side}: ev_score={_ev_s:.3f} "
                    f"-> offset_mult={_applied_mult:.3f} "
                    f"(delta={_delta:+.0f}JPY, price={order_price:.0f})"
                )
            else:
                _ev_offset_mult_applied = _ev_mult

        _vel_offset_applied = False
        order_price, effective_offset_ratio, _vel_mult, _delta = self._apply_offset_multiplier(
            side=side,
            order_price=order_price,
            spread_at_order=spread_at_order,
            effective_offset_ratio=effective_offset_ratio,
            offset_mult=sg_velocity_offset_mult,
        )
        if _vel_mult is not None and _delta is not None:
            _vel_offset_applied = True
            logger.info(
                f"[195# vel_offset] {side}: velocity={sg_velocity_bps:.2f}bps "
                f"-> offset_mult={_vel_mult:.2f} "
                f"(delta={_delta:+.0f}JPY, price={order_price:.0f})"
            )

        # 657# A-4: toxic_sell_veto offset boost
        _toxic_veto_offset_applied = False
        if sg_toxic_veto_offset_mult is not None and sg_toxic_veto_offset_mult > 1.0:
            order_price, effective_offset_ratio, _tv_mult, _tv_delta = self._apply_offset_multiplier(
                side=side,
                order_price=order_price,
                spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio,
                offset_mult=sg_toxic_veto_offset_mult,
            )
            if _tv_mult is not None and _tv_delta is not None:
                _toxic_veto_offset_applied = True
                logger.info(
                    f"[657# toxic_veto_offset] {side}: "
                    f"offset_mult={_tv_mult:.2f} "
                    f"(delta={_tv_delta:+.0f}JPY, price={order_price:.0f})"
                )

        _trend_5s_guard_mult: float | None = None
        if (
            side == "sell"
            and sg_trend_5s_guard_offset_mult is not None
            and sg_trend_5s_guard_offset_mult > 1.0
        ):
            order_price, effective_offset_ratio, _trend_5s_guard_mult, _tg_delta = self._apply_offset_multiplier(
                side=side,
                order_price=order_price,
                spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio,
                offset_mult=sg_trend_5s_guard_offset_mult,
            )
            if _trend_5s_guard_mult is not None and _tg_delta is not None:
                logger.info(
                    "[684# trend_5s_guard] %s: offset_mult=%.2f (delta=%+.0fJPY, price=%.0f)",
                    side,
                    _trend_5s_guard_mult,
                    _tg_delta,
                    order_price,
                )

        _as_trailing_mult: float | None = None
        if sg_as_trailing_offset_mult is not None and sg_as_trailing_offset_mult > 1.0:
            order_price, effective_offset_ratio, _as_trailing_mult, _as_delta = self._apply_offset_multiplier(
                side=side,
                order_price=order_price,
                spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio,
                offset_mult=sg_as_trailing_offset_mult,
            )
            if _as_trailing_mult is not None and _as_delta is not None:
                logger.info(
                    "[694# as_trailing_guard] %s: offset_mult=%.2f (delta=%+.0fJPY, price=%.0f)",
                    side,
                    _as_trailing_mult,
                    _as_delta,
                    order_price,
                )

        order_price, effective_offset_ratio, _trend_mult, _delta = self._apply_offset_multiplier(
            side=side,
            order_price=order_price,
            spread_at_order=spread_at_order,
            effective_offset_ratio=effective_offset_ratio,
            offset_mult=trending_offset_mult if side == "sell" else None,
        )
        if _trend_mult is not None and _delta is not None:
            logger.info(
                f"[196# trend_offset] sell: trending regime "
                f"-> offset_mult={_trend_mult:.1f} "
                f"(delta={_delta:+.0f}JPY, price={order_price:.0f})"
            )

        _tox_mult: float | None = None
        if self.config.offset_toxicity_stage_enabled and toxicity_offset_mult > 1.0:
            order_price, effective_offset_ratio, _tox_mult, _tox_delta = self._apply_offset_multiplier(
                side=side,
                order_price=order_price,
                spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio,
                offset_mult=toxicity_offset_mult,
            )
            if _tox_mult is not None and _tox_delta is not None:
                logger.info(
                    f"[240# toxicity_offset] {side}: "
                    f"offset_mult={_tox_mult:.2f} "
                    f"(delta={_tox_delta:+.0f}JPY, price={order_price:.0f})"
                )

        _vg_supp_mult: float | None = None
        if (
            self.config.offset_vg_supplement_enabled
            and
            side == "sell"
            and not self._maker_price.last_vg_triggered
            and sg_velocity_bps is not None
            and abs(sg_velocity_bps) > self.config.volatility_guard_velocity_threshold_bps
            and not _vel_offset_applied
        ):
            _vg_supp_boost = self.config.volatility_guard_offset_boost_factor
            order_price, effective_offset_ratio, _vg_supp_mult, _vg_supp_delta = self._apply_offset_multiplier(
                side=side,
                order_price=order_price,
                spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio,
                offset_mult=_vg_supp_boost,
            )
            if _vg_supp_mult is not None and _vg_supp_delta is not None:
                logger.info(
                    f"[202# C] VG sell supplement: velocity_bps="
                    f"{sg_velocity_bps:.1f}bps -> offset_mult={_vg_supp_mult:.2f} "
                    f"(delta={_vg_supp_delta:+.0f}JPY, price={order_price:.0f})"
                )

        _macro_offset_mult: float | None = None
        _macro_boost_applied = False
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
                order_price, effective_offset_ratio, _macro_offset_mult, _macro_delta = self._apply_offset_multiplier(
                    side=side,
                    order_price=order_price,
                    spread_at_order=spread_at_order,
                    effective_offset_ratio=effective_offset_ratio,
                    offset_mult=_m_mult,
                )
                if _macro_offset_mult is not None and _macro_delta is not None:
                    _macro_boost_applied = True
                    logger.info(
                        "[458# macro_boost] %s: macro_trend=%s "
                        "-> offset_mult=%.2f (delta=%+.0fJPY, price=%.0f)",
                        side,
                        _lt,
                        _macro_offset_mult,
                        _macro_delta,
                        order_price,
                    )

        _alert_om = self._alert_offset_mult
        _a_mult: float | None = None
        if _alert_om != 1.0:
            order_price, effective_offset_ratio, _a_mult, _a_delta = self._apply_offset_multiplier(
                side=side,
                order_price=order_price,
                spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio,
                offset_mult=_alert_om,
            )
            if _a_mult is not None and _a_delta is not None:
                logger.warning(
                    f"[215# alert_mode] {side}: offset_mult={_a_mult:.2f} "
                    f"(delta={_a_delta:+.0f}JPY, price={order_price:.0f})"
                )

        _execution_pre_clamp_offset: float | None = None
        _exec_stages: dict[str, float | None] = {
            "ev": _ev_offset_mult_applied,
            "velocity": _vel_mult if _vel_offset_applied else None,
            "toxic_veto": sg_toxic_veto_offset_mult if _toxic_veto_offset_applied else None,  # 657# A-4
            "trend_5s_guard": _trend_5s_guard_mult,
            "as_trailing_guard": _as_trailing_mult,
            "trending": _trend_mult,
            "toxicity": _tox_mult,
            "vg_supp": _vg_supp_mult,
            "alert": _a_mult,
        }
        _executor_offset_stages_json: str | None = None
        if any(v is not None for v in _exec_stages.values()):
            _executor_offset_stages_json = _json.dumps(
                _exec_stages,
                separators=(",", ":"),
            )
        if self.config.execution_final_clamp_enabled:
            _robust_sigma, _robust_ofi = self._maker_price.get_robust_inputs(side)
            _fc_ceil = self.config.resolve_offset_ceiling(
                side,
                utc_hour=current_utc_hour(),
                sigma=_robust_sigma,
                adverse_ofi=_robust_ofi,
            )
            _ceiling = clamp_offset_ratio_to_ceiling(
                effective_offset_ratio=effective_offset_ratio,
                ceiling_ratio=_fc_ceil,
            )
            if _ceiling.clamped:
                _execution_pre_clamp_offset = effective_offset_ratio
                # 641# P1-A: resolve_hard_skip_mult で regime 別 override を参照
                _regime_val: str | None = None
                if hasattr(self, "_current_regime_value"):
                    _regime_val = self._current_regime_value()
                _hs_mult = self.config.resolve_hard_skip_mult(side, _regime_val)
                if _hs_mult > 0 and effective_offset_ratio > _fc_ceil * _hs_mult:
                    logger.warning(
                        f"[421# final_clamp] HARD SKIP: {side} "
                        f"pre_clamp_offset={effective_offset_ratio:.4f} "
                        f"> ceiling({_fc_ceil:.4f})x{_hs_mult:.1f} -- "
                        f"market too extreme, skipping cycle"
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
                        execution_hard_skip_mult_used=_hs_mult,
                        early_return_record=self._make_cycle_skip_record(
                            side=side,
                            cancel_reason=CR.FINAL_CLAMP_HARD_SKIP,
                            cycle_id=cycle_id,
                            decision_trace_id=decision_trace_id,
                            order_price=order_price,
                            spread_at_order=spread_at_order,
                            spread_offset_ratio=effective_offset_ratio,
                        ),
                    )
                if spread_at_order is not None and spread_at_order > 0:
                    order_price = self._recalc_price_with_new_offset(
                        side,
                        order_price,
                        spread_at_order,
                        effective_offset_ratio,
                        _fc_ceil,
                    )
                else:
                    logger.warning(
                        f"[421# final_clamp] {side}: spread unavailable -- "
                        f"ratio clamped but price NOT recalculated"
                    )
                logger.info(
                    f"[421# final_clamp] {side}: offset "
                    f"{effective_offset_ratio:.4f}->{_fc_ceil:.4f} "
                    f"(clamped, price={order_price:.0f})"
                )
                effective_offset_ratio = _ceiling.updated_ratio
                self._record_offset_stage_stat("clamp", applied=True)
            else:
                self._record_offset_stage_stat("clamp", applied=False)
        else:
            self._record_offset_stage_stat("clamp", applied=False)

        self._record_offset_stage_stat("ev", applied=_ev_offset_applied)
        self._record_offset_stage_stat("velocity", applied=_vel_offset_applied)
        self._record_offset_stage_stat("toxic_veto", applied=_toxic_veto_offset_applied)
        self._record_offset_stage_stat(
            "trend_5s_guard",
            applied=_trend_5s_guard_mult is not None,
        )
        self._record_offset_stage_stat(
            "as_trailing_guard",
            applied=_as_trailing_mult is not None,
        )
        self._record_offset_stage_stat("macro", applied=_macro_boost_applied)
        self._record_offset_stage_stat("toxicity", applied=_tox_mult is not None)
        self._record_offset_stage_stat("vg_supp", applied=_vg_supp_mult is not None)
        self._record_offset_stage_stat("alert", applied=_a_mult is not None)
        self._maybe_log_offset_pipeline_stats()

        # 620# sidecar injection: ceiling clamp の後に適用。
        # sidecar は ceiling 制約を尊重した上での微調整として機能する。
        if sidecar_offset_bps != 0.0 and order_price > 0:
            _sidecar_delta = round(sidecar_offset_bps / 10000.0 * order_price)
            if side == "buy":
                order_price = round(order_price + _sidecar_delta)
            else:
                order_price = round(order_price - _sidecar_delta)
            logger.info(
                f"[372# sidecar] {side}: offset={sidecar_offset_bps:+.4f}bps "
                f"-> delta={_sidecar_delta:+.0f}JPY, price={order_price:.0f}"
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
