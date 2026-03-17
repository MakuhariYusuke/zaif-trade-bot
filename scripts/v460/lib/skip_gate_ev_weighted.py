"""461# SkipGate ev_weighted 判定 Mixin.

skip_gate_evaluator.py から ev_weighted 統合判定ロジックを抽出。

責務:
  - 188# C-1 ev_weighted 統合判定 (primary + alt horizon)
  - 190# A/B 安全弁 + 片側 balance 緩和
  - 193# offset 修飾子モード (ev_as_offset)

MAX LINES: 250
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig
    from scripts.v460.lib.skip_gate_evaluator import (
        _SkipDecisionLike,
        _SkipGateLike,
    )

logger = logging.getLogger(__name__)


class SkipGateEvWeightedMixin:
    """ev_weighted 統合判定メソッドを提供する Mixin.

    SkipGateEvaluator が継承して使用する。
    """

    def _try_ev_weighted_decision(
        self,
        side: str,
        features: dict[str, object] | dict[str, float],
        regime: str,
        threshold_offset: float,
        primary_decision: _SkipDecisionLike,
        *,
        one_sided_balance: bool = False,
    ) -> _SkipDecisionLike | None:
        """188# C-1: ev_weighted 統合判定.

        副 horizon モデルが存在し ev_weighted_enabled=True の場合、
        primary (短期) と alt (長期) の predicted_pnl を ev_weighted で統合。
        AS mode では ev_weighted は適用しない (確率ベースのため加重平均が不適切)。

        193#: ev_as_offset_enabled=True 時は offset 修飾子モードに切り替え。
        190# A: 連続 skip 安全弁 + B: 片側 balance 時の threshold 緩和。

        Returns:
            統合判定の SkipDecision, または None (ev_weighted 不適用時).
        """
        config: FillTestConfig = self._config  # type: ignore[attr-defined]
        if not config.skip_gate_ev_weighted_enabled:
            return None

        # AS mode では ev_weighted 不適用
        alt_gate: _SkipGateLike | None
        if side == "buy":
            alt_gate = self._gate_alt_buy  # type: ignore[attr-defined]
        elif side == "sell":
            alt_gate = self._gate_alt_sell  # type: ignore[attr-defined]
        else:
            return None

        if alt_gate is None:
            return None

        # PnL mode のみ ev_weighted (AS mode は確率空間の加重平均が不適切)
        if alt_gate.config.mode != "pnl":
            logger.debug(
                "[skip_gate] 188# ev_weighted skipped: alt model mode=%s (pnl required)",
                alt_gate.config.mode,
            )
            return None

        try:
            alt_decision = alt_gate.evaluate(
                features,
                side=side,
                regime=regime,
                threshold_offset=threshold_offset,
            )
        except Exception as e:
            logger.warning("[skip_gate] 188# alt model evaluate failed: %s", e)
            return None

        # primary と alt の pred_pnl を ev_weighted 合成
        w30 = config.skip_gate_ev_w30
        w120 = config.skip_gate_ev_w120
        primary_pnl = primary_decision.predicted_pnl_bps
        alt_pnl = alt_decision.predicted_pnl_bps

        # buy: primary=pnl30 (短期), alt=pnl120 (長期)
        # sell: primary=pnl120 (長期), alt=pnl30 (短期)
        if side == "buy":
            ev_score = w30 * primary_pnl + w120 * alt_pnl
        else:
            ev_score = w30 * alt_pnl + w120 * primary_pnl

        # 193#: offset 修飾子モード
        if config.skip_gate_ev_as_offset_enabled:
            return self._ev_weighted_as_offset(
                side, ev_score, primary_decision,
            )

        # --- 旧モード: ハードゲート (ev_as_offset_enabled=False) ---
        threshold_used = primary_decision.threshold_used
        if threshold_used is None:
            threshold_used = 0.0

        # 190# B: 片側 balance 時の threshold 緩和
        _threshold_relaxation = config.skip_gate_ev_one_sided_threshold_shift
        if one_sided_balance and _threshold_relaxation != 0.0:
            _original_threshold = threshold_used
            threshold_used += _threshold_relaxation
            logger.debug(
                "[skip_gate] 190# B: one_sided_balance threshold relaxation: "
                "%.3f → %.3f (shift=%.3f)",
                _original_threshold, threshold_used, _threshold_relaxation,
            )

        should_skip = ev_score < threshold_used

        # 190# A: 連続 skip 安全弁
        _max_consecutive = config.skip_gate_ev_max_consecutive_skip
        if should_skip and _max_consecutive > 0:
            self._ev_consecutive_skip_count += 1  # type: ignore[attr-defined]
            if self._ev_consecutive_skip_count >= _max_consecutive:  # type: ignore[attr-defined]
                logger.warning(
                    "[skip_gate] 190# A: ev_weighted consecutive skip safety valve: "
                    "%d consecutive skips >= limit %d — forcing PASS "
                    "(score=%.3f, threshold=%.3f)",
                    self._ev_consecutive_skip_count, _max_consecutive,  # type: ignore[attr-defined]
                    ev_score, threshold_used,
                )
                self._ev_consecutive_skip_count = 0  # type: ignore[attr-defined]
                should_skip = False
        elif not should_skip:
            self._ev_consecutive_skip_count = 0  # type: ignore[attr-defined]

        logger.debug(
            "[skip_gate] 188# ev_weighted: side=%s pnl_primary=%.3f pnl_alt=%.3f "
            "ev=%.3f threshold=%.3f skip=%s consec=%d",
            side, primary_pnl, alt_pnl, ev_score, threshold_used, should_skip,
            self._ev_consecutive_skip_count,  # type: ignore[attr-defined]
        )

        _reason = (
            "ev_weighted_skip" if should_skip
            else "ev_weighted_pass_safety" if self._ev_consecutive_skip_count == 0 and ev_score < threshold_used  # type: ignore[attr-defined]
            else "ev_weighted_pass"
        )

        from scripts.v460.ml.skip_gate import SkipDecision
        return SkipDecision(
            should_skip=should_skip,
            predicted_pnl_bps=ev_score,
            threshold_bps=primary_decision.threshold_bps,
            features_used=primary_decision.features_used if hasattr(primary_decision, "features_used") else 0,
            reason=_reason,
            model_used="ev_weighted",
            as_probability=primary_decision.as_probability,
            threshold_used=threshold_used,
        )

    def _ev_weighted_as_offset(
        self,
        side: str,
        ev_score: float,
        primary_decision: _SkipDecisionLike,
    ) -> _SkipDecisionLike:
        """193#: ev_weighted を offset 修飾子として機能させる."""
        config: FillTestConfig = self._config  # type: ignore[attr-defined]
        _emergency = config.skip_gate_ev_emergency_skip_threshold
        if ev_score < _emergency:
            logger.warning(
                "[skip_gate] 193# ev_weighted EMERGENCY SKIP: "
                "ev_score=%.3f < emergency_threshold=%.3f",
                ev_score, _emergency,
            )
            self._ev_consecutive_skip_count = 0  # type: ignore[attr-defined]

            from scripts.v460.ml.skip_gate import SkipDecision
            return SkipDecision(
                should_skip=True,
                predicted_pnl_bps=ev_score,
                threshold_bps=primary_decision.threshold_bps,
                features_used=primary_decision.features_used if hasattr(primary_decision, "features_used") else 0,
                reason="ev_weighted_emergency_skip",
                model_used="ev_weighted",
                as_probability=primary_decision.as_probability,
                threshold_used=_emergency,
            )

        # offset 修飾子モード: 安全弁カウンタリセット
        self._ev_consecutive_skip_count = 0  # type: ignore[attr-defined]

        # 200# M: DRY — compute_ev_offset_multiplier に共通化 + warning zone
        from scripts.v460.lib.fill_config import compute_ev_offset_multiplier
        _clamped_mult = compute_ev_offset_multiplier(
            ev_score=ev_score,
            sensitivity=config.skip_gate_ev_offset_sensitivity,
            min_mult=config.skip_gate_ev_offset_min_mult,
            max_mult=config.skip_gate_ev_offset_max_mult,
            warning_threshold=config.skip_gate_ev_warning_threshold,
            warning_factor=config.skip_gate_ev_warning_offset_factor,
        )

        logger.info(
            "[skip_gate] 193# ev_weighted→offset: side=%s ev_score=%.3f "
            "→ offset_mult=%.3f (sens=%.3f, clamp=[%.2f,%.2f])",
            side, ev_score, _clamped_mult,
            config.skip_gate_ev_offset_sensitivity,
            config.skip_gate_ev_offset_min_mult,
            config.skip_gate_ev_offset_max_mult,
        )

        from scripts.v460.ml.skip_gate import SkipDecision
        return SkipDecision(
            should_skip=False,
            predicted_pnl_bps=ev_score,
            threshold_bps=primary_decision.threshold_bps,
            features_used=primary_decision.features_used if hasattr(primary_decision, "features_used") else 0,
            reason="ev_weighted_offset",
            model_used="ev_weighted",
            as_probability=primary_decision.as_probability,
            threshold_used=0.0,
        )
