"""325# Mixin: OrchestratorGuardsMixin — リスクガード・veto・side 判定.

fill_loop_orchestrator.py の God Object 分割 (325#).
責務: kill 判定, toxicity 評価, guard fire カウンタ, MCB/SAD フィード,
      残高チェック, 時間帯フィルタ, 停止条件, 滞留注文クリア.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ztb.metrics.fill_quality import FillRecord
    from ztb.risk.sell_dynamic_kill import DynamicKillManager
    from ztb.risk.toxicity_types import ToxicityAssessment

logger = logging.getLogger(__name__)


class OrchestratorGuardsMixin:
    """リスクガード・kill 判定・veto 管理 (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: kill 判定, toxicity 評価, guard fire, MCB/SAD feed,
          残高チェック, 時間帯フィルタ, 停止条件, 滞留注文クリア
      NG: ループ制御, サイクル実行, 状態永続化, adaptation
    325# God Object 分割: fill_loop_orchestrator から抽出
    ────────────────────────────────────────────────────
    """

    def _is_side_killed(self, side: str) -> bool:
        """275# DRY: side パラメータ化 — dynamic kill 判定を単一メソッドに統合.

        旧 _is_sell_killed / _is_buy_killed の対称コードを統一。
        Glosten-Milgrom (1985) 逆選択モデルに基づく動的 kill 判定を
        side 非依存で実行する。

        286# 283# P1-4: buy 側に在庫連動緩和を追加 (Ho & Stoll 1981)。
        在庫偏重時に buy kill 閾値を段階的に緩和し、在庫リバランスを促進。

        343# kill リリース追跡: kill→非kill 遷移時にサイクル番号を記録し、
        skip_gate が kill 直後の過剰抑制を回避できるようにする。

        535# Pre-emptive CV kill: CV adverse velocity が持続する場合に
        rolling PnL の悪化を待たず事前に kill する (532# §4 対応)。
        """
        # 535# Pre-emptive CV kill: sell 側のみ (532# §4: sell_dynamic_kill の事後反応問題)
        if side == "sell" and self.config.sell_preemptive_cv_kill_enabled:
            preemptive_killed = self._check_preemptive_cv_kill()
            if preemptive_killed:
                return self._apply_kill_release_tracking(side, killed=True)
        mgr = self._sell_kill_mgr if side == "sell" else self._buy_kill_mgr
        regime: str | None = None
        if self._regime_detector is not None:
            regime = self._regime_detector.current_regime.value

        # 286# 283# P1-4 / 337#: 在庫連動の kill 閾値緩和 (Ho & Stoll 1981 対称)
        threshold_offset_bps = 0.0
        if (
            side == "buy"
            and self.config.buy_dynamic_kill_inv_relaxation_enabled
        ):
            imbalance = self._maker_price.inv_net_imbalance
            if imbalance < 0:  # buy 偏重 = BTC 不足
                raw_offset = abs(imbalance) * self.config.buy_dynamic_kill_inv_relaxation_scale
                threshold_offset_bps = min(
                    raw_offset,
                    self.config.buy_dynamic_kill_inv_relaxation_max_bps,
                )
                if threshold_offset_bps > 0.01:
                    logger.debug(
                        f"[286# P1-4] buy kill threshold relaxed by "
                        f"+{threshold_offset_bps:.3f}bps "
                        f"(imbalance={imbalance:.3f})"
                    )
        elif (
            side == "sell"
            and self.config.sell_dynamic_kill_inv_relaxation_enabled
        ):
            imbalance = self._maker_price.inv_net_imbalance
            if imbalance > 0:  # sell 偏重 = BTC 過剰
                raw_offset = abs(imbalance) * self.config.sell_dynamic_kill_inv_relaxation_scale
                threshold_offset_bps = min(
                    raw_offset,
                    self.config.sell_dynamic_kill_inv_relaxation_max_bps,
                )
                if threshold_offset_bps > 0.01:
                    logger.debug(
                        f"[337#] sell kill threshold relaxed by "
                        f"+{threshold_offset_bps:.3f}bps "
                        f"(imbalance={imbalance:.3f})"
                    )

        killed, telemetry = mgr.check_kill(
            regime=regime,
            threshold_offset_bps=threshold_offset_bps,
        )
        # 223# probe/force_release メトリクス
        if telemetry.probe_fired:
            self._inc_guard_fire(f"dynamic_kill_probe_{side}")
        if telemetry.force_release_fired:
            self._inc_guard_fire(f"dynamic_kill_force_release_{side}")
        if killed:
            logger.info(
                f"[275# DRY] {side} kill: regime={regime or 'default'}, "
                f"threshold_used={telemetry.threshold_used}, "
                f"cooldown_remaining={telemetry.cooldown_remaining}"
            )
        return self._apply_kill_release_tracking(side, killed)

    def _apply_kill_release_tracking(self, side: str, killed: bool) -> bool:
        """343# kill リリース追跡 — kill(True)→非kill(False) 遷移を検出."""
        if side == "buy":
            was_active = self._kill_was_active_buy
            if was_active and not killed:
                self._kill_released_at_cycle_buy = self._cycle_count
                logger.info(
                    "[343#] buy kill released at cycle %d",
                    self._kill_released_at_cycle_buy,
                )
            self._kill_was_active_buy = killed
        else:
            was_active = self._kill_was_active_sell
            if was_active and not killed:
                self._kill_released_at_cycle_sell = self._cycle_count
                logger.info(
                    "[343#] sell kill released at cycle %d",
                    self._kill_released_at_cycle_sell,
                )
            self._kill_was_active_sell = killed
        return killed

    def _check_preemptive_cv_kill(self) -> bool:
        """535# Pre-emptive CV kill: CV adverse velocity 持続時に sell を事前ブロック.

        532# §4 対応: sell_dynamic_kill は損失発生後に反応する構造的欠陥がある。
        CV (cross-venue) の adverse velocity が連続して高い場合、rolling PnL が
        悪化する前に sell を pre-emptive にブロックする。

        Returns:
            True if sell should be pre-emptively killed.
        """
        # cooldown 中は kill 継続
        if self._preemptive_cv_sell_cooldown > 0:
            self._preemptive_cv_sell_cooldown -= 1
            return True

        cfg = self.config
        cv_hint = self._maker_price.cross_venue_lead_lag_hint
        if cv_hint is None:
            self._preemptive_cv_sell_adverse_count = 0
            return False

        # adverse_side == "sell" かつ velocity と confidence が閾値以上
        if (
            cv_hint.adverse_side == "sell"
            and abs(cv_hint.reference_velocity_bps) >= cfg.sell_preemptive_cv_velocity_threshold
            and cv_hint.confidence >= cfg.sell_preemptive_cv_confidence_floor
        ):
            self._preemptive_cv_sell_adverse_count += 1
            if self._preemptive_cv_sell_adverse_count >= cfg.sell_preemptive_cv_consecutive_threshold:
                self._preemptive_cv_sell_cooldown = cfg.sell_preemptive_cv_cooldown_cycles
                self._inc_guard_fire("preemptive_cv_sell_kill")
                logger.warning(
                    "[535#] sell pre-emptive CV kill activated: "
                    "velocity=%+.2fbps/s, confidence=%.2f, "
                    "consecutive=%d, cooldown=%d cycles",
                    cv_hint.reference_velocity_bps,
                    cv_hint.confidence,
                    self._preemptive_cv_sell_adverse_count,
                    cfg.sell_preemptive_cv_cooldown_cycles,
                )
                self._preemptive_cv_sell_adverse_count = 0
                return True
        else:
            self._preemptive_cv_sell_adverse_count = 0

        return False

    def _track_side_pnl(self, record: "FillRecord") -> None:
        """275# DRY: side パラメータ化 — PnL 追跡を単一メソッドに統合.

        Ho & Stoll (1981) の在庫リスク管理に基づき、side 別の PnL 履歴を
        対称に追跡して kill / relaxation 判定へ渡す。

        286# 283# P2-7: 評価窓二重化の基盤。
        348# balance_forced 撤廃: forced downweight 論理を削除。
        """
        if not (record.filled and record.post_fill_30s_pnl is not None):
            return
        pnl = record.post_fill_30s_pnl
        if record.side == "sell":
            self._sell_kill_mgr.track(pnl)
        elif record.side == "buy":
            self._buy_kill_mgr.track(pnl)

    # ------------------------------------------------------------------
    # 240# Toxicity Budget — assess_toxicity (副作用なし)
    # ------------------------------------------------------------------
    def _assess_toxicity(
        self, mgr: "DynamicKillManager",
    ) -> "ToxicityAssessment | None":
        """240# toxicity budget 評価 (副作用なし)."""
        if not mgr.config.toxicity_budget_enabled:
            return None
        regime: str | None = None
        if self._regime_detector is not None:
            regime = self._regime_detector.current_regime.value
        return mgr.assess_toxicity(regime=regime)

    def _assess_buy_toxicity(self) -> "ToxicityAssessment | None":
        """240# buy 側の toxicity budget 評価."""
        return self._assess_toxicity(self._buy_kill_mgr)

    def _assess_sell_toxicity(self) -> "ToxicityAssessment | None":
        """240# sell 側の toxicity budget 評価."""
        return self._assess_toxicity(self._sell_kill_mgr)

    # ------------------------------------------------------------------
    # 216# E: Guard 発火カウンタ — インクリメント・ヘルパー
    # ------------------------------------------------------------------
    def _inc_guard_fire(self, guard_name: str) -> None:
        """累積 guard 発火カウンタをインクリメント."""
        if self._guard_fire_counts is None:
            self._guard_fire_counts = {}
        self._guard_fire_counts[guard_name] = self._guard_fire_counts.get(guard_name, 0) + 1

    def _guard_category_totals(self) -> dict[str, int] | None:
        """244# guard_fire_counts のカテゴリ別合計を返す."""
        if not self._guard_fire_counts:
            return None
        from scripts.v460.lib.guard_reason_classifier import guard_category_totals
        return guard_category_totals(self._guard_fire_counts)

    # ------------------------------------------------------------------
    # 272# DRY: toxic_veto 減算ヘルパー
    # ------------------------------------------------------------------
    def _tick_toxic_veto(self, context: str) -> None:
        """全 toxic_veto カウンタを 1 減算し、期限切れを除去する."""
        if not self._toxic_veto:
            return
        for _vs in list(self._toxic_veto.keys()):
            self._toxic_veto[_vs] -= 1
            if self._toxic_veto[_vs] <= 0:
                del self._toxic_veto[_vs]
                logger.info(f"[226# S2] Toxic veto expired ({context}): {_vs}")

    # ------------------------------------------------------------------
    # 272# DRY: MCB/SAD フィードヘルパー
    # ------------------------------------------------------------------
    def _feed_mcb_sad(self) -> None:
        """MCB/SAD に最新の mid_price / spread をフィードする."""
        _now = time.time()
        if self._mcb is not None and self._mcb.config.enabled:
            _mcb_mid = self._maker_price.last_mid_price
            if _mcb_mid is not None and _mcb_mid > 0:
                self._mcb.update(_mcb_mid, _now)
        if self._sad is not None and self._sad.config.enabled:
            _sad_spread = self._maker_price.last_spread_raw
            if _sad_spread is not None and _sad_spread > 0:
                self._sad.update(_sad_spread, _now)

    @staticmethod
    def _opposite_side(side: str) -> str:
        """反対サイドを返す."""
        return "sell" if side == "buy" else "buy"

    # ------------------------------------------------------------------
    # 181# 停止条件モニター — C/D/Chase 安全弁
    # ------------------------------------------------------------------
    def _check_regime_stop_conditions(
        self, filled_count: int, total_count: int,
    ) -> None:
        """fill_rate / avg_pnl30 を判定し、閾値違反時に fallback を起動."""
        strategy = self._cycle_strategy
        policy = strategy.policy
        if not (policy.dynamic_cycle_enabled or policy.chase_enabled):
            return
        if total_count > 0 and filled_count / total_count < policy.fill_rate_floor:
            logger.warning(
                f"[181# stop] fill_rate={filled_count/total_count:.2%} → fallback"
            )
            strategy.activate_fallback(self.config.fallback_duration_sec)
            return
        _pnl_window = self.config.sell_dynamic_kill_window * 2
        pnls = [
            r.post_fill_30s_pnl for r in self._recent_records
            if r.filled and r.post_fill_30s_pnl is not None
        ][-_pnl_window:]
        _min_pnl_samples = max(1, self.config.min_adapt_samples // 5)
        if len(pnls) >= _min_pnl_samples:
            avg = sum(pnls) / len(pnls)
            if avg < policy.pnl_floor_bps:
                logger.warning(f"[181# stop] avg_pnl30={avg:.2f}bps → fallback")
                strategy.activate_fallback(self.config.fallback_duration_sec)

    def _is_time_filtered(self, side: str | None = None) -> bool:
        """時間帯フィルター — 121# TimeFilter に委譲."""
        regime = self._current_regime_value()
        return self._time_filter.is_filtered(side=side, regime=regime)

    async def _check_balance_for_side(
        self, side: str, *, regime_mult: float = 1.0,
    ) -> bool:
        """残高 pre-flight check — 121# BalanceChecker に委譲."""
        return await self._balance_checker.check(
            side, self.adapter, self.config.symbol,
            regime_mult=regime_mult,
        )

    async def _cancel_stale_orders(self) -> int:
        """042# 起動時の滞留注文自動クリア.

        Returns:
            キャンセルした注文数。
        """
        cancelled_count = 0
        try:
            open_orders = await self.adapter.get_open_orders(self.config.symbol)
            if not open_orders:
                logger.info("[startup] No stale orders found.")
                return 0
            for order in open_orders:
                try:
                    await self.adapter.cancel_order(order.order_id)
                    cancelled_count += 1
                    logger.warning(
                        f"[startup] Cancelled stale order: "
                        f"id={order.order_id}, side={order.side}, "
                        f"price={order.price}, qty={order.quantity}"
                    )
                except Exception as e:
                    logger.error(
                        f"[startup] Failed to cancel stale order "
                        f"{order.order_id}: {e}"
                    )
            logger.info(
                f"[startup] Stale order cleanup complete: "
                f"{cancelled_count}/{len(open_orders)} cancelled."
            )
        except Exception as e:
            logger.warning(f"[startup] Stale order check failed (non-fatal): {e}")
        return cancelled_count
