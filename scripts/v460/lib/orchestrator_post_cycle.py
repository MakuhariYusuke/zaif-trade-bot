"""325# Mixin: OrchestratorPostCycleMixin — サイクル後処理/適応/間隔制御.

fill_loop_orchestrator.py の God Object 分割 (325#).
責務: post-cycle PnL 処理, 進捗ログ, adaptation 委譲, dynamic interval.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
    from ztb.metrics.fill_quality import FillRecord

logger = logging.getLogger(__name__)


class OrchestratorPostCycleMixin:
    """サイクル後処理 + adaptation 委譲 (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: PnL 計上, loss cooldown, DD update, loss_cap, batch persist,
          progress log, adaptation/lot 委譲, dynamic interval, heartbeat cleanup
      NG: ガード評価, セッション生成/終了, ループ制御, サイクル実行
    325# God Object 分割: fill_loop_orchestrator から抽出
    ────────────────────────────────────────────────────
    """

    def _process_post_cycle(
        self,
        record: "FillRecord",
        next_side: str,
        st: "RunSessionState",
    ) -> None:
        """run_continuous の 約定後処理.

        265# extract method: PnL 追跡, loss cooldown, toxic veto, DD update,
        soft/hard loss_cap, FastFillDefense, batch persistence。
        ~240 行の post-cycle ロジックを run_continuous から分離。

        Args:
            record: run_single_cycle の結果.
            next_side: このサイクルの side.
            st: ループ間共有状態.
        """
        from ztb.metrics.fill_quality import compute_record_pnl_jpy

        st.total_count += 1
        if record.filled:
            st.filled_count += 1
            self._track_side_pnl(record)
            # 202# A: 単一サイクル大損失クールダウン
            if (
                record.post_fill_30s_pnl is not None
                and record.post_fill_30s_pnl <= self.config.loss_cooldown_threshold_bps
            ):
                self._loss_cooldown_mult = self.config.loss_cooldown_interval_mult
                _lb = self.config.loss_boost_offset_mult
                if _lb > 1.0:
                    self._maker_price.set_loss_boost(_lb)
                logger.warning(
                    f"[202# A] Large cycle loss {record.post_fill_30s_pnl:.2f}bps "
                    f"<= {self.config.loss_cooldown_threshold_bps:.1f}bps — "
                    f"next interval ×{self._loss_cooldown_mult:.1f}"
                    f", offset ×{_lb:.1f}"
                )
            else:
                self._loss_cooldown_mult = 1.0
            # 205# §9.2: Toxic Fill veto
            if (
                self.config.toxic_fill_veto_cycles > 0
                and record.post_fill_30s_pnl is not None
                and record.post_fill_30s_pnl <= self.config.toxic_fill_veto_threshold_bps
            ):
                if self._toxic_veto is None:
                    self._toxic_veto = {}
                self._toxic_veto[next_side] = self.config.toxic_fill_veto_cycles
                self._inc_guard_fire("toxic_veto_set")
                logger.warning(
                    f"[205# §9.2] Toxic fill veto: {next_side} blocked for "
                    f"{self.config.toxic_fill_veto_cycles} cycles "
                    f"(pnl={record.post_fill_30s_pnl:.2f}bps "
                    f"<= {self.config.toxic_fill_veto_threshold_bps:.1f}bps)"
                )
            # 033# F4: 累積 PnL
            pnl_jpy = compute_record_pnl_jpy(record)
            if pnl_jpy is not None:
                st.cumulative_pnl_jpy += pnl_jpy
            # 249# BTC delta
            if record.order_quantity is not None:
                _fill_qty = float(record.order_quantity)
                if next_side == "buy":
                    st.cumulative_btc_delta += _fill_qty
                else:
                    st.cumulative_btc_delta -= _fill_qty
            # 250# adverse selection
            if record.adverse_selected is True:
                st.cumulative_adverse_count += 1
                if record.post_fill_30s_pnl is not None:
                    st.cumulative_adverse_bps += record.post_fill_30s_pnl
            # 348# balance_forced 撤廃: forced buy/sell KPI 分離トラッキングを削除
            # 168# §4.1: daily drawdown PnL update
            if record.post_fill_30s_pnl is not None:
                dd_result = self._daily_drawdown_guard.update_pnl(
                    record.post_fill_30s_pnl,
                    side=next_side,
                )
                if dd_result.get("soft_triggered"):
                    # 303# B: side-aware soft lot reduction
                    _triggered_side = dd_result.get("soft_triggered_side", "")
                    if (
                        self.config.daily_drawdown_soft_lot_side_aware
                        and _triggered_side in ("buy", "sell")
                    ):
                        # side 別 lot 倍率を 0.5 に縮小
                        if _triggered_side == "buy":
                            self._dd_soft_lot_scale_buy = 0.5
                        else:
                            self._dd_soft_lot_scale_sell = 0.5
                        logger.warning(
                            f"[daily_drawdown] 303# side-aware soft lot: "
                            f"{_triggered_side} scale → 0.5 "
                            f"(buy={self._dd_soft_lot_scale_buy}, "
                            f"sell={self._dd_soft_lot_scale_sell})"
                        )
                    else:
                        # 従来の集約 lot 縮小
                        old_lot = self._current_lot
                        new_lot = self._current_lot / 2
                        if new_lot >= self.config.order_quantity:
                            self._current_lot = new_lot
                            self._balance_checker.pre_shrink_lot = self._current_lot
                            logger.warning(
                                f"[daily_drawdown] soft lot reduction: "
                                f"{old_lot:.4f} → {self._current_lot:.4f} BTC"
                            )
                        else:
                            self._soft_drawdown_interval_multiplier = self.config.soft_drawdown_interval_multiplier
                            logger.warning(
                                f"[daily_drawdown] min lot reached ({old_lot:.4f} BTC), "
                                f"applying 3x interval multiplier instead of lot reduction"
                            )
        # 431# clamp observability: ceiling clamp 発火検出
        # 306# ceiling + 418# final_clamp の両方を検出。
        # offset_stages JSON パース不要: effective_offset == ceiling なら clamp 発火。
        # 431# SR-1 fix: skip_gate_skipped は bool|None — `not None` は True なので
        # 明示的に `is False` で比較（gate 評価済み＆非 skip の record のみ計上）。
        if record.skip_gate_skipped is False and record.effective_offset_used is not None:
            st.ceiling_check_count += 1
            # 467#: hour_ceiling_mult 反映
            from scripts.v460.lib.hour_rules import current_utc_hour
            _ceil = self.config.resolve_offset_ceiling(record.side, utc_hour=current_utc_hour())
            if _ceil > 0 and abs(record.effective_offset_used - _ceil) < 1e-6:
                st.clamp_fire_count += 1
        st.batch.append(record)
        self._recent_records.append(record)

        # soft/hard loss_cap
        if self.config.loss_cap_auto and not self._soft_loss_cap_triggered:
            if self._soft_cap_jpy_snapshot is not None:
                soft_cap_jpy = self._soft_cap_jpy_snapshot
            else:
                soft_cap_jpy = (
                    self.config.loss_cap_jpy
                    * self.config.soft_loss_cap_ratio
                    / self.config.loss_cap_ratio
                )
            if st.cumulative_pnl_jpy <= -soft_cap_jpy:
                old_lot = self._current_lot
                self._current_lot = max(
                    self.config.order_quantity,
                    self._current_lot / self.config.soft_loss_cap_lot_divisor,
                )
                self._soft_loss_cap_triggered = True
                self._balance_checker.pre_shrink_lot = self._current_lot
                logger.warning(
                    f"[loss_cap] SOFT CAP: cumPnL={st.cumulative_pnl_jpy:.0f} JPY "
                    f"<= -{soft_cap_jpy:.0f} JPY "
                    f"({self.config.soft_loss_cap_ratio:.0%}). "
                    f"ロット半減: {old_lot:.4f} → {self._current_lot:.4f} BTC"
                )

        if st.cumulative_pnl_jpy <= -self.config.loss_cap_jpy:
            logger.error(
                f"LOSS CAP REACHED (HARD): cumulative PnL = {st.cumulative_pnl_jpy:.0f} JPY "
                f"(cap = -{self.config.loss_cap_jpy:.0f} JPY). Stopping fill test."
            )
            self._kill_switch.kill("hard_loss_cap")

        # FastFillDefense
        if record.filled:
            self._fast_fill_defense.evaluate_fill(
                side=record.side,
                queue_wait_sec=record.queue_wait_sec,
                fill_price=record.fill_price,
                mid_at_fill=record.mid_at_fill,
                post_fill_pnl_bps=record.post_fill_30s_pnl,
            )
        elif not record.filled:
            self._fast_fill_defense.reset_on_unfilled(record.side)

        # batch persistence
        if len(st.batch) >= st.batch_size:
            if self._batch_persistence.try_save_batch(st.batch):
                st.batch = []
                self._batch_persistence.reset_flush_timer()
                self._adaptation_engine.invalidate_cache()
        else:
            st.batch = self._batch_persistence.maybe_flush(st.batch, "run_loop")

    # ------------------------------------------------------------------
    # 265# extract: progress log + state save + adaptation
    # ------------------------------------------------------------------
    async def _log_progress_and_adapt(
        self,
        next_side: str,
        st: "RunSessionState",
    ) -> None:
        """run_continuous の per-cycle 後半: 進捗ログ、state save、adaptation.

        265# extract method: progress log, health monitor, state persistence,
        dynamic loss_cap refresh, parameter/lot adaptation, stop conditions。

        Args:
            next_side: このサイクルの side.
            st: ループ間共有状態.
        """
        # 進捗ログ
        if self._cycle_count % self.config.progress_log_interval == 0:
            regime_tag = (
                self._regime_detector.current_regime.value
                if self._regime_detector else "n/a"
            )
            _fill_rate_pct = (
                st.filled_count / st.total_count * 100.0
                if st.total_count > 0 else 0.0
            )
            logger.info(
                f"Progress: {self._cycle_count} cycles, "
                f"fill rate={st.filled_count}/{st.total_count} "
                f"({_fill_rate_pct:.1f}%), "
                f"cumPnL={st.cumulative_pnl_jpy:.1f}JPY, "
                f"btcDelta={st.cumulative_btc_delta:+.4f}BTC, "
                f"lot={self._current_lot:.4f}BTC, "
                f"regime={regime_tag}, "
                f"none_regime={self._none_regime_cycle_count}/{self._total_regime_cycle_count}, "
                f"unsaved_batch={len(st.batch)}"
            )
            # 249# Total Equity MTM
            _mtm_mid = self._maker_price.last_mid_price if self._maker_price else None
            if _mtm_mid and _mtm_mid > 0:
                _equity_btc_val = st.cumulative_btc_delta * _mtm_mid
                _total_equity_delta = st.cumulative_pnl_jpy + _equity_btc_val
                logger.info(
                    f"[249# MTM] totalEquityΔ={_total_equity_delta:+.1f}JPY "
                    f"(spreadPnL={st.cumulative_pnl_jpy:+.1f} + "
                    f"btcMTM={_equity_btc_val:+.1f} "
                    f"@mid={_mtm_mid:.0f})"
                )
            # 250# P/L 3分離
            if st.cumulative_adverse_count > 0:
                _as_rate = (
                    st.cumulative_adverse_count / st.filled_count * 100.0
                    if st.filled_count > 0 else 0.0
                )
                logger.info(
                    f"[250# AS] adverseFills={st.cumulative_adverse_count} "
                    f"({_as_rate:.1f}%), "
                    f"cumASbps={st.cumulative_adverse_bps:+.1f}bps"
                )
            # 244# Guard reason category summary
            if self._guard_fire_counts:
                from scripts.v460.lib.guard_reason_classifier import (
                    guard_category_totals,
                )
                _cat_totals = guard_category_totals(self._guard_fire_counts)
                logger.info(
                    f"Guard category: "
                    f"market={_cat_totals['market']}, "
                    f"system={_cat_totals['system']}, "
                    f"recovery={_cat_totals['recovery']}"
                )
            # 431# clamp observability (428#/430# P3)
            if st.ceiling_check_count > 0:
                _clamp_rate = (
                    st.clamp_fire_count / st.ceiling_check_count * 100.0
                )
                _log_fn = logger.warning if _clamp_rate >= 90.0 else logger.info
                _log_fn(
                    f"[431# clamp] clampFires={st.clamp_fire_count}/"
                    f"{st.ceiling_check_count} ({_clamp_rate:.1f}%)"
                )
            # 348# balance_forced 撤廃: forced buy/sell KPI 分離ログを削除

        # 113# resilience: HealthMonitor + GC
        health_status = self._health_monitor.maybe_check(self._cycle_count)
        if health_status and health_status.get("level") == "critical":
            logger.error(
                f"[resilience] Health CRITICAL at cycle {self._cycle_count}: "
                f"{health_status}"
            )
        self._health_monitor.maybe_gc()

        # 113# resilience: 状態永続化
        _now_mono_save = time.monotonic()
        _progress_save = (
            self._cycle_count % self.config.progress_log_interval == 0
        )
        _time_save = (
            _now_mono_save - self._last_state_save_time
            >= self._STATE_SAVE_INTERVAL_SEC
        )
        if _progress_save or _time_save:
            self._update_lock_heartbeat()
            self._state_persistence.save(self._build_state_snapshot(
                total_count=st.total_count,
                filled_count=st.filled_count,
                cumulative_pnl_jpy=st.cumulative_pnl_jpy,
            ))
            self._last_state_save_time = _now_mono_save
            if _time_save and not _progress_save:
                logger.info(
                    f"[225# F2] Normal-cycle time-based state save "
                    f"(cycle={self._cycle_count})"
                )

        # 044# A-7: loss_cap 定期更新
        if (
            self.config.loss_cap_auto
            and self._cycle_count % self._loss_cap_update_interval == 0
            and self._cycle_count > 0
        ):
            await self._update_dynamic_loss_cap()

        # 032# P0: 方策 A 適応
        if (
            self.config.enable_auto_adapt
            and self._cycle_count % self.config.adapt_interval_cycles == 0
            and st.total_count >= self.config.min_adapt_samples
        ):
            self._try_auto_adapt(st.total_count, st.filled_count)

        # 033# 方策 B: 動的ロットサイジング
        if (
            self.config.enable_dynamic_lot
            and self._cycle_count % self.config.lot_adapt_interval_cycles == 0
            and st.total_count >= self.config.min_adapt_samples
        ):
            self._try_auto_lot_size()

        # 181# 停止条件モニター (277# config 化: stop_condition_check_interval)
        if (
            self._cycle_strategy is not None
            and self._cycle_count > 0
            and self._cycle_count % self.config.stop_condition_check_interval == 0
        ):
            self._check_regime_stop_conditions(st.filled_count, st.total_count)

    # ── 306# L1: σ 連動 dynamic cycle interval ────────────────────
    def _compute_dynamic_interval(self, base_interval: float) -> float:
        """309# σ に比例するサイクル間隔: σ 高 → 長く (Cooldown), σ 低 → 短く.

        interval = base_interval × (σ / σ_ref), clamped to [min_sec, max_sec].
        σ=0 (推定前) はフォールバックとして base_interval をそのまま返す。

        Avellaneda-Stoikov (2008): 高ボラ時は informed flow の密度が
        上がり、maker の逆選択リスクが増大する。最適応答は
        スプレッド拡大 + 実行頻度低下 (Cooldown)。
        308# 盲点2: 旧実装 (σ_ref/σ) は taker 戦術であり maker には逆効果。
        """
        sigma = self._maker_price.last_sigma
        if sigma <= 0:
            return base_interval
        cfg = self.config
        ratio = sigma / cfg.dynamic_cycle_interval_sigma_ref
        adjusted = base_interval * ratio
        return max(cfg.dynamic_cycle_interval_min_sec,
                   min(adjusted, cfg.dynamic_cycle_interval_max_sec))

    async def cleanup_heartbeat(self) -> None:
        """175# 異常終了時の heartbeat タスク cleanup.

        run_continuous の呼び出し元で finally ブロックから呼ぶことで、
        未処理例外発生時の heartbeat タスクリークを防止する。
        """
        # 254# getattr → クラスレベルデフォルト直接参照
        task = self._heartbeat_task
        if task is not None and not task.done():
            task.cancel()
            self._heartbeat_task = None
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info("[cleanup] heartbeat task cancelled (exception path)")

    def _build_adapt_kwargs(self) -> dict[str, object]:
        """120# AdaptationEngine に委譲."""
        return self._adaptation_engine._build_adapt_kwargs()

    def _build_lot_kwargs(self) -> dict[str, object]:
        """120# AdaptationEngine に委譲."""
        return self._adaptation_engine._build_lot_kwargs()

    async def _update_dynamic_loss_cap(self) -> None:
        """041# 動的 loss_cap — 120# AdaptationEngine に委譲."""
        await self._adaptation_engine.update_dynamic_loss_cap(
            self.adapter, self.config.symbol,
        )

    def _try_auto_adapt(self, total_count: int, filled_count: int) -> None:
        """032# P0: 方策 A — 120# AdaptationEngine に委譲."""
        result = self._adaptation_engine.try_auto_adapt(
            total_count=total_count,
            filled_count=filled_count,
            base_offset_ratio=self._maker_price.base_offset_ratio,
            base_offset_ratio_buy=self._maker_price.base_offset_ratio_buy,
            base_offset_ratio_sell=self._maker_price.base_offset_ratio_sell,
            regime_detector=self._regime_detector,
            fast_fill_defense=self._fast_fill_defense,
        )
        # offset 変更を MakerPriceCalculator に反映
        if result.base_offset_changed or result.buy_offset_changed or result.sell_offset_changed:
            self._maker_price.update_base_offsets(
                result.new_base_offset,
                result.new_buy_offset,
                result.new_sell_offset,
            )

    def _try_auto_lot_size(self) -> None:
        """033# 方策 B — 120# AdaptationEngine に委譲."""
        changed, new_lot = self._adaptation_engine.try_auto_lot_size(
            self._current_lot,
            regime_detector=self._regime_detector,
        )
        if changed:
            self._current_lot = new_lot
