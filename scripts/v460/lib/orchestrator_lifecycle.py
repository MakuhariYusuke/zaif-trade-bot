"""325# Mixin: OrchestratorLifecycleMixin — セッション初期化/終了/状態管理.

fill_loop_orchestrator.py の God Object 分割 (325#).
責務: warmup, state snapshot/restore, init_run_session, finalize_run, cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
    from ztb.metrics.fill_quality import FillRecord

logger = logging.getLogger(__name__)


class OrchestratorLifecycleMixin:
    """セッション生存期間管理 (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: DD/kill warmup, state snapshot/restore, session init/finalize, cleanup
      NG: ループ制御, サイクル実行, ガード評価, adaptation
    325# God Object 分割: fill_loop_orchestrator から抽出
    ────────────────────────────────────────────────────
    """

    def _warmup_daily_drawdown_from_records(
        self, records: list["FillRecord"],
    ) -> None:
        """203# F: fill records から当日分の PnL を DD guard に投入.

        state file が stale/missing の場合のセーフティネット。

        277# fix (B1): warmup は DD guard と同一 TZ で日付境界を判定する。
        """
        guard = self._daily_drawdown_guard
        today_str = guard._today()
        day_reset_tz = guard._day_reset_tz
        daily_pnl_sum = 0.0
        daily_fill_count = 0
        daily_pnl_buy = 0.0
        daily_pnl_sell = 0.0
        for r in records:
            if not r.filled or r.post_fill_30s_pnl is None:
                continue
            r_date = datetime.fromtimestamp(r.timestamp, tz=day_reset_tz).strftime("%Y%m%d")
            if r_date != today_str:
                continue
            daily_pnl_sum += r.post_fill_30s_pnl
            daily_fill_count += 1
            if r.side == "buy":
                daily_pnl_buy += r.post_fill_30s_pnl
            elif r.side == "sell":
                daily_pnl_sell += r.post_fill_30s_pnl

        if daily_fill_count > 0:
            guard.state.daily_pnl_bps = daily_pnl_sum
            guard.state.daily_fill_count = daily_fill_count
            guard.state.current_day = today_str
            guard.state.daily_pnl_bps_buy = daily_pnl_buy
            guard.state.daily_pnl_bps_sell = daily_pnl_sell
            if guard._per_side_enabled:
                if daily_pnl_buy <= guard._per_side_hard_limit_bps:
                    guard.state.side_halted_buy = True
                    guard.state.side_halt_remaining_buy = guard._per_side_halt_cycles
                if daily_pnl_sell <= guard._per_side_hard_limit_bps:
                    guard.state.side_halted_sell = True
                    guard.state.side_halt_remaining_sell = guard._per_side_halt_cycles
            if daily_pnl_sum <= guard._soft_limit_bps:
                guard._soft_triggered_today = True
            if daily_pnl_sum <= guard._hard_limit_bps:
                guard.state.halted = True
                guard.state.halt_triggered_at = time.time()
            logger.warning(
                f"[203# F] DD warmup from fill records: {daily_fill_count} fills today, "
                f"daily_pnl={daily_pnl_sum:+.2f}bps, "
                f"buy={daily_pnl_buy:+.2f}bps, sell={daily_pnl_sell:+.2f}bps, "
                f"halted={guard.state.halted}"
            )

    # ------------------------------------------------------------------
    # 209# H4: DynamicKillManager warmup
    # ------------------------------------------------------------------
    def _warmup_kill_managers_from_records(
        self, records: list["FillRecord"],
    ) -> None:
        """209# H4: fill records から sell/buy kill manager の PnL 履歴を復元.

        225# F1: 当日分のみ replay — B2 日替わり kill reset との矛盾を防止。
        """
        utc_today = datetime.now(timezone.utc).strftime("%Y%m%d")
        sell_count = 0
        buy_count = 0
        skipped_old = 0
        for r in records:
            if not r.filled or r.post_fill_30s_pnl is None:
                continue
            r_date = datetime.fromtimestamp(
                r.timestamp, tz=timezone.utc,
            ).strftime("%Y%m%d")
            if r_date != utc_today:
                skipped_old += 1
                continue
            if r.side == "sell":
                self._sell_kill_mgr.track(r.post_fill_30s_pnl)
                sell_count += 1
            elif r.side == "buy":
                self._buy_kill_mgr.track(r.post_fill_30s_pnl)
                buy_count += 1
        if sell_count > 0 or buy_count > 0 or skipped_old > 0:
            logger.info(
                f"[209# H4] Kill manager warmup from fill records (today only): "
                f"sell={sell_count}, buy={buy_count}, "
                f"skipped_old={skipped_old}"
            )

    # ------------------------------------------------------------------
    # 210# DRY: FillTestState 構築の共通化
    # ------------------------------------------------------------------
    def _build_state_snapshot(
        self,
        *,
        total_count: int,
        filled_count: int,
        cumulative_pnl_jpy: float,
    ) -> object:
        """現在の状態から FillTestState スナップショットを構築."""
        from scripts.v460.lib.resilience import FillTestState

        return FillTestState(
            run_id=self._run_id,
            cycle_count=self._cycle_count,
            total_count=total_count,
            filled_count=filled_count,
            cumulative_pnl_jpy=cumulative_pnl_jpy,
            current_lot=self._current_lot,
            soft_loss_cap_triggered=self._soft_loss_cap_triggered,
            base_offset_ratio=self._maker_price.base_offset_ratio,
            base_offset_ratio_buy=self._maker_price.base_offset_ratio_buy,
            base_offset_ratio_sell=self._maker_price.base_offset_ratio_sell,
            daily_drawdown_state=self._daily_drawdown_guard.export_state(),
            toxic_veto=dict(self._toxic_veto) if self._toxic_veto else None,
            one_sided_consecutive_count=self._one_sided_consecutive_count,
            soft_drawdown_interval_multiplier=self._soft_drawdown_interval_multiplier,
            guard_fire_counts=dict(self._guard_fire_counts) if self._guard_fire_counts else None,
            guard_category_totals=self._guard_category_totals(),
            sell_kill_state=self._sell_kill_mgr.export_state(),
            buy_kill_state=self._buy_kill_mgr.export_state(),
            mcb_state=(
                self._mcb.export_state() if self._mcb is not None else None
            ),
            sad_state=(
                self._sad.export_state() if self._sad is not None else None
            ),
            degraded_liquidation_duty_counter=self._degraded_liquidation_duty_counter,
            inventory_escape_duty_counter=self._inventory_escape_duty_counter,
            one_sided_cooldown_remaining=self._one_sided_cooldown_remaining,
            one_sided_freeze_remaining=self._one_sided_freeze_remaining,
            one_sided_frozen_side=self._one_sided_frozen_side,
            consecutive_no_feasible=(
                dict(self._consecutive_no_feasible)
                if self._consecutive_no_feasible
                else None
            ),
            phantom_guard_metrics=(
                self._phantom_guard.get_metrics()
                if self._phantom_guard is not None
                else None
            ),
            **self._get_regime_state_fields(),
        )

    # ------------------------------------------------------------------
    # 272# DRY: skip-path state save ヘルパー
    # ------------------------------------------------------------------
    def _maybe_skip_state_save(
        self,
        st: "RunSessionState",
        context: str,
    ) -> None:
        """_STATE_SAVE_INTERVAL_SEC 経過時のみ state 保存する."""
        _now_mono = time.monotonic()
        if _now_mono - self._last_state_save_time >= self._STATE_SAVE_INTERVAL_SEC:
            self._state_persistence.save(self._build_state_snapshot(
                total_count=st.total_count,
                filled_count=st.filled_count,
                cumulative_pnl_jpy=st.cumulative_pnl_jpy,
            ))
            self._last_state_save_time = _now_mono
            logger.info(f"[272#] skip-time state save ({context})")

    # ------------------------------------------------------------------
    # 216# §6 DRY: State 復元共通ヘルパー
    # ------------------------------------------------------------------
    def _restore_common_state(self, saved_state: object | None) -> None:
        """DD / toxic_veto / one-sided / guard_fire_counts の共通復元."""
        if saved_state is None:
            return
        # 168# §4.1 #3: 日次ドローダウンガード状態復元
        if saved_state.daily_drawdown_state:
            self._daily_drawdown_guard.import_state(saved_state.daily_drawdown_state)
        # 207# §1: toxic veto 状態復元
        if saved_state.toxic_veto:
            self._toxic_veto = dict(saved_state.toxic_veto)
            logger.info(f"[207# §1] Toxic veto restored: {self._toxic_veto}")
        # 210# L-2: one-sided 連続カウンタ復元
        if saved_state.one_sided_consecutive_count > 0:
            self._one_sided_consecutive_count = saved_state.one_sided_consecutive_count
            logger.info(
                f"[210# L-2] One-sided count restored: "
                f"{self._one_sided_consecutive_count}"
            )
        # 224#: soft drawdown interval 乗数復元
        _sd_mult = saved_state.soft_drawdown_interval_multiplier
        if _sd_mult != 1.0:
            self._soft_drawdown_interval_multiplier = _sd_mult
            logger.info(
                f"[224#] Soft drawdown interval multiplier restored: {_sd_mult:.1f}"
            )
        # 216# E: Guard 発火カウンタ復元
        if saved_state.guard_fire_counts:
            self._guard_fire_counts = dict(saved_state.guard_fire_counts)
            logger.info(f"[216# E] Guard fire counts restored: {self._guard_fire_counts}")
        # 209# H4: DynamicKillManager 状態復元
        if saved_state.sell_kill_state:
            self._sell_kill_mgr.import_state(saved_state.sell_kill_state)
            logger.info(
                f"[209# H4] Sell kill state restored: "
                f"history={len(self._sell_kill_mgr._pnl_history)}, "
                f"cooldown={self._sell_kill_mgr._cooldown}, "
                f"kills={self._sell_kill_mgr._total_kills}"
            )
        if saved_state.buy_kill_state:
            self._buy_kill_mgr.import_state(saved_state.buy_kill_state)
            logger.info(
                f"[209# H4] Buy kill state restored: "
                f"history={len(self._buy_kill_mgr._pnl_history)}, "
                f"cooldown={self._buy_kill_mgr._cooldown}, "
                f"kills={self._buy_kill_mgr._total_kills}"
            )
        # 225# MCB/SAD 状態復元
        _mcb_state = saved_state.mcb_state
        if _mcb_state and self._mcb is not None:
            self._mcb.import_state(_mcb_state)
            logger.info(
                f"[225#] MCB state restored: "
                f"buffer={len(self._mcb._price_buffer)}, "
                f"halts={self._mcb._total_halts}"
            )
        _sad_state = saved_state.sad_state
        if _sad_state and self._sad is not None:
            self._sad.import_state(_sad_state)
            logger.info(
                f"[225#] SAD state restored: "
                f"buffer={len(self._sad._spread_buffer)}, "
                f"frozens={self._sad._total_frozens}"
            )
        # 236# エスカレーション・縮退カウンタ復元
        _duty = saved_state.degraded_liquidation_duty_counter
        if _duty > 0:
            self._degraded_liquidation_duty_counter = _duty
            logger.info(f"[236#] Degraded duty counter restored: {_duty}")
        _ie_duty = saved_state.inventory_escape_duty_counter
        if _ie_duty > 0:
            self._inventory_escape_duty_counter = _ie_duty
            logger.info(f"[269#] Inventory escape duty counter restored: {_ie_duty}")
        _cd = saved_state.one_sided_cooldown_remaining
        if _cd > 0:
            self._one_sided_cooldown_remaining = _cd
            logger.info(f"[236#] One-sided cooldown remaining restored: {_cd}")
        _fr = saved_state.one_sided_freeze_remaining
        if _fr > 0:
            self._one_sided_freeze_remaining = _fr
            logger.info(f"[236#] One-sided freeze remaining restored: {_fr}")
        # 254# 250# P1-4: frozen_side 永続化復元
        _fs = saved_state.one_sided_frozen_side
        if _fs is not None:
            self._one_sided_frozen_side = _fs
            logger.info(f"[254#] One-sided frozen side restored: {_fs}")
        _cnf = saved_state.consecutive_no_feasible
        if _cnf:
            self._consecutive_no_feasible = dict(_cnf)
            logger.info(f"[236#] Consecutive no-feasible restored: {_cnf}")

    # ------------------------------------------------------------------
    # 265# extract: run_continuous 初期化 → _init_run_session
    # ------------------------------------------------------------------
    async def _init_run_session(self) -> "RunSessionState":
        """run_continuous の初期化フェーズ.

        265# extract method: lock 取得, trades health check, レジューム復元,
        state/regime/DD warmup, PnL 累積計算。
        """
        from scripts.v460.lib.event_logger import log_event as _log_event
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
        from ztb.data.trades_health import check_trades_health
        from ztb.metrics.fill_quality import (
            compute_record_pnl_jpy,
            filter_clean_records,
        )

        # 044# 単一起動ロック取得
        self._acquire_lock()

        # 135# P2-09→P1: trades データ健全性チェック (277# 定数命名)
        _TRADES_HEALTH_LOOKBACK_DAYS = 3
        _TRADES_HEALTH_STALE_HOURS = 36.0
        _TRADES_HEALTH_MAX_MISSING = 1
        try:
            th = check_trades_health(
                lookback_days=_TRADES_HEALTH_LOOKBACK_DAYS,
                stale_threshold_hours=_TRADES_HEALTH_STALE_HOURS,
                max_missing_days=_TRADES_HEALTH_MAX_MISSING,
            )
            if not th.healthy:
                logger.warning(f"[trades_health] {th.message}")
                if th.missing_days:
                    logger.warning(
                        "[trades_health] retrain 品質が低下する可能性あり。"
                        "fill_test 内蔵 TradesRecorder の動作状態を確認してください"
                    )
                _log_event(
                    "trades_health_alert",
                    self._results_dir,
                    run_id=self._run_id,
                    git_sha=self._git_sha,
                    reason=f"trades unhealthy: {th.message}",
                    details={
                        "healthy": th.healthy,
                        "latest_day": th.available_days[-1] if th.available_days else None,
                        "missing_days": th.missing_days,
                        "stale_hours": round(th.stale_hours, 1),
                    },
                )
            else:
                logger.info(f"[trades_health] {th.message}")
        except Exception as e:
            logger.warning(f"[trades_health] check failed: {e}")

        # 041# 動的 loss_cap
        if self.config.loss_cap_auto:
            await self._update_dynamic_loss_cap()

        # 101# §4: soft_cap スナップショット
        self._soft_cap_jpy_snapshot = (
            self.config.loss_cap_jpy
            * self.config.soft_loss_cap_ratio
            / self.config.loss_cap_ratio
        )

        # 042# 起動時の滞留注文クリア
        await self._cancel_stale_orders()

        # レジューム
        existing_records = self.resume_from_existing()
        clean_records, quarantine_records = filter_clean_records(existing_records)
        if quarantine_records:
            logger.warning(
                f"[quarantine] {len(quarantine_records)} records excluded from "
                f"PnL computation (blank git_sha)"
            )

        # 088# schema health check
        if not self._run_id or not self._run_id.strip():
            logger.error("[schema_health] CRITICAL: run_id is empty — data quality at risk")
        if not self._git_sha or not self._git_sha.strip():
            logger.error("[schema_health] CRITICAL: git_sha is empty — records will be quarantined")
        else:
            logger.info(
                f"[schema_health] OK: run_id={self._run_id}, git_sha={self._git_sha}, "
                f"clean={len(clean_records)}, quarantine={len(quarantine_records)}"
            )

        st = RunSessionState()
        st.total_count = len(existing_records)
        st.filled_count = sum(1 for r in existing_records if r.filled)
        st.batch_size = self.config.batch_size

        # 033# F4: レジューム時の累積 PnL 計算
        for r in clean_records:
            pnl_jpy = compute_record_pnl_jpy(r)
            if pnl_jpy is not None:
                st.cumulative_pnl_jpy += pnl_jpy
            if r.filled and r.order_quantity is not None:
                _qty = float(r.order_quantity)
                if r.side == "buy":
                    st.cumulative_btc_delta += _qty
                elif r.side == "sell":
                    st.cumulative_btc_delta -= _qty
            if r.filled and r.adverse_selected is True:
                st.cumulative_adverse_count += 1
                if r.post_fill_30s_pnl is not None:
                    st.cumulative_adverse_bps += r.post_fill_30s_pnl

        # 101# §2: soft_loss_cap_triggered レジューム復元
        if existing_records and self.config.loss_cap_auto:
            soft_cap_jpy = (
                self.config.loss_cap_jpy
                * self.config.soft_loss_cap_ratio
                / self.config.loss_cap_ratio
            )
            if st.cumulative_pnl_jpy <= -soft_cap_jpy:
                self._soft_loss_cap_triggered = True
                logger.info(
                    f"[resume] soft_loss_cap already triggered: "
                    f"cumPnL={st.cumulative_pnl_jpy:.0f} JPY <= -{soft_cap_jpy:.0f} JPY"
                )

        # 101# P1-5: regime detector warm-up
        regime_restored = False
        if self._regime_detector is not None:
            saved_state = self._state_persistence.load()
            if saved_state is not None and saved_state.regime_prices:
                regime_restored = self._regime_detector.restore_state({
                    "confirmed": saved_state.regime_confirmed,
                    "stability": saved_state.regime_stability,
                    "prices": saved_state.regime_prices,
                    "raw_history": saved_state.regime_raw_history or [],
                })
            self._restore_common_state(saved_state)
        else:
            saved_state = self._state_persistence.load()
            self._restore_common_state(saved_state)

        # 203# F: DD warmup
        if (
            self._daily_drawdown_guard.enabled
            and existing_records
            and (
                self._daily_drawdown_guard.state.daily_fill_count == 0
                or self._daily_drawdown_guard.needs_warmup_repair()
            )
        ):
            self._warmup_daily_drawdown_from_records(existing_records)

        # 209# H4: Kill manager warmup
        if existing_records and len(self._sell_kill_mgr._pnl_history) == 0:
            self._warmup_kill_managers_from_records(existing_records)

        if self._regime_detector is not None and existing_records and not regime_restored:
            filled_with_mid = [
                r for r in existing_records
                if r.filled and r.mid_at_fill is not None
            ]
            warmup_window = self._regime_detector.config.window * self.config.regime_warmup_multiplier
            warmup_records = filled_with_mid[-warmup_window:]
            for r in warmup_records:
                assert r.mid_at_fill is not None
                self._regime_detector.update(r.timestamp, r.mid_at_fill)
            if warmup_records:
                logger.info(
                    f"[regime] warm-up (fallback): fed {len(warmup_records)} records, "
                    f"regime={self._regime_detector.current_regime.value}"
                )

        del existing_records, clean_records, quarantine_records  # メモリ解放

        st.batch = self._batch_persistence.take_unsaved()
        return st

    # ------------------------------------------------------------------
    # 265# extract: final cleanup → _finalize_run
    # ------------------------------------------------------------------
    async def _finalize_run(
        self,
        st: "RunSessionState",
        heartbeat_task: asyncio.Task[None],
    ) -> list["FillRecord"]:
        """run_continuous の最終クリーンアップ.

        Returns:
            全レコード (リロード済み).
        """
        from ztb.metrics.fill_quality import iter_fill_records_glob

        # 残りバッチを保存
        if st.batch:
            if not self._batch_persistence.try_save_batch(st.batch):
                self._batch_persistence.emergency_dump(st.batch, "final")

        # 最終状態保存
        self._state_persistence.save(self._build_state_snapshot(
            total_count=st.total_count,
            filled_count=st.filled_count,
            cumulative_pnl_jpy=st.cumulative_pnl_jpy,
        ))
        self._last_state_save_time = time.monotonic()

        # heartbeat 停止
        heartbeat_task.cancel()
        self._heartbeat_task = None
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        logger.info(
            f"Fill test completed: {st.total_count} cycles, "
            f"{st.filled_count} filled"
        )
        return list(iter_fill_records_glob(str(self._results_dir)))

    def _cleanup_sync(self) -> None:
        """atexit: 残存注文キャンセル + 未保存データ退避 + ロック解放 (同期 wrapper).

        024# R1: 未保存バッチを緊急ダンプに退避.
        044# A-4: 残存注文キャンセルを確実に実行.
        044# Bug7: ロックファイルを解放.
        129# OB recorder: 最終 flush.
        """
        # 129# OB recorder: バッファ残を書き出し
        try:
            n = self._ob_recorder.flush()
            if n:
                logger.info(f"OB recorder: flushed {n} snapshots on exit")
        except Exception as e:
            logger.error(f"OB recorder final flush failed: {e}")

        # 135# P0-04: trades recorder 最終 flush
        try:
            n_tr = self._trades_recorder.flush()
            if n_tr:
                logger.info(f"Trades recorder: flushed {n_tr} trades on exit")
        except Exception as e:
            logger.error(f"Trades recorder final flush failed: {e}")

        # 未保存バッチの退避
        unsaved = self._batch_persistence.unsaved_batch
        if unsaved:
            logger.warning(
                f"Saving {len(unsaved)} unsaved records on exit"
            )
            self._batch_persistence.emergency_dump(unsaved, "atexit")
            self._batch_persistence.take_unsaved()  # クリア

        # 044# A-4: 残存注文のキャンセル (確実に await する)
        if self._pending_order_id:
            logger.warning(f"Cleaning up pending order: {self._pending_order_id}")
            try:
                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None

                if running_loop is not None and running_loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(
                        self.adapter.cancel_order(self._pending_order_id),
                        running_loop,
                    )
                    try:
                        fut.result(timeout=5.0)
                        logger.info(f"Cancelled pending order: {self._pending_order_id}")
                    except Exception as e2:
                        logger.warning(f"Cleanup via running loop failed: {e2}")
                else:
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(
                            self.adapter.cancel_order(self._pending_order_id)
                        )
                        logger.info(f"Cancelled pending order: {self._pending_order_id}")
                    finally:
                        loop.close()
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")

        # 044# Bug7: ロックファイル解放
        self._release_lock()
