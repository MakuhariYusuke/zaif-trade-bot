"""163# Mixin: FillLoopOrchestratorMixin -- run_continuous + ループ制御.

メインオーケストレーションループ: side 選択, skip chain, adaptation, 状態保存。

WARNING -- AI Coding Agent / 人間開発者への注意:
    このファイルは Mixin クラスであり、単独でインスタンス化しないこと。
    FillTestRunner.__init__ で生成される属性に依存する。
    責務: ループ制御 (side kill, time filter, balance forced, adaptation, cleanup)
    1 サイクルの実行ロジック (発注/約定/PnL) は fill_cycle_executor に属する。
    OB ラッパー / SkipGate 評価を追加しないこと。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.event_logger import log_event as _log_event
from scripts.v460.lib.regime_policy import CycleStrategy
from scripts.v460.lib.resilience import FillTestState
from ztb.data.trades_health import check_trades_health
from ztb.metrics.fill_quality import FillRecord, filter_clean_records, load_fill_records_glob

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig

logger = logging.getLogger(__name__)


class FillLoopOrchestratorMixin:
    """run_continuous + side kill / filter / adaptation / cleanup (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: ループ制御, skip chain 評価, adaptation, 状態保存, cleanup
      NG: 1 サイクル実行, OB 取得, SkipGate 評価, PnL 計測
    MAX LINES: 1200 (超えたら skip chain を独立モジュールに抽出せよ)
    ────────────────────────────────────────────────────
    """

    def _is_sell_killed(self) -> bool:
        """133# P0-10 / 136# P1-03: sell 動的 kill 判定 — SellDynamicKillManager に委譲.

        §9 #3: 現在レジームを check_kill() に渡し regime_thresholds を有効化。
        """
        regime: str | None = None
        if self._regime_detector is not None:
            regime = self._regime_detector.current_regime.value
        killed, telemetry = self._sell_kill_mgr.check_kill(regime=regime)
        if killed:
            logger.info(
                f"[136# §9] sell kill: regime={regime or 'default'}, "
                f"threshold_used={telemetry.threshold_used}, "
                f"cooldown_remaining={telemetry.cooldown_remaining}"
            )
        return killed

    def _track_sell_pnl(self, record: "FillRecord") -> None:
        """133# P0-10 / 136# P1-03: sell fill の PnL を追跡 — SellDynamicKillManager に委譲."""
        if (
            record.filled
            and record.side == "sell"
            and record.post_fill_30s_pnl is not None
        ):
            self._sell_kill_mgr.track(record.post_fill_30s_pnl)

    def _is_buy_killed(self) -> bool:
        """157# §19: buy 動的 kill 判定 — BuyDynamicKillManager に委譲.

        sell_dynamic_kill の buy 側対称版。
        """
        regime: str | None = None
        if self._regime_detector is not None:
            regime = self._regime_detector.current_regime.value
        killed, telemetry = self._buy_kill_mgr.check_kill(regime=regime)
        if killed:
            logger.info(
                f"[157# §19] buy kill: regime={regime or 'default'}, "
                f"threshold_used={telemetry.threshold_used}, "
                f"cooldown_remaining={telemetry.cooldown_remaining}"
            )
        return killed

    def _track_buy_pnl(self, record: "FillRecord") -> None:
        """157# §19: buy fill の PnL を追跡 — BuyDynamicKillManager に委譲."""
        if (
            record.filled
            and record.side == "buy"
            and record.post_fill_30s_pnl is not None
        ):
            self._buy_kill_mgr.track(record.post_fill_30s_pnl)

    # ------------------------------------------------------------------
    # 179# S1: _effective_sleep — regime 応答サイクル間隔の一元化
    # ------------------------------------------------------------------
    async def _effective_sleep(self, *, multiplier: float = 1.0) -> None:
        """179# CycleStrategy に委譲し、regime 別サイクル間隔で sleep.

        skip/halt/error continue 全パスがこのメソッドを経由する。
        - multiplier=1.0 : 通常スキップ
        - multiplier=5.0 : halt (daily drawdown)
        正常サイクル完了パスは rapid_exit ロジックを含むため直接呼ばない。
        """
        regime = self._current_regime_value()
        base = self._cycle_strategy.effective_interval(regime)
        await asyncio.sleep(base * multiplier)

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
        # fill_rate
        if total_count > 0 and filled_count / total_count < policy.fill_rate_floor:
            logger.warning(
                f"[181# stop] fill_rate={filled_count/total_count:.2%} → fallback"
            )
            strategy.activate_fallback(3600.0)
            return
        # avg pnl30 (直近 100 filled)
        records = getattr(self, "_recent_records", getattr(self, "_records", []))
        pnls = [
            r.post_fill_30s_pnl for r in records[-100:]
            if getattr(r, "filled", False) and getattr(r, "post_fill_30s_pnl", None) is not None
        ]
        if len(pnls) >= 10:
            avg = sum(pnls) / len(pnls)
            if avg < policy.pnl_floor_bps:
                logger.warning(f"[181# stop] avg_pnl30={avg:.2f}bps → fallback")
                strategy.activate_fallback(3600.0)

    def _is_time_filtered(self, side: str | None = None) -> bool:
        """時間帯フィルター — 121# TimeFilter に委譲.

        163#: regime 連動動的ゲーティング — current_regime を渡す。
        """
        regime = self._current_regime_value()
        return self._time_filter.is_filtered(side=side, regime=regime)

    # 106# R2: bps 換算定数 — FillRecordHelpersMixin._BPS_FACTOR を MRO 経由で継承

    async def _check_balance_for_side(
        self, side: str, *, regime_mult: float = 1.0,
    ) -> bool:
        """残高 pre-flight check — 121# BalanceChecker に委譲.

        145# §8-#1: regime_mult を渡してレジーム倍率込みで残高判定.
        """
        return await self._balance_checker.check(
            side, self.adapter, self.config.symbol,
            regime_mult=regime_mult,
        )

    # ------------------------------------------------------------------
    # 158# P2-4: Lock 管理 — LockManager に委譲
    # ------------------------------------------------------------------
    def _acquire_lock(self) -> None:
        """044# 単一起動ロック — LockManager に委譲."""
        self._lock_manager.acquire()

    def _release_lock(self) -> None:
        """044# ロックファイル解放 — LockManager に委譲."""
        self._lock_manager.release()

    def _update_lock_heartbeat(self) -> None:
        """129# heartbeat 更新 — LockManager に委譲."""
        self._lock_manager.update_heartbeat()

    async def _cancel_stale_orders(self) -> int:
        """042# 起動時の滞留注文自動クリア.

        前回プロセスが異常終了した際に残った未約定注文をキャンセルする。
        これにより、303s のポーリング浪費を回避。

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

    async def run_continuous(self, hours: float) -> list[FillRecord]:
        """指定時間、連続してサイクルを実行.

        009# §4.4: 7 日間 (168h) の実測想定.
        中断→再開時は既存 fill_records を自動復元 (レジューム対応).

        024# R1-R4: 保存失敗耐性・例外分離・メモリ制御を強化.
        032# P0: 方策 A パラメータ適応統合.
        033# 方策 B: 動的ロットサイジング統合.
        033# F4: 累積 PnL 安全キャップ (000# §3.9).
        """
        end_time = time.time() + hours * 3600

        # 044# 単一起動ロック取得
        self._acquire_lock()

        # 135# P2-09→P1: trades データ健全性チェック
        # 160# fix: max_missing_days=1 で retrain_scheduler と同じ許容レベルに統一
        try:
            th = check_trades_health(
                lookback_days=3,
                stale_threshold_hours=36.0,
                max_missing_days=1,
            )
            if not th.healthy:
                logger.warning(f"[trades_health] {th.message}")
                if th.missing_days:
                    logger.warning(
                        "[trades_health] retrain 品質が低下する可能性あり。"
                        "fill_test 内蔵 TradesRecorder の動作状態を確認してください"
                    )
                # 148# P1: trades stale 自動イベント記録
                # 159# §2.1 fix: latest_ts/age_hours → available_days[-1]/stale_hours
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

        # 041# 動的 loss_cap: API 残高から算出
        if self.config.loss_cap_auto:
            await self._update_dynamic_loss_cap()

        # 101# §4: soft_cap スナップショット — 起動時の残高ベースで固定
        # 動的 loss_cap_jpy が変動しても soft_cap は連動しない
        self._soft_cap_jpy_snapshot = (
            self.config.loss_cap_jpy
            * self.config.soft_loss_cap_ratio
            / self.config.loss_cap_ratio
        )

        # 042# 起動時の滞留注文クリア (前回プロセスの残注文防止)
        await self._cancel_stale_orders()

        # レジューム: 既存レコードから状態復元
        existing_records = self.resume_from_existing()
        # 046# clean/quarantine 分離: ゾンビプロセス由来レコードを除外して集計
        clean_records, quarantine_records = filter_clean_records(existing_records)
        if quarantine_records:
            logger.warning(
                f"[quarantine] {len(quarantine_records)} records excluded from "
                f"PnL computation (blank git_sha)"
            )

        # 088# schema health check: run_id / git_sha の自己検証
        if not self._run_id or not self._run_id.strip():
            logger.error("[schema_health] CRITICAL: run_id is empty — data quality at risk")
        if not self._git_sha or not self._git_sha.strip():
            logger.error("[schema_health] CRITICAL: git_sha is empty — records will be quarantined")
        else:
            logger.info(
                f"[schema_health] OK: run_id={self._run_id}, git_sha={self._git_sha}, "
                f"clean={len(clean_records)}, quarantine={len(quarantine_records)}"
            )
        # 024# O4: メモリ制御 — 全レコード保持ではなくカウンタのみ
        total_count = len(existing_records)  # 全件カウント (quarantine 含む)
        filled_count = sum(1 for r in existing_records if r.filled)

        # 033# F4: レジューム時の累積 PnL 計算 (クリーンレコードのみ)
        cumulative_pnl_jpy = 0.0
        for r in clean_records:
            if r.filled and r.post_fill_30s_pnl is not None and r.fill_price:
                cumulative_pnl_jpy += (
                    r.post_fill_30s_pnl / self._BPS_FACTOR * r.fill_price * r.order_quantity
                )

        # 101# §2: soft_loss_cap_triggered をレジューム復元
        # 前回 run 中に soft cap 発動していた場合、再起動で False に戻ると
        # 二重ロット半減が発生する。cumulative_pnl_jpy から論理的に判定。
        if existing_records and self.config.loss_cap_auto:
            soft_cap_jpy = (
                self.config.loss_cap_jpy
                * self.config.soft_loss_cap_ratio
                / self.config.loss_cap_ratio
            )
            if cumulative_pnl_jpy <= -soft_cap_jpy:
                self._soft_loss_cap_triggered = True
                logger.info(
                    f"[resume] soft_loss_cap already triggered: "
                    f"cumPnL={cumulative_pnl_jpy:.0f} JPY <= -{soft_cap_jpy:.0f} JPY"
                )
        # 101# P1-5: regime detector warm-up — 既存レコードの mid price で初期化
        # window=20 に対して再起動後 20 サイクルは判定不安定になるため、
        # レジューム時の既存レコード (直近 window*3 件) で事前投入する。
        # 121# A4: StatePersistence から regime state を優先復元 (warm-up より正確)
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
            # 168# §4.1 #3: 日次ドローダウンガード状態復元
            if saved_state is not None and saved_state.daily_drawdown_state:
                self._daily_drawdown_guard.import_state(saved_state.daily_drawdown_state)
        else:
            # regime_detector がない場合でも daily_drawdown state は復元
            saved_state = self._state_persistence.load()
            if saved_state is not None and saved_state.daily_drawdown_state:
                self._daily_drawdown_guard.import_state(saved_state.daily_drawdown_state)

        if self._regime_detector is not None and existing_records and not regime_restored:
            # fallback: 旧方式の warm-up (state 復元失敗時)
            filled_with_mid = [
                r for r in existing_records
                if r.filled and r.mid_at_fill is not None
            ]
            # window*multiplier (バッファ上限に合わせる) の直近分だけ投入
            warmup_window = self._regime_detector.config.window * self.config.regime_warmup_multiplier
            warmup_records = filled_with_mid[-warmup_window:]
            for r in warmup_records:
                assert r.mid_at_fill is not None  # filtered above
                self._regime_detector.update(r.timestamp, r.mid_at_fill)
            if warmup_records:
                logger.info(
                    f"[regime] warm-up (fallback): fed {len(warmup_records)} records, "
                    f"regime={self._regime_detector.current_regime.value}"
                )

        del existing_records, clean_records, quarantine_records  # メモリ解放

        batch: list[FillRecord] = self._batch_persistence.take_unsaved()  # 前回未保存分を引き継ぐ
        batch_size = self.config.batch_size  # 032# #18: 設定化

        # 148# P0: heartbeat 更新タスク — stale 誤判定防止
        async def _heartbeat_loop() -> None:
            """lock heartbeat を周期的に更新."""
            while not self._kill_switch.is_killed():
                self._update_lock_heartbeat()
                await asyncio.sleep(self.config.lock_heartbeat_period_sec)

        heartbeat_task = asyncio.create_task(_heartbeat_loop())
        self._heartbeat_task: asyncio.Task[None] | None = heartbeat_task  # 175# cleanup 用

        logger.info(f"Starting fill test: {hours}h, interval={self.config.cycle_interval_sec}s")

        while time.time() < end_time and not self._kill_switch.is_killed():
            # 168# §4.1 #3: 日次ドローダウンガード — halt 中はスキップ
            if self._daily_drawdown_guard.is_halted():
                # 日次 PnL 超過 → UTC 日替わりまでスキップ
                batch.append(self._make_skip_record(
                    side="none",
                    cancel_reason=CR.DAILY_DRAWDOWN_HALT,
                    order_quantity=0.0,
                    regime=self._current_regime_value(),
                ))
                total_count += 1
                batch = self._batch_persistence.maybe_flush(batch, "daily_drawdown_halt")
                self._update_lock_heartbeat()
                await self._effective_sleep(multiplier=5.0)  # 179# S1: halt 中は 5x 間隔
                continue

            # 129# D.2: 残高制約による side 強制切替追跡
            _balance_forced = False
            _is_rescue = False  # 158# P1-1: balance_forced rescue フラグ
            # 073# side 別時間帯フィルター: side 決定後にフィルタリング
            # side 別リスト未設定時はグローバルリスト (041# 互換)
            next_side = self._next_side()

            # side 別チェック (073#): side固有リストがあれば side 別判定
            side_filtered = self._is_time_filtered(side=next_side)
            if side_filtered:
                # 反対 side でもフィルタされるか確認
                alt_side = "sell" if next_side == "buy" else "buy"
                alt_filtered = self._is_time_filtered(side=alt_side)
                if alt_filtered:
                    # 両 side ともフィルタ → スリープ
                    # 140# §8.1-#2: skip record を生成し可観測性確保 (132# F4)
                    if not self._time_filter.in_filter:
                        self._time_filter.on_enter()
                        batch.append(self._make_skip_record(
                            side=next_side,
                            cancel_reason=CR.TIME_FILTER_BOTH_SIDES,
                            order_quantity=0.0,
                            regime=self._current_regime_value(),  # 160#
                        ))
                    else:
                        # 079# heartbeat: 長時間抑制中にプロセス生存を定期ログ
                        now_ts = time.time()
                        if now_ts - self._time_filter.last_heartbeat_time >= self.config.heartbeat_interval_sec:
                            utc_h = datetime.now(timezone.utc).hour
                            try:
                                import psutil  # lazy import
                                proc = psutil.Process()
                                mem_mb = proc.memory_info().rss / (1024 * 1024)
                                mem_info = f"mem={mem_mb:.1f}MB, "
                            except Exception:
                                mem_info = ""
                            logger.info(
                                f"[heartbeat] Still in time_filter zone "
                                f"(UTC {utc_h}h), "
                                f"{mem_info}"
                                f"unsaved_batch={len(batch)}, "
                                f"cycles={self._cycle_count}"
                            )
                            self._time_filter.last_heartbeat_time = now_ts
                            # 129# lock heartbeat 更新
                            self._update_lock_heartbeat()
                        # 107# R1: 重複 flush → _maybe_flush_batch 統合
                        batch = self._batch_persistence.maybe_flush(batch, "time_filter")
                    await self._effective_sleep()  # 179# S1
                    continue
                else:
                    # 反対 side は通過 → side 切り替え
                    # 086# Bug: alt_side が _last_side と同じ場合、片側蓄積が発生する
                    # (例: _last_side=buy, next=sell がブロック, alt=buy → double buy)
                    # この場合は両方ブロックと同じ扱いにして待機する
                    if alt_side == self._last_side:
                        self._time_filter.consecutive_086_wait += 1
                        max_wait = self.config.max_086_consecutive_wait
                        utc_h = datetime.now(timezone.utc).hour
                        # 110# デッドロック解除: 連続待機が上限を超えたら alt_side を許可
                        if max_wait > 0 and self._time_filter.consecutive_086_wait > max_wait:
                            logger.info(
                                f"[time_filter] 086# deadlock break: "
                                f"{self._time_filter.consecutive_086_wait} consecutive waits "
                                f"exceeded max={max_wait}, allowing {alt_side} "
                                f"(110# デッドロック解除)"
                            )
                            self._time_filter.consecutive_086_wait = 0
                            next_side = alt_side
                            # ↓ alt_side 許可 → 通常フローに合流
                        else:
                            logger.info(
                                f"[time_filter] {next_side} filtered at UTC {utc_h}h, "
                                f"alt={alt_side} would repeat last side → "
                                f"treating as both-filtered "
                                f"(086# 片側蓄積防止, wait={self._time_filter.consecutive_086_wait}/{max_wait})"
                            )
                            if not self._time_filter.in_filter:
                                self._time_filter.on_enter()
                                # 140# §8.1-#2: 086 deadlock 進入時も record 生成
                                batch.append(self._make_skip_record(
                                    side=next_side,
                                    cancel_reason=CR.TIME_FILTER_086_DEADLOCK,
                                    order_quantity=0.0,
                                    regime=self._current_regime_value(),  # 160#
                                ))
                            # 107# R1: 重複 flush → _maybe_flush_batch 統合
                            batch = self._batch_persistence.maybe_flush(batch, "alt_side==last_side wait")
                            await self._effective_sleep()  # 179# S1
                            continue
                    else:
                        # 086# ではない通常の side 切り替え → カウンタリセット
                        self._time_filter.consecutive_086_wait = 0
                    utc_h = datetime.now(timezone.utc).hour
                    logger.debug(
                        f"[time_filter] {next_side} filtered at UTC {utc_h}h, "
                        f"switching to {alt_side}"
                    )
                    next_side = alt_side

            # 047# Issue12: 離脱時のみログ出力
            self._time_filter.on_exit()

            # 158# §20-A: メインループ毎にレジーム更新 — skip パスでも遷移を保証
            # Root cause fix: trending_sell_skip 等のスキップパスでは
            # run_single_cycle が呼ばれず regime_detector.update() が滞留→デッドロック
            # fallback price (直近 OB mid) を毎ループ投入して遷移機会を確保
            if self._regime_detector is not None:
                _fb_price, _fb_time = self._maker_price.get_fallback_price()
                if _fb_price is not None:
                    _pre_regime = self._regime_detector.current_regime
                    _regime_result = self._regime_detector.update(
                        time.time(), _fb_price
                    )
                    if _regime_result.regime != _pre_regime:
                        logger.info(
                            f"[158# §20-A] Regime transition in main loop: "
                            f"{_pre_regime.value} → {_regime_result.regime.value} "
                            f"(stability={_regime_result.stability}, "
                            f"trend_pct={_regime_result.trend_pct:.4f})"
                        )

            # 041# 残高 pre-flight check: 不足サイドはスキップ
            # 145# §8-#1: レジーム倍率込みで残高判定 (preflight-lot alignment)
            _regime_mult = self._regime_lot_multiplier()
            if await self._check_balance_for_side(next_side, regime_mult=_regime_mult):
                # 091# 即座に反対 side を試す: time_filter との組合せで停滞するのを防止
                opposite = "sell" if next_side == "buy" else "buy"
                tried_opposite = False
                if not await self._check_balance_for_side(opposite, regime_mult=_regime_mult):
                    # 反対 side は残高 OK → 即座に切替
                    logger.info(
                        f"[balance] {next_side} insufficient, "
                        f"switching to {opposite} immediately (091#)"
                    )
                    # 120# A5: 不足 side を N サイクル凍結 (API 呼出し節約)
                    # 158# YAML 外部化: balance_freeze_cycles
                    self._side_selector.freeze_side(
                        next_side, cycles=self.config.balance_freeze_cycles,
                    )
                    next_side = opposite
                    self._last_side = opposite  # 次回は再び元の side
                    self._preflight_skip_count = 0
                    tried_opposite = True
                    _balance_forced = True  # 129# D.2

                if not tried_opposite:
                    # 両 side とも残高不足 → 従来通りの処理
                    self._last_side = next_side  # → 次の _next_side() が反対を返す
                    self._preflight_skip_count += 1

                    # 140# §8.1-#2: preflight skip record 生成 (132# F4)
                    # 145# §9-#5: _make_skip_record DRY 化
                    batch.append(self._make_skip_record(
                        side=next_side,
                        cancel_reason=CR.PREFLIGHT_INSUFFICIENT,
                        order_quantity=self._current_lot,
                        regime=self._current_regime_value(),  # 160#
                    ))
                    # 107# R1: 重複 flush → _maybe_flush_batch 統合
                    batch = self._batch_persistence.maybe_flush(batch, "preflight skip")

                    # 051# P2-3: Balance auto-shrink — 連続失敗でロット縮小を試行
                    # 052#: 最低ロットを min_order_btc に統一 (Coincheck 0.001 BTC)
                    min_lot = max(self.config.order_quantity, self.config.min_order_btc)
                    if (
                        self._preflight_skip_count >= self.config.balance_shrink_consecutive
                        and not self._balance_checker.balance_shrink_active
                        and self._current_lot > min_lot
                    ):
                        old_lot = self._current_lot
                        # 105#: 0.001 BTC 単位に切り捨て (浮動小数点丸め誤差 → API 400 防止)
                        raw_shrunk = self._current_lot / self.config.balance_shrink_divisor
                        _mob = self.config.min_order_btc
                        self._current_lot = max(
                            min_lot,
                            int(raw_shrunk / _mob) * _mob,
                        )
                        self._balance_checker.balance_shrink_active = True
                        logger.warning(
                            f"[balance_shrink] 連続 preflight 失敗 {self._preflight_skip_count} 回. "
                            f"ロット縮小: {old_lot:.4f} → {self._current_lot:.4f} BTC"
                        )
                        # カウンタリセットして縮小ロットで再試行
                        self._preflight_skip_count = 0
                        await self._effective_sleep()  # 179# S1
                        continue

                    # 138# P1-10: preflight pause — SAFE_STOP 前に一時停止で回復を待つ
                    if (
                        self.config.preflight_pause_enabled
                        and self._preflight_skip_count >= self.config.preflight_pause_threshold
                        and self._preflight_pause_count < self.config.preflight_max_pauses
                    ):
                        self._preflight_pause_count += 1
                        pause_sec = self.config.preflight_pause_sec
                        logger.warning(
                            f"[preflight_pause] 連続 preflight 失敗 {self._preflight_skip_count} 回 "
                            f"(閾値 {self.config.preflight_pause_threshold}). "
                            f"pause #{self._preflight_pause_count}/{self.config.preflight_max_pauses} "
                            f"→ {pause_sec:.0f}s 待機後に再開"
                        )
                        # 140# §8.1-#1: batch.append 導線に統一 (undefined _append_fill_record 修正)
                        # 143# 140§7 #2: cycle_id に timestamp 付与で一意化
                        # 145# §9-#5: _make_skip_record DRY 化
                        batch.append(self._make_skip_record(
                            side="none",
                            cancel_reason=CR.PREFLIGHT_PAUSE,
                            cycle_id=f"preflight_pause_{self._preflight_pause_count}_{int(time.time())}",
                            order_quantity=0.0,
                            regime=self._current_regime_value(),  # 160#
                        ))
                        batch = self._batch_persistence.maybe_flush(batch, "preflight_pause")
                        self._preflight_skip_count = 0
                        await asyncio.sleep(pause_sec)
                        continue

                    # 044# F8: 連続 preflight 失敗上限 → SAFE_STOP
                    if self._preflight_skip_count >= self.config.max_preflight_skip:
                        logger.error(
                            f"SAFE_STOP: 連続 preflight スキップ {self._preflight_skip_count} 回 "
                            f"(上限 {self.config.max_preflight_skip}). "
                            f"buy/sell 両方で残高不足の可能性. 停止します."
                        )
                        self._kill_switch.kill("preflight_skip_exceeded")
                        break
                    await self._effective_sleep()  # 179# S1
                    continue

            # preflight 成功 → カウンタリセット
            self._preflight_skip_count = 0
            # 051# P2-3: 成功時に balance_shrink を解除し、ロットを原値に復元
            self._balance_checker.restore_lot_on_success()
            # 120# A5: 残高回復 → freeze 解除
            self._side_selector.unfreeze_side()

            # --- サイクル実行 ---
            # 133# P0-08: balance_forced_switch 時のハードスキップ
            # 154# C-1/C-2: deadlock 防止 — 片側残高のみの場合は forced でも実行許可
            if _balance_forced and self.config.skip_balance_forced:
                # 154# C-1: 元の side (forced 前) も残高不足かチェック
                # → 両方不足 = preflight で弾かれるはずなので到達しない
                # → 元 side だけ不足 = forced side が唯一の選択肢 → 実行すべき
                original_side = "buy" if next_side == "sell" else "sell"
                original_also_insufficient = await self._check_balance_for_side(
                    original_side, regime_mult=_regime_mult
                )
                # 154# C-2: 連続 forced skip カウンタによるフォールバック
                _deadlock_limit = self.config.balance_forced_deadlock_limit
                _over_deadlock_limit = (
                    _deadlock_limit > 0
                    and self._balance_forced_skip_count >= _deadlock_limit
                )

                if original_also_insufficient or _over_deadlock_limit:
                    # 片側しか取引できない or デッドロック上限超過 → 実行許可
                    _reason = (
                        "one_sided_balance" if original_also_insufficient
                        else f"deadlock_limit({self._balance_forced_skip_count})"
                    )
                    logger.info(
                        f"[154# C-1] balance_forced but {_reason} — "
                        f"proceeding with {next_side} (original_side={original_side} "
                        f"insufficient={original_also_insufficient})"
                    )
                    self._balance_forced_skip_count = 0
                    # → continue しない: run_single_cycle へ進む
                elif self.config.balance_forced_rescue_enabled:
                    # 158# P1-1: rescue モード — skip せず offset 倍増で安全実行
                    _prev_skip_count = self._balance_forced_skip_count  # 173# ログ用に退避
                    self._balance_forced_skip_count = 0
                    _is_rescue = True  # run_single_cycle に渡すフラグ
                    logger.info(
                        f"[158# P1-1] balance_forced rescue mode: "
                        f"executing {next_side} with offset ×"
                        f"{self.config.balance_forced_rescue_offset_mult:.1f} "
                        f"(was consecutive skip={_prev_skip_count})"
                    )
                    # → continue しない: run_single_cycle へ進む (rescue=True)
                else:
                    # 両方残高 OK → 従来通りスキップ (forced switch は損失回避のため)
                    self._balance_forced_skip_count += 1
                    logger.info(
                        f"[133# P0-08] Skipping cycle — balance_forced_switch=True "
                        f"(avg -1.98bps loss). side={next_side}, "
                        f"consecutive={self._balance_forced_skip_count}"
                    )
                    # 145# §9-#5: _make_skip_record DRY 化
                    _skip_record = self._make_skip_record(
                        side=next_side,
                        cancel_reason=CR.BALANCE_FORCED_SKIP,
                        order_quantity=self._current_lot,
                        balance_forced_switch=True,
                        balance_forced_consecutive=self._balance_forced_skip_count,
                        regime=self._current_regime_value(),  # 160#
                    )
                    batch.append(_skip_record)
                    total_count += 1
                    batch = self._batch_persistence.maybe_flush(batch, "balance_forced_skip")
                    # 167# DL-5: _last_side を更新 (rescue=true 時は到達しないが防御的に)
                    self._last_side = next_side
                    await self._effective_sleep()  # 179# S1
                    continue

            # 133# P0-09: unknown regime での buy スキップ
            # 156# §12: balance_forced 時はバイパス (片側残高で buy するしかない局面)
            if (
                self.config.skip_buy_unknown_regime
                and next_side == "buy"
                and not _balance_forced
                and self._regime_detector is not None
                and self._regime_detector.current_regime.value == "unknown"
            ):
                logger.info(
                    f"[133# P0-09] Skipping buy — unknown regime "
                    f"(avg -1.384bps loss)"
                )
                # 145# §9-#5: _make_skip_record DRY 化
                _skip_record = self._make_skip_record(
                    side="buy",
                    cancel_reason=CR.UNKNOWN_REGIME_BUY_SKIP,
                    order_quantity=self._current_lot,
                    regime="unknown",
                )
                batch.append(_skip_record)
                total_count += 1
                batch = self._batch_persistence.maybe_flush(batch, "unknown_buy_skip")
                # 166# deadlock fix: _last_side を更新して sell 側も試行可能に
                self._last_side = "buy"
                await self._effective_sleep()  # 179# S1
                continue

            # 169# B1': ranging_buy at low_vol ハードスキップ (Gemini 10.2-D「休むも相場」)
            # ranging_buy は全損失の 69% (-220.84 JPY)。低ボラ環境ではスプレッドエッジが
            # 極小化し offset 調整では対処困難 → 完全スキップが clean
            # balance_forced 時はバイパス (片側残高で buy するしかない局面)
            if (
                self.config.skip_ranging_buy_low_vol
                and next_side == "buy"
                and not _balance_forced
                and self._regime_detector is not None
                and self._regime_detector.current_regime.value == "ranging"
            ):
                _vol_ratio = self._regime_detector.last_volatility_ratio
                if _vol_ratio < self.config.low_vol_threshold:
                    logger.info(
                        f"[169# B1'] Skipping buy — ranging regime at low vol "
                        f"(vol_ratio={_vol_ratio:.4f} < threshold={self.config.low_vol_threshold})"
                    )
                    _skip_record = self._make_skip_record(
                        side="buy",
                        cancel_reason=CR.RANGING_LOW_VOL_SKIP,
                        order_quantity=self._current_lot,
                        regime="ranging",
                    )
                    batch.append(_skip_record)
                    total_count += 1
                    batch = self._batch_persistence.maybe_flush(batch, "ranging_low_vol_skip")
                    self._last_side = "buy"
                    await self._effective_sleep()  # 179# S1
                    continue

            # 155# §9: trending レジーム時の sell 抑制
            # trending sell avg -0.687 bps — 逆行トレード削減
            # 156# §10 #1: balance_forced 時はバイパス (片側残高で sell するしかない局面)
            # 156# D-4: trending 方向別分解 — trending_down sell は開放可能
            # 158# §20-B: 連続 skip 安全弁 — max_consecutive 超過で強制許可
            if (
                self.config.skip_sell_trending
                and next_side == "sell"
                and not _balance_forced
                and self._regime_detector is not None
                and self._regime_detector.current_regime.is_trending
            ):
                # D-4: trending_up_only モードならば trending_down は通過させる
                # 176# A: TRENDING (方向不明) も trending_up ではない → 通過
                _current_regime = self._regime_detector.current_regime
                _should_skip = True
                if (
                    self.config.skip_sell_trending_up_only
                    and _current_regime.value != "trending_up"
                ):
                    logger.debug(
                        f"[176# A] Allowing sell in {_current_regime.value} regime "
                        f"(skip_sell_trending_up_only=True, only trending_up is blocked)"
                    )
                    _should_skip = False

                # 158# §20-B: 連続 skip 安全弁
                _max_consec = self.config.max_consecutive_trending_sell_skip
                if (
                    _should_skip
                    and _max_consec > 0
                    and self._trending_sell_skip_count >= _max_consec
                ):
                    logger.warning(
                        f"[158# §20-B] Trending sell skip safety valve: "
                        f"{self._trending_sell_skip_count} consecutive skips "
                        f">= limit {_max_consec} — forcing sell execution"
                    )
                    self._trending_sell_skip_count = 0
                    _should_skip = False

                # 166# HF4: 資金枯渇リバランス緩和
                # 8.2A: buy 側が残高不足の場合、sell スキップを中断してリバランス
                if _should_skip:
                    _buy_insufficient = await self._check_balance_for_side(
                        "buy", regime_mult=_regime_mult
                    )
                    if _buy_insufficient:
                        logger.info(
                            f"[166# HF4] Trending sell skip relaxation: "
                            f"buy side insufficient  forcing sell for rebalance "
                            f"(was consecutive={self._trending_sell_skip_count})"
                        )
                        self._trending_sell_skip_count = 0
                        _should_skip = False

                # 171# Guard Paradox 対策: 在庫が buy 偏重なら sell 抑制を解除
                # (sell ガード過剰 → buy 偏重 → balance_forced_skip の正フィードバックを断切)
                if _should_skip:
                    _inv_bypass = self.config.sell_guard_inv_bypass_threshold
                    _inv_imb = self._maker_price.inv_net_imbalance
                    if _inv_bypass > 0 and _inv_imb >= _inv_bypass:
                        logger.info(
                            f"[171#] Trending sell skip relaxation: "
                            f"inv_imbalance={_inv_imb:.3f} >= "
                            f"threshold={_inv_bypass} — "
                            f"forcing sell for inventory rebalance"
                        )
                        self._trending_sell_skip_count = 0
                        _should_skip = False

                if _should_skip:
                    self._trending_sell_skip_count += 1
                    logger.info(
                        f"[155# §9] Skipping sell — {_current_regime.value} regime "
                        f"(avg -0.687bps loss) "
                        f"[consecutive={self._trending_sell_skip_count}"
                        f"/{_max_consec if _max_consec > 0 else '∞'}]"
                    )
                    _skip_record = self._make_skip_record(
                        side="sell",
                        cancel_reason=CR.TRENDING_SELL_SKIP,
                        order_quantity=self._current_lot,
                        regime=_current_regime.value,
                    )
                    batch.append(_skip_record)
                    total_count += 1
                    batch = self._batch_persistence.maybe_flush(batch, "trending_sell_skip")
                    # 167# DL-4: _last_side を更新して buy 側も試行可能に
                    self._last_side = next_side  # = "sell"
                    await self._effective_sleep()  # 179# S1
                    continue
                else:
                    # skip しない → カウンタリセット
                    self._trending_sell_skip_count = 0

            # 157# §19: buy 動的 kill — rolling PnL が閾値以下なら buy 停止
            # balance_forced 時はバイパス (sell_dynamic_kill と対称)
            if (
                self.config.buy_dynamic_kill_enabled
                and next_side == "buy"
                and not _balance_forced
                and self._is_buy_killed()
            ):
                logger.info(
                    f"[157# §19] Skipping buy — rolling PnL below "
                    f"{self.config.buy_dynamic_kill_threshold_bps}bps"
                )
                _skip_record = self._make_skip_record(
                    side="buy",
                    cancel_reason=CR.BUY_DYNAMIC_KILL,
                    order_quantity=self._current_lot,
                    regime=self._current_regime_value(),  # 160#
                )
                batch.append(_skip_record)
                total_count += 1
                batch = self._batch_persistence.maybe_flush(batch, "buy_dynamic_kill")
                # 166# deadlock fix: _last_side を更新して sell 側も試行可能に
                self._last_side = "buy"
                await self._effective_sleep()  # 179# S1
                continue

            # 133# P0-10: sell 動的 kill — rolling PnL が閾値以下なら sell 停止
            # 156# §12: balance_forced 時はバイパス (片側残高で sell するしかない局面)
            # 171# Guard Paradox 対策: 在庫 buy 偏重時もバイパス (sell 抑制の正フィードバック断切)
            _inv_bypass_sell_kill = (
                self.config.sell_guard_inv_bypass_threshold > 0
                and self._maker_price.inv_net_imbalance
                    >= self.config.sell_guard_inv_bypass_threshold
            )
            if (
                self.config.sell_dynamic_kill_enabled
                and next_side == "sell"
                and not _balance_forced
                and not _inv_bypass_sell_kill
                and self._is_sell_killed()
            ):
                logger.info(
                    f"[133# P0-10] Skipping sell — rolling PnL below "
                    f"{self.config.sell_dynamic_kill_threshold_bps}bps"
                )
                # 145# §9-#5: _make_skip_record DRY 化
                _skip_record = self._make_skip_record(
                    side="sell",
                    cancel_reason=CR.SELL_DYNAMIC_KILL,
                    order_quantity=self._current_lot,
                    regime=self._current_regime_value(),  # 160#
                )
                batch.append(_skip_record)
                total_count += 1
                batch = self._batch_persistence.maybe_flush(batch, "sell_dynamic_kill")
                # 166# deadlock fix: _last_side を更新して buy 側も試行可能に
                self._last_side = "sell"
                await self._effective_sleep()  # 179# S1
                continue

            try:
                record = await self.run_single_cycle(
                    side_override=next_side,
                    balance_forced_switch=_balance_forced,
                    balance_forced_rescue=_is_rescue,
                )
                # 154# C-2: 実サイクル実行 → forced skip カウンタリセット
                self._balance_forced_skip_count = 0
                # 158# §20-B: 実サイクル実行 → trending sell skip カウンタリセット
                self._trending_sell_skip_count = 0
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt — stopping gracefully")
                self._kill_switch.kill("keyboard_interrupt")
                break
            except Exception as e:
                # 024# R2: 例外分類 — サイクル実行エラーは継続可能
                logger.error(f"Cycle execution error: {e}", exc_info=True)
                # 128# 例外時も dust sweep ロットを復元
                self._balance_checker.restore_lot_after_dust_sweep()
                # 166# SR-4: 例外 continue でも side 交互を保証
                self._last_side = next_side
                await self._effective_sleep()  # 179# S1
                continue

            # 128# dust sweep 後のロット復元 (サイクル完了ごとに確実に実行)
            self._balance_checker.restore_lot_after_dust_sweep()

            total_count += 1
            if record.filled:
                filled_count += 1
                # 133# P0-10: sell PnL 追跡 (動的 kill 判定用)
                self._track_sell_pnl(record)
                # 157# §19: buy PnL 追跡 (動的 kill 判定用)
                self._track_buy_pnl(record)
                # 033# F4: 累積 PnL インクリメンタル追跡
                if record.post_fill_30s_pnl is not None and record.fill_price:
                    cumulative_pnl_jpy += (
                        record.post_fill_30s_pnl / self._BPS_FACTOR
                        * record.fill_price * record.order_quantity
                    )
                # 168# §4.1 #3: 日次ドローダウンガード PnL 更新
                if record.post_fill_30s_pnl is not None:
                    dd_result = self._daily_drawdown_guard.update_pnl(
                        record.post_fill_30s_pnl,
                    )
                    if dd_result.get("soft_triggered"):
                        old_lot = self._current_lot
                        self._current_lot = max(
                            self.config.order_quantity,
                            self._current_lot / 2,
                        )
                        self._balance_checker.pre_shrink_lot = self._current_lot
                        logger.warning(
                            f"[daily_drawdown] soft lot reduction: "
                            f"{old_lot:.4f} → {self._current_lot:.4f} BTC"
                        )
            batch.append(record)

            # --- 046# soft/hard 二段 loss_cap ---
            # soft cap: ロット半減 (一度だけ)
            # 101# §4: _soft_cap_jpy_snapshot を使用 (動的 loss_cap_jpy に連動させない)
            if self.config.loss_cap_auto and not self._soft_loss_cap_triggered:
                if self._soft_cap_jpy_snapshot is not None:
                    soft_cap_jpy = self._soft_cap_jpy_snapshot
                else:
                    soft_cap_jpy = (
                        self.config.loss_cap_jpy
                        * self.config.soft_loss_cap_ratio
                        / self.config.loss_cap_ratio
                    )
                if cumulative_pnl_jpy <= -soft_cap_jpy:
                    old_lot = self._current_lot
                    self._current_lot = max(
                        self.config.order_quantity,  # 最小ロットは下回らない
                        self._current_lot / self.config.soft_loss_cap_lot_divisor,
                    )
                    self._soft_loss_cap_triggered = True
                    # 051# P2-3: shrink 復元先も更新
                    self._balance_checker.pre_shrink_lot = self._current_lot
                    logger.warning(
                        f"[loss_cap] SOFT CAP: cumPnL={cumulative_pnl_jpy:.0f} JPY "
                        f"<= -{soft_cap_jpy:.0f} JPY "
                        f"({self.config.soft_loss_cap_ratio:.0%}). "
                        f"ロット半減: {old_lot:.4f} → {self._current_lot:.4f} BTC"
                    )

            # hard cap: SAFE_STOP (既存 033# F4)
            if cumulative_pnl_jpy <= -self.config.loss_cap_jpy:
                logger.error(
                    f"LOSS CAP REACHED (HARD): cumulative PnL = {cumulative_pnl_jpy:.0f} JPY "
                    f"(cap = -{self.config.loss_cap_jpy:.0f} JPY). Stopping fill test."
                )
                self._kill_switch.kill("hard_loss_cap")

            # --- 100# 即約定防御: FastFillDefense クラスに委譲 ---
            # P0-5: side-aware (sell boost が buy に伝播しない)
            # P0-3: two-layer neg_edge detection (即時 proxy + post-fill PnL)
            # P1-2: side 別 base_offset_ratio による cap
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

            # --- バッチ保存 (024# R1: 独立 try/except) ---
            if len(batch) >= batch_size:
                if self._batch_persistence.try_save_batch(batch):
                    batch = []
                    self._batch_persistence.reset_flush_timer()
                    self._adaptation_engine.invalidate_cache()  # 120# TTL キャッシュ無効化
                # 失敗時: batch は保持 → 次回再試行
            # 079# 時間ベース定期flush: batch_size 未満でも一定時間経過で保存
            else:
                batch = self._batch_persistence.maybe_flush(batch, "run_loop")

            # 進捗ログ
            if self._cycle_count % self.config.progress_log_interval == 0:
                regime_tag = (
                    self._regime_detector.current_regime.value
                    if self._regime_detector else "n/a"
                )
                _fill_rate_pct = (
                    filled_count / total_count * 100.0
                    if total_count > 0 else 0.0
                )
                logger.info(
                    f"Progress: {self._cycle_count} cycles, "
                    f"fill rate={filled_count}/{total_count} "
                    f"({_fill_rate_pct:.1f}%), "
                    f"cumPnL={cumulative_pnl_jpy:.1f}JPY, "
                    f"lot={self._current_lot:.4f}BTC, "
                    f"regime={regime_tag}, "
                    f"unsaved_batch={len(batch)}"
                )

            # 113# resilience: HealthMonitor 定期チェック + GC
            health_status = self._health_monitor.maybe_check(self._cycle_count)
            if health_status and health_status.get("level") == "critical":
                logger.error(
                    f"[resilience] Health CRITICAL at cycle {self._cycle_count}: "
                    f"{health_status}"
                )
            self._health_monitor.maybe_gc()

            # 113# resilience: 状態永続化 (progress_log_interval ごと)
            if self._cycle_count % self.config.progress_log_interval == 0:
                # 129# lock heartbeat 更新 (state 保存と同期)
                self._update_lock_heartbeat()
                self._state_persistence.save(FillTestState(
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
                    **self._get_regime_state_fields(),
                ))

            # --- 044# A-7: loss_cap 定期更新 (残高変動を反映) ---
            if (
                self.config.loss_cap_auto
                and self._cycle_count % self._loss_cap_update_interval == 0
                and self._cycle_count > 0
            ):
                await self._update_dynamic_loss_cap()

            # --- 032# P0: 方策 A パラメータ適応 ---
            if (
                self.config.enable_auto_adapt
                and self._cycle_count % self.config.adapt_interval_cycles == 0
                and total_count >= self.config.min_adapt_samples
            ):
                self._try_auto_adapt(total_count, filled_count)

            # --- 033# 方策 B: 動的ロットサイジング ---
            if (
                self.config.enable_dynamic_lot
                and self._cycle_count % self.config.lot_adapt_interval_cycles == 0
                and total_count >= self.config.min_adapt_samples
            ):
                self._try_auto_lot_size()

            # --- 181# 停止条件モニター: C/D/Chase 安全弁 ---
            if (
                hasattr(self, "_cycle_strategy")
                and self._cycle_count > 0
                and self._cycle_count % 30 == 0  # ~1h@120s, ~30min@60s
            ):
                self._check_regime_stop_conditions(filled_count, total_count)

            # 次サイクルまで待機
            # 054# S3: rapid exit 時は interval を短縮
            if time.time() < end_time and not self._kill_switch.is_killed():
                # 169# Config Hot-Reload: サイクル間で YAML 変更を検出・反映
                self._config_reloader.maybe_reload(self)

                if self._side_selector.rapid_exit_side is not None:
                    interval = self.config.early_exit_rapid_interval_sec
                    logger.info(
                        f"[early_exit] Rapid exit: interval shortened to "
                        f"{interval:.0f}s (next side={self._side_selector.rapid_exit_side})"
                    )
                else:
                    # 179# S1: regime 別サイクル間隔
                    regime = self._current_regime_value()
                    interval = self._cycle_strategy.effective_interval(regime)
                await asyncio.sleep(interval)

        # 残りバッチを保存
        if batch:
            if not self._batch_persistence.try_save_batch(batch):
                # 最終手段: 緊急ダンプ
                self._batch_persistence.emergency_dump(batch, "final")

        # 113# resilience: 最終状態保存
        self._state_persistence.save(FillTestState(
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
            **self._get_regime_state_fields(),
        ))

        # 148# heartbeat タスク終了
        heartbeat_task.cancel()
        self._heartbeat_task = None
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        logger.info(
            f"Fill test completed: {total_count} cycles, "
            f"{filled_count} filled"
        )
        # 024# O4: 集計用に全レコードをリロード
        return load_fill_records_glob(str(self._results_dir))

    async def cleanup_heartbeat(self) -> None:
        """175# 異常終了時の heartbeat タスク cleanup.

        run_continuous の呼び出し元で finally ブロックから呼ぶことで、
        未処理例外発生時の heartbeat タスクリークを防止する。
        """
        task = getattr(self, "_heartbeat_task", None)
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
        # 161# fix: 既存ループ検出 + フォールバックで asyncio anti-pattern 回避
        if self._pending_order_id:
            logger.warning(f"Cleaning up pending order: {self._pending_order_id}")
            try:
                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None

                if running_loop is not None and running_loop.is_running():
                    # ループ実行中 — future でスケジュール (best effort)
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
                    # ループなし — 新規ループで実行 (atexit 時の標準パス)
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
