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
import logging
import time
from typing import TYPE_CHECKING, Optional

from scripts.v460.lib import cancel_reasons as CR

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig
    from ztb.metrics.fill_quality import FillRecord

logger = logging.getLogger(__name__)


class FillLoopOrchestratorMixin:
    """run_continuous + side kill / filter / adaptation / cleanup (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: ループ制御, skip chain 評価, adaptation, 状態保存, cleanup
      NG: 1 サイクル実行, OB 取得, SkipGate 評価, PnL 計測
    194#: per-cycle skip chain は CycleGateAggregator に集約
    MAX LINES: 1200 (194# で 1309→1172 に削減済み)
    ────────────────────────────────────────────────────
    """

    # 201# review: 動的属性のクラスレベル宣言 (mypy 検出 + IDE 補完)
    _soft_drawdown_interval_multiplier: float = 1.0
    _halt_start_cycle: int | None = None
    _last_balance_forced_time: float = 0.0
    _balance_forced_freq_count: int = 0
    # 209# M6: 動的生成されていた属性をクラスレベルに宣言
    _in_hard_skip_hour: bool = False
    _halt_iter_count: int = 0
    # 202# A: 単一サイクル大損失クールダウン乗数 (次サイクルのみ有効)
    _loss_cooldown_mult: float = 1.0
    # 205# §9.2: Toxic Fill 同一サイド拒否 — side → 残存拒否サイクル数
    _toxic_veto: dict[str, int] | None = None
    # 207# §4: one-sided 連続実行カウンタ (205# §4.2 Codex 対応)
    _one_sided_consecutive_count: int = 0
    # 215# P0-C: alert_mode.json オーバーライド (サイクル先頭で更新)
    _alert_offset_mult: float = 1.0
    _alert_lot_mult: float = 1.0
    _alert_interval_mult: float = 1.0
    # 216# E: Guard 発火カウンタ (累積、再起動時復元)
    _guard_fire_counts: dict[str, int] | None = None

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

    def _warmup_daily_drawdown_from_records(
        self, records: list["FillRecord"],
    ) -> None:
        """203# F: fill records から当日分の PnL を DD guard に投入.

        state file が stale/missing の場合のセーフティネット。
        import_state が skip された後に呼ばれ、DD guard を正確な状態に復元する。
        """
        from datetime import datetime, timezone

        utc_today = datetime.now(timezone.utc).strftime("%Y%m%d")
        daily_pnl_sum = 0.0
        daily_fill_count = 0
        # 209# M1: 1回走査で全指標を計算 (元は2回走査)
        daily_pnl_buy = 0.0
        daily_pnl_sell = 0.0
        for r in records:
            if not r.filled or r.post_fill_30s_pnl is None:
                continue
            # timestamp (epoch) を UTC 日付に変換
            r_date = datetime.fromtimestamp(r.timestamp, tz=timezone.utc).strftime("%Y%m%d")
            if r_date != utc_today:
                continue
            daily_pnl_sum += r.post_fill_30s_pnl
            daily_fill_count += 1
            if r.side == "buy":
                daily_pnl_buy += r.post_fill_30s_pnl
            elif r.side == "sell":
                daily_pnl_sell += r.post_fill_30s_pnl

        if daily_fill_count > 0:
            # 1件ずつ update_pnl を呼ばず、直接 state を注入 (効率的)
            guard = self._daily_drawdown_guard
            guard.state.daily_pnl_bps = daily_pnl_sum
            guard.state.daily_fill_count = daily_fill_count
            guard.state.current_day = utc_today
            # 207# §2: per-side PnL 注入
            guard.state.daily_pnl_bps_buy = daily_pnl_buy
            guard.state.daily_pnl_bps_sell = daily_pnl_sell
            # per-side halt 判定
            if guard._per_side_enabled:
                if daily_pnl_buy <= guard._per_side_hard_limit_bps:
                    guard.state.side_halted_buy = True
                    guard.state.side_halt_remaining_buy = guard._per_side_halt_cycles
                if daily_pnl_sell <= guard._per_side_hard_limit_bps:
                    guard.state.side_halted_sell = True
                    guard.state.side_halt_remaining_sell = guard._per_side_halt_cycles
            # soft limit チェック (hard 超過時も soft は必ず超過)
            if daily_pnl_sum <= guard._soft_limit_bps:
                guard._soft_triggered_today = True
            # hard limit チェック
            if daily_pnl_sum <= guard._hard_limit_bps:
                guard.state.halted = True
                import time as _time
                guard.state.halt_triggered_at = _time.time()
            logger.warning(
                f"[203# F] DD warmup from fill records: {daily_fill_count} fills today, "
                f"daily_pnl={daily_pnl_sum:+.2f}bps, "
                f"buy={daily_pnl_buy:+.2f}bps, sell={daily_pnl_sell:+.2f}bps, "
                f"halted={guard.state.halted}"
            )

    # ------------------------------------------------------------------
    # 209# H4: DynamicKillManager warmup — fill records から rolling PnL 復元
    # ------------------------------------------------------------------
    def _warmup_kill_managers_from_records(
        self, records: list["FillRecord"],
    ) -> None:
        """209# H4: fill records から sell/buy kill manager の PnL 履歴を復元.

        state file が stale/missing の場合のセーフティネット。
        DD warmup と同様、既存 fill records の post_fill_30s_pnl を replay する。
        """
        sell_count = 0
        buy_count = 0
        for r in records:
            if not r.filled or r.post_fill_30s_pnl is None:
                continue
            if r.side == "sell":
                self._sell_kill_mgr.track(r.post_fill_30s_pnl)
                sell_count += 1
            elif r.side == "buy":
                self._buy_kill_mgr.track(r.post_fill_30s_pnl)
                buy_count += 1
        if sell_count > 0 or buy_count > 0:
            logger.info(
                f"[209# H4] Kill manager warmup from fill records: "
                f"sell={sell_count}, buy={buy_count}"
            )

    # ------------------------------------------------------------------
    # 210# DRY: FillTestState 構築の共通化 (3 箇所の重複排除)
    # ------------------------------------------------------------------
    def _build_state_snapshot(
        self,
        *,
        total_count: int,
        filled_count: int,
        cumulative_pnl_jpy: float,
    ) -> object:
        """現在の状態から FillTestState スナップショットを構築.

        saves/halt saves/final save の 3 箇所で同一フィールドを
        構築していたものを一元化し DRY 原則を担保する。
        """
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
            # 210# L-2: one-sided 連続カウンタ永続化
            one_sided_consecutive_count=self._one_sided_consecutive_count,
            # 216# E: Guard 発火カウンタ永続化
            guard_fire_counts=dict(self._guard_fire_counts) if self._guard_fire_counts else None,
            # 209# H4: DynamicKillManager 状態永続化
            sell_kill_state=self._sell_kill_mgr.export_state(),
            buy_kill_state=self._buy_kill_mgr.export_state(),
            **self._get_regime_state_fields(),
        )

    # ------------------------------------------------------------------
    # 216# E: Guard 発火カウンタ — インクリメント・ヘルパー
    # ------------------------------------------------------------------
    def _inc_guard_fire(self, guard_name: str) -> None:
        """累積 guard 発火カウンタをインクリメント."""
        if self._guard_fire_counts is None:
            self._guard_fire_counts = {}
        self._guard_fire_counts[guard_name] = self._guard_fire_counts.get(guard_name, 0) + 1

    # ------------------------------------------------------------------
    # 216# §6 DRY: State 復元共通ヘルパー
    # ------------------------------------------------------------------
    def _restore_common_state(self, saved_state: "FillTestState | None") -> None:
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

    # ------------------------------------------------------------------
    # 179# S1: _effective_sleep — regime 応答サイクル間隔の一元化
    # ------------------------------------------------------------------
    async def _effective_sleep(self, *, multiplier: float = 1.0) -> None:
        """179# CycleStrategy に委譲し、regime 別サイクル間隔で sleep.

        skip/halt/error continue 全パスがこのメソッドを経由する。
        - multiplier=1.0 : 通常スキップ
        - multiplier=5.0 : halt (daily drawdown)
        正常サイクル完了パスは rapid_exit ロジックを含むため直接呼ばない。
        200# P0-2: _soft_drawdown_interval_multiplier を追加乗算。
        """
        regime = self._current_regime_value()
        base = self._cycle_strategy.effective_interval(regime)
        # 200# P0-2: soft drawdown で lot 半減不可 → interval 延長
        soft_dd_mult = getattr(self, "_soft_drawdown_interval_multiplier", 1.0)
        # 217# fix: alert_mode の interval_mult をスキップ/halt パスにも適用
        alert_im = getattr(self, "_alert_interval_mult", 1.0)
        _raw = base * multiplier * soft_dd_mult * alert_im
        # 211#: max_cycle_sleep_sec キャップを _effective_sleep にも適用
        # (209# M4 は通常パスのみ → halt 中 30 分スリープの原因)
        _max = self.config.max_cycle_sleep_sec
        _sleep = min(_raw, _max) if _max > 0 else _raw
        await asyncio.sleep(_sleep)

    def _make_loop_skip_record(
        self,
        *,
        timestamp: float | None = None,
        side: str,
        cancel_reason: str,
        cycle_id: str | None = None,
        order_quantity: float | None = None,
        order_price: float = 0.0,
        spread_at_order: float | None = None,
        spread_offset_ratio: float | None = None,
        balance_forced_switch: bool = False,
        **extra: object,
    ) -> FillRecord:
        """run_continuous 系 skip record の共通 wrapper.

        ループ側の skip は常に現在レジームを記録するため、呼び出し側の重複指定を除く。
        """
        return self._make_skip_record(
            timestamp=timestamp,
            side=side,
            cancel_reason=cancel_reason,
            cycle_id=cycle_id,
            order_quantity=order_quantity,
            order_price=order_price,
            spread_at_order=spread_at_order,
            spread_offset_ratio=spread_offset_ratio,
            regime=self._current_regime_value(),
            balance_forced_switch=balance_forced_switch,
            **extra,
        )

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
        from datetime import datetime, timezone

        from scripts.v460.lib.event_logger import log_event as _log_event
        from ztb.data.trades_health import check_trades_health
        from ztb.metrics.fill_quality import (
            compute_record_pnl_jpy,
            filter_clean_records,
            iter_fill_records_glob,
        )
        from scripts.v460.lib.resilience import FillTestState

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
            pnl_jpy = compute_record_pnl_jpy(r)
            if pnl_jpy is not None:
                cumulative_pnl_jpy += pnl_jpy

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
            self._restore_common_state(saved_state)
        else:
            # regime_detector がない場合でも共通 state は復元
            saved_state = self._state_persistence.load()
            self._restore_common_state(saved_state)

        # 203# F: DD warmup — state 復元失敗時 (stale/missing) は fill records から再計算
        # state file が壊れている or 保存されていない場合のセーフティネット
        # 215# P0-A: needs_warmup_repair() で per-side/soft 整合性も検出
        if (
            self._daily_drawdown_guard.enabled
            and existing_records
            and (
                self._daily_drawdown_guard.state.daily_fill_count == 0
                or self._daily_drawdown_guard.needs_warmup_repair()
            )
        ):
            self._warmup_daily_drawdown_from_records(existing_records)

        # 209# H4: Kill manager warmup — state 復元で PnL 履歴が空なら records から再計算
        if existing_records and len(self._sell_kill_mgr._pnl_history) == 0:
            self._warmup_kill_managers_from_records(existing_records)

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
            # 200# 10-A: 日替わり時に soft_drawdown_interval_multiplier をリセット
            # P0-2 で追加した multiplier が日次境界で reset されないバグの修正
            if self._daily_drawdown_guard.maybe_reset_day():
                _old_mult = getattr(self, "_soft_drawdown_interval_multiplier", 1.0)
                if _old_mult != 1.0:
                    logger.info(
                        f"[daily_drawdown] Day reset → soft_drawdown_interval_multiplier "
                        f"{_old_mult:.1f} → 1.0"
                    )
                    self._soft_drawdown_interval_multiplier = 1.0
                # 207# §4: toxic veto も日替わりでクリア
                if self._toxic_veto:
                    logger.info(
                        f"[daily_drawdown] Day reset → toxic veto cleared: {self._toxic_veto}"
                    )
                    self._toxic_veto = {}
                # 209# M-3: one-sided 連続カウンタも日替わりでリセット
                if self._one_sided_consecutive_count > 0:
                    logger.info(
                        f"[daily_drawdown] Day reset → one_sided_consecutive_count "
                        f"{self._one_sided_consecutive_count} → 0"
                    )
                    self._one_sided_consecutive_count = 0

            # 168# §4.1 #3: 日次ドローダウンガード — halt 中はスキップ
            if self._daily_drawdown_guard.is_halted():
                # 日次 PnL 超過 → UTC 日替わりまでスキップ
                # 200# K: halt record 削減 — 開始/終了 + N回毎のみ記録
                # 203# G: _halt_iter_count で正確にカウント (_cycle_count は halt 中不変)
                _halt_entering = getattr(self, "_halt_start_cycle", None) is None
                if _halt_entering:
                    self._inc_guard_fire("dd_halt")
                    self._halt_start_cycle = self._cycle_count
                    self._halt_iter_count = 0
                else:
                    self._halt_iter_count = getattr(self, "_halt_iter_count", 0) + 1
                # 211# fix: halt 中は progress_log_interval(50) ではなく
                # 専用間隔 _HALT_PERSIST_INTERVAL(10) で state/record を保存。
                # 600s sleep × 50 = 8.3h は長すぎ、再起動時に進捗が大幅に巻き戻る。
                _HALT_PERSIST_INTERVAL = 10
                _should_record_halt = (
                    _halt_entering  # 開始時
                    or self._halt_iter_count % _HALT_PERSIST_INTERVAL == 0
                )
                if _should_record_halt:
                    batch.append(self._make_loop_skip_record(
                        side="none",
                        cancel_reason=CR.DAILY_DRAWDOWN_HALT,
                        order_quantity=0.0,
                    ))
                    total_count += 1
                    batch = self._batch_persistence.maybe_flush(batch, "daily_drawdown_halt")
                self._update_lock_heartbeat()
                # 203# E: HALT 開始時は必ず state 保存 + 以降は N iter 毎
                # 211# fix: halt 専用間隔に統一 (旧: progress_log_interval=50 → 8.3h)
                if _should_record_halt:
                    self._state_persistence.save(self._build_state_snapshot(
                        total_count=total_count,
                        filled_count=filled_count,
                        cumulative_pnl_jpy=cumulative_pnl_jpy,
                    ))
                # 211#: halt サイクル可視化ログ (entering + _HALT_PERSIST_INTERVAL 毎)
                if _should_record_halt:
                    logger.info(
                        f"[daily_drawdown] Halt cycle #{self._halt_iter_count}"
                        f" (next log @+{_HALT_PERSIST_INTERVAL} iters)"
                    )
                await self._effective_sleep(multiplier=5.0)  # 179# S1: halt 中は 5x 間隔
                continue

            # 200# K: halt 終了時の記録 (前サイクルが halt だった場合)
            if getattr(self, "_halt_start_cycle", None) is not None:
                _halt_iters = getattr(self, "_halt_iter_count", 0)
                logger.info(
                    f"[daily_drawdown] Halt ended after {_halt_iters} iterations"
                )
                self._halt_start_cycle = None
                self._halt_iter_count = 0

            # 215# P0-C: alert_mode.json — オペレータ緊急介入チェック
            from scripts.v460.lib.alert_mode import load_alert_mode
            _alert = load_alert_mode(self._results_dir)
            if _alert.halt:
                batch.append(self._make_loop_skip_record(
                    side="none",
                    cancel_reason=CR.OPERATOR_HALT,
                    order_quantity=0.0,
                ))
                total_count += 1
                batch = self._batch_persistence.maybe_flush(batch, "operator_halt")
                self._update_lock_heartbeat()
                await self._effective_sleep(multiplier=5.0)
                continue
            # 215# P0-C: 非 halt オーバーライドをインスタンス変数に保存
            # (fill_cycle_executor から参照)
            self._alert_offset_mult = _alert.offset_mult
            self._alert_lot_mult = _alert.lot_mult
            self._alert_interval_mult = _alert.interval_mult

            # 211# P1-B: Micro Circuit Breaker — 短期価格急変の自動防御
            _mcb_warning = False
            if self._mcb.config.enabled:
                _mcb_mid = getattr(self._maker_price, "last_mid_price", None)
                if _mcb_mid is not None and _mcb_mid > 0:
                    self._mcb.update(_mcb_mid, time.time())
                _mcb_result = self._mcb.check(time.time())
                from scripts.v460.lib.micro_circuit_breaker import MCBLevel
                if _mcb_result.level == MCBLevel.HALT:
                    self._inc_guard_fire("mcb_halt")
                    batch.append(self._make_loop_skip_record(
                        side="none",
                        cancel_reason=CR.MCB_HALT,
                        order_quantity=0.0,
                    ))
                    total_count += 1
                    batch = self._batch_persistence.maybe_flush(batch, "mcb_halt")
                    self._update_lock_heartbeat()
                    await self._effective_sleep(multiplier=5.0)
                    continue
                if _mcb_result.level == MCBLevel.WARNING:
                    _mcb_warning = True
                    self._inc_guard_fire("mcb_warning")
                    # WARNING: offset/interval を拡大 (alert_mode との積算)
                    self._alert_offset_mult *= _mcb_result.offset_mult
                    self._alert_interval_mult *= _mcb_result.interval_mult

            # 211# P1-C: Spread Anomaly Detector — 流動性枯渇検知
            _sad_warning = False
            if self._sad.config.enabled:
                # 217# fix: last_spread は 60s staleness guard 付き (210# M5)。
                # cycle 間隔 120s では常に stale → None になるため、
                # staleness guard なしの last_spread_raw を使用する。
                _sad_spread = getattr(self._maker_price, "last_spread_raw", None)
                if _sad_spread is not None and _sad_spread > 0:
                    self._sad.update(_sad_spread, time.time())
                _sad_result = self._sad.check(time.time())
                from scripts.v460.lib.spread_anomaly_detector import SADLevel
                if _sad_result.level == SADLevel.FROZEN:
                    self._inc_guard_fire("sad_frozen")
                    batch.append(self._make_loop_skip_record(
                        side="none",
                        cancel_reason=CR.SAD_FROZEN,
                        order_quantity=0.0,
                    ))
                    total_count += 1
                    batch = self._batch_persistence.maybe_flush(batch, "sad_frozen")
                    self._update_lock_heartbeat()
                    await self._effective_sleep(multiplier=5.0)
                    continue
                if _sad_result.level == SADLevel.DRY:
                    _sad_warning = True
                    self._inc_guard_fire("sad_dry")
                    self._alert_offset_mult *= _sad_result.offset_mult
                    self._alert_interval_mult *= _sad_result.interval_mult
                    self._alert_lot_mult *= _sad_result.lot_mult
                elif _sad_result.level == SADLevel.WIDE:
                    _sad_warning = True
                    self._inc_guard_fire("sad_wide")
                    self._alert_offset_mult *= _sad_result.offset_mult

            # 211# P1-D: MCB×SAD AND Escalation
            # 両方が同時に WARNING 以上 → 即 HALT (false positive 抑制)
            if _mcb_warning and _sad_warning:
                self._inc_guard_fire("mcb_sad_escalation")
                batch.append(self._make_loop_skip_record(
                    side="none",
                    cancel_reason=CR.MCB_SAD_ESCALATION,
                    order_quantity=0.0,
                ))
                total_count += 1
                batch = self._batch_persistence.maybe_flush(
                    batch, "mcb_sad_escalation"
                )
                self._update_lock_heartbeat()
                await self._effective_sleep(multiplier=5.0)
                continue

            # 205# §9.4: 時間帯 Hard Skip (Kyle proxy)
            # soft offset (158# P1-6) では抑制不十分な最悪時間帯はサイクル全停止
            if self.config.hard_skip_utc_hours:
                from datetime import datetime, timezone
                _utc_h = datetime.now(timezone.utc).hour
                if _utc_h in self.config.hard_skip_utc_hours:
                    # 初回のみ skip record を記録
                    _hard_skip_entering = not getattr(self, "_in_hard_skip_hour", False)
                    self._in_hard_skip_hour = True
                    if _hard_skip_entering:
                        self._inc_guard_fire("hard_skip_utc")
                        batch.append(self._make_loop_skip_record(
                            side="none",
                            cancel_reason=CR.HARD_SKIP_UTC_HOUR,
                            order_quantity=0.0,
                        ))
                        total_count += 1
                        batch = self._batch_persistence.maybe_flush(batch, "hard_skip_utc_hour")
                        logger.info(
                            f"[205# §9.4] Hard skip: UTC {_utc_h}h is in "
                            f"hard_skip_utc_hours={self.config.hard_skip_utc_hours}"
                        )
                    self._update_lock_heartbeat()
                    await self._effective_sleep()
                    continue
                else:
                    if getattr(self, "_in_hard_skip_hour", False):
                        logger.info(f"[205# §9.4] Hard skip ended (UTC {_utc_h}h)")
                        self._in_hard_skip_hour = False

            # 205# §9.5: 片側 DD Halt のサイクルカウンタ更新
            self._daily_drawdown_guard.tick_side_halt()

            # 205# §9.2: Toxic Fill 同一サイド拒否 — 初期化のみ (デクリメントはサイクル末尾)
            if self._toxic_veto is None:
                self._toxic_veto = {}

            # 129# D.2: 残高制約による side 強制切替追跡
            _balance_forced = False
            _is_rescue = False  # 158# P1-1: balance_forced rescue フラグ
            _one_sided_balance = False  # 190# B: 片側 balance フラグ (ev_weighted threshold 緩和用)
            # 073# side 別時間帯フィルター: side 決定後にフィルタリング
            # side 別リスト未設定時はグローバルリスト (041# 互換)
            next_side = self._next_side()

            # 205# §9.5: 片側 DD Halt チェック — 封鎖されたサイドは回避
            if self._daily_drawdown_guard.is_side_halted(next_side):
                _alt = "sell" if next_side == "buy" else "buy"
                if self._daily_drawdown_guard.is_side_halted(_alt):
                    # 両サイド封鎖 → 集約 halt と同等扱い
                    batch.append(self._make_loop_skip_record(
                        side="none",
                        cancel_reason=CR.PER_SIDE_DD_HALT,
                        order_quantity=0.0,
                    ))
                    total_count += 1
                    batch = self._batch_persistence.maybe_flush(batch, "per_side_dd_both_halt")
                    self._update_lock_heartbeat()
                    await self._effective_sleep(multiplier=5.0)
                    continue
                else:
                    logger.info(
                        f"[205# §9.5] Per-side DD halt: {next_side} blocked, "
                        f"switching to {_alt}"
                    )
                    next_side = _alt

            # 205# §9.2: Toxic Fill 同一サイド拒否 — 封鎖されたサイドは反対に切替
            # 207# §5b: alt_side が per_side_dd で封鎖されている場合も考慮
            if self._toxic_veto and next_side in self._toxic_veto:
                _alt = "sell" if next_side == "buy" else "buy"
                _alt_blocked = (
                    _alt in self._toxic_veto
                    or self._daily_drawdown_guard.is_side_halted(_alt)
                )
                if _alt_blocked:
                    # 両サイド封鎖 (veto + per_side_dd 含む) → スキップ
                    # 209# H-1: デッドロック防止 — skip 時も veto カウンタを減算
                    self._inc_guard_fire("toxic_veto_block")
                    for _vs in list(self._toxic_veto.keys()):
                        self._toxic_veto[_vs] -= 1
                        if self._toxic_veto[_vs] <= 0:
                            del self._toxic_veto[_vs]
                            logger.info(f"[209# H-1] Toxic veto expired (both-blocked): {_vs}")
                    batch.append(self._make_loop_skip_record(
                        side="none",
                        cancel_reason=CR.TOXIC_FILL_SIDE_VETO,
                        order_quantity=0.0,
                    ))
                    total_count += 1
                    batch = self._batch_persistence.maybe_flush(batch, "toxic_veto_both")
                    await self._effective_sleep()
                    continue
                else:
                    logger.info(
                        f"[205# §9.2] Toxic veto: {next_side} blocked "
                        f"(remaining={self._toxic_veto[next_side]}), "
                        f"switching to {_alt}"
                    )
                    next_side = _alt

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
                        batch.append(self._make_loop_skip_record(
                            side=next_side,
                            cancel_reason=CR.TIME_FILTER_BOTH_SIDES,
                            order_quantity=0.0,
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
                                batch.append(self._make_loop_skip_record(
                                    side=next_side,
                                    cancel_reason=CR.TIME_FILTER_086_DEADLOCK,
                                    order_quantity=0.0,
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

            # 158# §20-A: skip パスでも regime 遷移保証 (fallback price 投入)
            if self._regime_detector is not None:
                _fb_price, _fb_time = self._maker_price.get_fallback_price()
                if _fb_price is not None:
                    _pre_regime = self._regime_detector.current_regime
                    _regime_result = self._regime_detector.update(
                        time.time(), _fb_price
                    )
                    # 182# confidence キャッシュ (Trend Mode 厳格化)
                    if hasattr(self, "_cycle_strategy"):
                        self._cycle_strategy.update_confidence(_regime_result.confidence)
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
                    # 200# E: 時間ベース頻度検出 — 短時間で連続 balance_forced が発生 → 警告
                    _now = time.time()
                    _last_bf_time = getattr(self, "_last_balance_forced_time", 0.0)
                    _bf_cooldown = self.config.balance_forced_cooldown_sec
                    if _bf_cooldown > 0 and (_now - _last_bf_time) < _bf_cooldown:
                        _bf_freq_count = getattr(self, "_balance_forced_freq_count", 0) + 1
                        self._balance_forced_freq_count = _bf_freq_count
                        logger.warning(
                            f"[200# E] balance_forced high frequency: "
                            f"{_bf_freq_count} events within {_bf_cooldown:.0f}s "
                            f"(interval={_now - _last_bf_time:.1f}s)"
                        )
                    else:
                        self._balance_forced_freq_count = 0
                    self._last_balance_forced_time = _now

                if not tried_opposite:
                    # 両 side とも残高不足 → 従来通りの処理
                    self._last_side = next_side  # → 次の _next_side() が反対を返す
                    self._preflight_skip_count += 1

                    # 140# §8.1-#2: preflight skip record 生成 (132# F4)
                    # 145# §9-#5: _make_skip_record DRY 化
                    batch.append(self._make_loop_skip_record(
                        side=next_side,
                        cancel_reason=CR.PREFLIGHT_INSUFFICIENT,
                        order_quantity=self._current_lot,
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
                        _pause_record_ts = time.time()
                        batch.append(self._make_loop_skip_record(
                            timestamp=_pause_record_ts,
                            side="none",
                            cancel_reason=CR.PREFLIGHT_PAUSE,
                            cycle_id=(
                                f"preflight_pause_{self._preflight_pause_count}_"
                                f"{int(_pause_record_ts)}"
                            ),
                            order_quantity=0.0,
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
            # 133# P0-08 / 154# C-1/C-2: balance_forced スキップ + deadlock 防止
            if _balance_forced and self.config.skip_balance_forced:
                # 154# C-1: 両側残高判定
                original_side = "buy" if next_side == "sell" else "sell"
                original_also_insufficient = await self._check_balance_for_side(
                    original_side, regime_mult=_regime_mult
                )
                # 154# C-2 + 182# regime 別緩和: trending 時は deadlock_limit 引き上げ
                _r = self._current_regime_value()
                _deadlock_limit = (
                    self._cycle_strategy.policy.deadlock_limit_trending
                    if _r and _r.startswith("trending") and hasattr(self, "_cycle_strategy")
                    else self.config.balance_forced_deadlock_limit
                )
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
                    _one_sided_balance = original_also_insufficient  # 190# B
                    # 202# B: 片側残高枯渇時は rescue offset で保護
                    if (
                        original_also_insufficient
                        and self.config.one_sided_balance_rescue_offset
                    ):
                        _is_rescue = True
                        logger.info(
                            f"[202# B] one_sided_balance rescue: offset ×"
                            f"{self.config.balance_forced_rescue_offset_mult:.1f}"
                        )
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
                        f"[133# P0-08] Skipping cycle — balance_forced_switch=True. "
                        f"side={next_side}, "
                        f"consecutive={self._balance_forced_skip_count}"
                    )
                    # 145# §9-#5: _make_skip_record DRY 化
                    _skip_record = self._make_loop_skip_record(
                        side=next_side,
                        cancel_reason=CR.BALANCE_FORCED_SKIP,
                        order_quantity=self._current_lot,
                        balance_forced_switch=True,
                        balance_forced_consecutive=self._balance_forced_skip_count,
                    )
                    batch.append(_skip_record)
                    total_count += 1
                    batch = self._batch_persistence.maybe_flush(batch, "balance_forced_skip")
                    # 167# DL-5: _last_side を更新 (rescue=true 時は到達しないが防御的に)
                    self._last_side = next_side
                    await self._effective_sleep()  # 179# S1
                    continue

            # ════════════════════════════════════════════════════════════
            # 194# CycleGateAggregator: per-cycle skip 判定の一元化
            # 旧: A10-A14 の散在 if/continue (220行) → 統合ゲート評価
            # ════════════════════════════════════════════════════════════

            # HF4 安全弁: trending_sell のための buy 残高チェック (async)
            _buy_side_insufficient = False
            if (
                self.config.skip_sell_trending
                and next_side == "sell"
                and not _balance_forced
                and self._regime_detector is not None
                and self._regime_detector.current_regime.is_trending
            ):
                _buy_side_insufficient = await self._check_balance_for_side(
                    "buy", regime_mult=_regime_mult,
                )

            _gate_result = self._cycle_gate.evaluate(
                side=next_side,
                regime=(
                    self._regime_detector.current_regime.value
                    if self._regime_detector is not None else None
                ),
                vol_ratio=(
                    self._regime_detector.last_volatility_ratio
                    if self._regime_detector is not None else None
                ),
                balance_forced=_balance_forced,
                inv_net_imbalance=self._maker_price.inv_net_imbalance,
                is_buy_killed=self._is_buy_killed(),
                is_sell_killed=self._is_sell_killed(),
                # 197# Gate 8-9: cached spread/mid for pre-check
                spread_jpy=self._maker_price.last_spread,
                mid_price=self._maker_price.last_mid_price,
                # 210# H3: velocity を gate に渡す (dead code 解消)
                # NOTE: last_mid_trend_bps は OB mid 差分ベースの instant velocity であり
                # 元々想定されていた trade_vel_60s とはデータソースが異なるが、
                # 符号規約 (正=上昇) と閾値比較のセマンティクスは同一。
                price_velocity_bps=self._maker_price.last_mid_trend_bps,
                trending_sell_skip_count=self._trending_sell_skip_count,
                buy_side_insufficient=_buy_side_insufficient,
            )

            if _gate_result.blocked:
                # カウンタ管理
                if _gate_result.blocking_reason == "trending_sell_skip":
                    self._trending_sell_skip_count += 1
                    _max_c = self.config.max_consecutive_trending_sell_skip
                    logger.info(
                        f"[194#] {_gate_result.blocking_reason} "
                        f"[consecutive={self._trending_sell_skip_count}"
                        f"/{_max_c if _max_c > 0 else '∞'}] "
                        f"[{_gate_result.audit_summary}]"
                    )
                else:
                    logger.info(
                        f"[194#] Cycle gate blocked: {_gate_result.blocking_reason} "
                        f"[{_gate_result.audit_summary}]"
                    )

                _skip_record = self._make_loop_skip_record(
                    side=next_side,
                    cancel_reason=_gate_result.cancel_reason,
                    order_quantity=self._current_lot,
                )
                batch.append(_skip_record)
                total_count += 1
                batch = self._batch_persistence.maybe_flush(
                    batch, _gate_result.cancel_reason,
                )
                self._last_side = next_side
                # 197# narrow_spread_pause: Gate 8 ブロック時は pause_sec 分待機
                if _gate_result.blocking_reason == "narrow_spread_pause":
                    await asyncio.sleep(self.config.narrow_spread_pause_sec)
                else:
                    await self._effective_sleep()
                continue
            else:
                # ゲート通過 → trending sell カウンタリセット
                if (
                    self.config.skip_sell_trending
                    and next_side == "sell"
                    and self._regime_detector is not None
                    and self._regime_detector.current_regime.is_trending
                ):
                    self._trending_sell_skip_count = 0

            try:
                record = await self.run_single_cycle(
                    side_override=next_side,
                    balance_forced_switch=_balance_forced,
                    balance_forced_rescue=_is_rescue,
                    one_sided_balance=_one_sided_balance,
                    trending_offset_mult=_gate_result.trending_offset_mult,
                )
                # 154# C-2: 実サイクル実行 → forced skip カウンタリセット
                self._balance_forced_skip_count = 0
                # 158# §20-B: 実サイクル実行 → trending sell skip カウンタリセット
                self._trending_sell_skip_count = 0
                # 207# §4: one-sided 連続実行追跡 (205# §4.2 Codex)
                if _one_sided_balance:
                    self._one_sided_consecutive_count += 1
                    _os_limit = self.config.one_sided_consecutive_limit
                    if _os_limit > 0 and self._one_sided_consecutive_count >= _os_limit:
                        logger.warning(
                            f"[207# §4] One-sided consecutive limit reached: "
                            f"{self._one_sided_consecutive_count}/{_os_limit} — "
                            f"interval ×{self.config.one_sided_consecutive_interval_mult:.1f}"
                        )
                else:
                    if self._one_sided_consecutive_count > 0:
                        logger.info(
                            f"[207# §4] One-sided streak ended: "
                            f"{self._one_sided_consecutive_count} consecutive → reset"
                        )
                    self._one_sided_consecutive_count = 0
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
                # 202# A: 単一サイクル大損失クールダウン — 連鎖損失防止
                if (
                    record.post_fill_30s_pnl is not None
                    and record.post_fill_30s_pnl <= self.config.loss_cooldown_threshold_bps
                ):
                    self._loss_cooldown_mult = self.config.loss_cooldown_interval_mult
                    # 211# 204# I: 大損後に次回 offset も拡大
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
                # 205# §9.2: Toxic Fill 同一サイド拒否 — 大損時に同一サイドを封鎖
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
                # 033# F4: 累積 PnL インクリメンタル追跡
                pnl_jpy = compute_record_pnl_jpy(record)
                if pnl_jpy is not None:
                    cumulative_pnl_jpy += pnl_jpy
                # 168# §4.1 #3: 日次ドローダウンガード PnL 更新
                if record.post_fill_30s_pnl is not None:
                    dd_result = self._daily_drawdown_guard.update_pnl(
                        record.post_fill_30s_pnl,
                        side=next_side,  # 205# §9.5: 片側 DD 追跡
                    )
                    if dd_result.get("soft_triggered"):
                        old_lot = self._current_lot
                        new_lot = self._current_lot / 2
                        if new_lot >= self.config.order_quantity:
                            # 200# P0-2: lot 半減可能
                            self._current_lot = new_lot
                            self._balance_checker.pre_shrink_lot = self._current_lot
                            logger.warning(
                                f"[daily_drawdown] soft lot reduction: "
                                f"{old_lot:.4f} → {self._current_lot:.4f} BTC"
                            )
                        else:
                            # 200# P0-2: 最小ロット到達 → interval 延長で exposure 削減
                            self._soft_drawdown_interval_multiplier = self.config.soft_drawdown_interval_multiplier
                            logger.warning(
                                f"[daily_drawdown] min lot reached ({old_lot:.4f} BTC), "
                                f"applying 3x interval multiplier instead of lot reduction"
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
                self._state_persistence.save(self._build_state_snapshot(
                    total_count=total_count,
                    filled_count=filled_count,
                    cumulative_pnl_jpy=cumulative_pnl_jpy,
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
                # 200# P0-2: soft drawdown interval 延長
                soft_dd_mult = getattr(self, "_soft_drawdown_interval_multiplier", 1.0)
                # 202# A: 単一サイクル大損失クールダウン (1回適用で自動リセット)
                _loss_cd = self._loss_cooldown_mult
                self._loss_cooldown_mult = 1.0  # 次サイクルではリセット

                # 207# §3: Toxic veto カウンタ減算 (サイクル末尾で実行 — off-by-one 防止)
                if self._toxic_veto:
                    for _veto_side in list(self._toxic_veto.keys()):
                        self._toxic_veto[_veto_side] -= 1
                        if self._toxic_veto[_veto_side] <= 0:
                            del self._toxic_veto[_veto_side]
                            logger.info(f"[205# §9.2] Toxic veto expired: {_veto_side}")

                # 207# §4: one-sided 連続実行制限到達時の interval 延長
                _os_limit = self.config.one_sided_consecutive_limit
                _os_mult = 1.0
                if (
                    _os_limit > 0
                    and self._one_sided_consecutive_count >= _os_limit
                ):
                    _os_mult = self.config.one_sided_consecutive_interval_mult

                # 209# M4: sleep 上限キャップ — 乗数積み重ねによる長時間無応答を防止
                # 215# P0-C: alert_mode interval_mult を追加
                _alert_im = self._alert_interval_mult
                _raw_sleep = interval * soft_dd_mult * _loss_cd * _os_mult * _alert_im
                _max_sleep = self.config.max_cycle_sleep_sec
                _clamped = min(_raw_sleep, _max_sleep) if _max_sleep > 0 else _raw_sleep
                await asyncio.sleep(_clamped)

        # 残りバッチを保存
        if batch:
            if not self._batch_persistence.try_save_batch(batch):
                # 最終手段: 緊急ダンプ
                self._batch_persistence.emergency_dump(batch, "final")

        # 113# resilience: 最終状態保存
        self._state_persistence.save(self._build_state_snapshot(
            total_count=total_count,
            filled_count=filled_count,
            cumulative_pnl_jpy=cumulative_pnl_jpy,
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
        return list(iter_fill_records_glob(str(self._results_dir)))

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
