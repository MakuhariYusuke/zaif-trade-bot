"""330# orchestrator_pre_cycle — run_continuous ガードフェーズ Mixin.

328# God Object Split Phase 3:
run_continuous (~1,281 行) のうち ~540 行の pre-cycle ガードロジックを
OrchestratorPreCycleMixin に分離。残る run_continuous は制御フロー
(~740 行) のみを保持し、将来の Phase 4 でさらに分割可能。

依存関係:
  FillLoopOrchestratorMixin
    ← OrchestratorGuardsMixin      (_inc_guard_fire, _tick_toxic_veto, _feed_mcb_sad, ...)
    ← OrchestratorLifecycleMixin   (_build_state_snapshot, _maybe_skip_state_save, ...)
    ← OrchestratorPostCycleMixin
    ← OrchestratorPreCycleMixin    (本ファイル — 新規)

CycleContext:
  run_continuous の 1 イテレーション内で可変なサイクルローカル状態を構造化。
  RunSessionState (ループ間共有) とは異なり、各イテレーション冒頭で再生成。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.hour_rules import current_utc_hour
from scripts.v460.lib.micro_circuit_breaker import MCBLevel
from scripts.v460.lib.spread_anomaly_detector import SADLevel

if TYPE_CHECKING:
    from scripts.v460.lib.fill_loop_orchestrator import RunSessionState

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 330# CycleContext: 1 イテレーション内のサイクルローカル状態
# ------------------------------------------------------------------
@dataclass
class CycleContext:
    """run_continuous 1 イテレーション内の可変状態.

    各ガードフェーズが参照・更新するサイクルローカル変数を構造化し、
    extract method 間での受渡しを型安全に行う。

    RunSessionState (ループ間共有) とは異なり、CycleContext は
    while ループの 1 回分のみ有効。毎イテレーション冒頭で再生成する。

    332# Phase 4: 抽出メソッド間の受渡しに使用される。
    348# balance_forced 概念を全廃。
    522# balance_switch / recovery_skew / inventory_escape を完全撤廃。
    残高不足時はスキップし、次サイクルで自然に反対 side が選択される。
    """

    next_side: str = ""
    #: 420# P1: SideSelector が最初に返した side (balance/veto 切替前)
    requested_side: str = ""
    #: 420# P1: side 切替理由 (522# 撤廃: balance_switch/recovery_skew は不使用)
    resolved_side_reason: str | None = None
    #: DEAD CODE (596#): 522# で trigger 消失、常時 False
    #: 片方のみ残高あり → degraded one-sided 実行
    one_sided_balance: bool = False
    #: レジームベースの lot 倍率 (preflight-lot alignment 用)
    regime_mult: float = 1.0


class OrchestratorPreCycleMixin:
    """330# run_continuous の pre-cycle ガードロジック Mixin.

    FillLoopOrchestratorMixin が継承して使用する。
    run_continuous の while ループ冒頭 ~540 行を 7 メソッドに分離。

    各メソッドは bool を返し、True = 呼び出し元で continue (サイクルスキップ)。
    CycleContext は side 解決以降のフェーズで共有する可変状態。
    """

    # ------------------------------------------------------------------
    # Phase 1: 日替わりリセット
    # ------------------------------------------------------------------
    def _process_daily_reset(
        self,
        st: "RunSessionState | None" = None,
    ) -> None:
        """200# 10-A: 日替わりリセット処理.

        330# extract from run_continuous L355-L405.
        UTC 日替わりで以下をリセット:
        - soft_drawdown_interval_multiplier
        - dd_soft_lot_scale (buy/sell)
        - toxic_veto
        - one_sided_consecutive_count
        - dynamic kill (cross-day 矛盾検出)
        - 499# cumulative_pnl_jpy / soft_loss_cap_triggered (当日スコープ)
        """
        if not self._daily_drawdown_guard.maybe_reset_day():
            return

        # 499# fix: cumulative_pnl_jpy は当日スコープ → 日替わりでゼロリセット
        if st is not None:
            _old_pnl = st.cumulative_pnl_jpy
            st.cumulative_pnl_jpy = 0.0
            logger.info(
                f"[499# daily_reset] cumulative_pnl_jpy {_old_pnl:.1f} → 0.0"
            )
        if self._soft_loss_cap_triggered:
            self._soft_loss_cap_triggered = False
            logger.info("[499# daily_reset] soft_loss_cap_triggered → False")

        _old_mult = self._soft_drawdown_interval_multiplier
        if _old_mult != 1.0:
            logger.info(
                f"[daily_drawdown] Day reset → soft_drawdown_interval_multiplier "
                f"{_old_mult:.1f} → 1.0"
            )
            self._soft_drawdown_interval_multiplier = 1.0

        # 303# B: side-aware soft lot scale も日替わりリセット
        if self._dd_soft_lot_scale_buy < 1.0 or self._dd_soft_lot_scale_sell < 1.0:
            logger.info(
                f"[daily_drawdown] Day reset → dd_soft_lot_scale "
                f"buy={self._dd_soft_lot_scale_buy:.2f} → 1.0, "
                f"sell={self._dd_soft_lot_scale_sell:.2f} → 1.0"
            )
            self._dd_soft_lot_scale_buy = 1.0
            self._dd_soft_lot_scale_sell = 1.0

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

        # 648# preflight_pause_count 日替わりリセット:
        # 長時間稼働で回復後も pause 枠を再利用可能にする
        if self._preflight_pause_count > 0:
            logger.info(
                f"[daily_drawdown] Day reset → preflight_pause_count "
                f"{self._preflight_pause_count} → 0"
            )
            self._preflight_pause_count = 0

        # 224# B2: 日替わりリセット × dynamic kill 矛盾検出
        # maybe_reset_day() は per-side halt/PnL を全クリアするが、
        # DynamicKillManager の rolling window は cross-day で残存。
        # kill がアクティブなまま halt が解除されると矛盾が生じるため警告。
        for _km_label, _km in [
            ("sell", self._sell_kill_mgr),
            ("buy", self._buy_kill_mgr),
        ]:
            _k_active, _k_mean, _k_count = _km.is_kill_active()
            if _k_active:
                logger.warning(
                    f"[224# B2] Day reset but {_km_label}_dynamic_kill still active: "
                    f"rolling_mean={_k_mean}, "
                    f"rolling_count={_k_count} — "
                    f"resetting kill state for clean day start"
                )
                _km.reset()
                self._inc_guard_fire("day_reset_kill_conflict")

        # 475# メモリリーク防止: 日替わり時に GC 強制実行 + メモリ使用量ロギング
        import gc
        collected = gc.collect()
        import psutil  # type: ignore[import-untyped]
        proc = psutil.Process()
        mem_mb = proc.memory_info().rss / (1024 * 1024)
        logger.info(
            f"[475# daily GC] collected {collected} objects, "
            f"RSS={mem_mb:.1f}MB"
        )

    # ------------------------------------------------------------------
    # Phase 2: DD halt チェック
    # ------------------------------------------------------------------
    async def _handle_dd_halt(self, st: RunSessionState) -> bool:
        """168# §4.1: 日次ドローダウンガード halt 処理.

        330# extract from run_continuous L407-L460.
        halt 中はスキップ record → MCB/SAD feed → sleep → True (continue)。
        halt 終了時はカウンタクリーンアップ → False (通常フロー)。
        """
        _halt_mult = self.config.halt_sleep_multiplier

        if self._daily_drawdown_guard.is_halted():
            # 日次 PnL 超過 → UTC 日替わりまでスキップ
            # 200# K: halt record 削減 — 開始/終了 + N回毎のみ記録
            # 203# G: _halt_iter_count で正確にカウント
            _halt_entering = self._halt_start_cycle is None
            if _halt_entering:
                self._inc_guard_fire("dd_halt")
                self._halt_start_cycle = self._cycle_count
                self._halt_iter_count = 0
            else:
                self._halt_iter_count = self._halt_iter_count + 1

            # 211# fix: halt 中は progress_log_interval ではなく
            # 専用間隔 halt_persist_interval で state/record を保存。
            _should_record_halt = (
                _halt_entering
                or self._halt_iter_count % self.config.halt_persist_interval == 0
            )
            if _should_record_halt:
                st.batch.append(self._make_loop_skip_record(
                    side="none",
                    cancel_reason=CR.DAILY_DRAWDOWN_HALT,
                    order_quantity=0.0,
                ))
                st.total_count += 1
                st.batch = self._batch_persistence.maybe_flush(
                    st.batch, "daily_drawdown_halt",
                )
            self._update_lock_heartbeat()

            # 203# E: HALT 開始時は必ず state 保存 + 以降は N iter 毎
            if _should_record_halt:
                self._state_persistence.save(self._build_state_snapshot(
                    total_count=st.total_count,
                    filled_count=st.filled_count,
                    cumulative_pnl_jpy=st.cumulative_pnl_jpy,
                ))
                self._last_state_save_time = time.monotonic()

            # 211#: halt サイクル可視化ログ
            if _should_record_halt:
                logger.info(
                    f"[daily_drawdown] Halt cycle #{self._halt_iter_count}"
                    f" (next log @+{self.config.halt_persist_interval} iters)"
                )

            # 226# S5: halt 中も MCB/SAD に price/spread をフィードし続ける。
            self._feed_mcb_sad()
            await self._effective_sleep(multiplier=_halt_mult)
            return True  # caller: continue

        # 200# K: halt 終了時の記録 (前サイクルが halt だった場合)
        if self._halt_start_cycle is not None:
            _halt_iters = self._halt_iter_count
            logger.info(
                f"[daily_drawdown] Halt ended after {_halt_iters} iterations"
            )
            self._halt_start_cycle = None
            self._halt_iter_count = 0

        return False

    # ------------------------------------------------------------------
    # Phase 3: Circuit Breakers (MCB + SAD + Escalation)
    # ------------------------------------------------------------------
    async def _check_circuit_breakers(self, st: RunSessionState) -> bool:
        """211# P1-B/C/D: MCB + SAD + MCB×SAD Escalation.

        330# extract from run_continuous L479-L538.
        MCB HALT / SAD FROZEN / MCB×SAD ESCALATION → True (continue)。
        MCB WARNING / SAD DRY/WIDE → alert_* を乗算して False。

        331# review BUG-1: MCB HALT 時も SAD baseline を維持するため
        先に _feed_mcb_sad() で両方の update を実行してから check する。
        """
        # 331# BUG-1 fix: update を先行一括実行 (MCB HALT early return で
        # SAD.update が skipped になるのを防止)
        self._feed_mcb_sad()

        _halt_mult = self.config.halt_sleep_multiplier
        _mcb_warning = False
        _sad_warning = False

        # 211# P1-B: Micro Circuit Breaker — 短期価格急変の自動防御
        if self._mcb is not None and self._mcb.config.enabled:
            _mcb_mid = self._maker_price.last_mid_price
            if _mcb_mid is not None and _mcb_mid > 0:
                self._mcb.update(_mcb_mid, time.time())
            _mcb_result = self._mcb.check(time.time())
            if _mcb_result.level == MCBLevel.HALT:
                self._inc_guard_fire("mcb_halt")
                # 659# T1-1: HALT 中に BTC ポジションがあれば警告
                _btc = getattr(self._balance_checker, "last_btc_free", None)
                if _btc is not None and _btc > 0:
                    logger.warning(
                        "[659# MCB] HALT with open BTC position: "
                        "%.8f BTC exposed during cooldown %.0fs",
                        _btc, self._mcb.config.halt_cooldown_sec,
                    )
                await self._execute_skip(
                    st, side="none", cancel_reason=CR.MCB_HALT,
                    heartbeat=True, multiplier=_halt_mult,
                )
                return True
            if _mcb_result.level == MCBLevel.WARNING:
                _mcb_warning = True
                self._inc_guard_fire("mcb_warning")
                self._alert_offset_mult *= _mcb_result.offset_mult
                self._alert_interval_mult *= _mcb_result.interval_mult

        # 211# P1-C: Spread Anomaly Detector — 流動性枯渇検知
        if self._sad is not None and self._sad.config.enabled:
            _sad_spread = self._maker_price.last_spread_raw
            if _sad_spread is not None and _sad_spread > 0:
                self._sad.update(_sad_spread, time.time())
            _sad_result = self._sad.check(time.time())
            if _sad_result.level == SADLevel.FROZEN:
                self._inc_guard_fire("sad_frozen")
                await self._execute_skip(
                    st, side="none", cancel_reason=CR.SAD_FROZEN,
                    heartbeat=True, multiplier=_halt_mult,
                )
                return True
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
        if _mcb_warning and _sad_warning:
            self._inc_guard_fire("mcb_sad_escalation")
            await self._execute_skip(
                st, side="none", cancel_reason=CR.MCB_SAD_ESCALATION,
                heartbeat=True, multiplier=_halt_mult,
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Phase 4: Hard Skip UTC
    # ------------------------------------------------------------------
    async def _check_hard_skip_utc(self, st: RunSessionState) -> bool:
        """205# §9.4: 時間帯 Hard Skip (Kyle proxy).

        330# extract from run_continuous L541-L573.
        最悪時間帯はサイクル全停止。初回のみ record 記録。
        """
        if not self.config.hard_skip_utc_hours:
            return False

        _utc_h = current_utc_hour()
        if _utc_h in self.config.hard_skip_utc_hours:
            _hard_skip_entering = not self._in_hard_skip_hour
            self._in_hard_skip_hour = True
            if _hard_skip_entering:
                self._inc_guard_fire("hard_skip_utc")
                st.batch.append(self._make_loop_skip_record(
                    side="none",
                    cancel_reason=CR.HARD_SKIP_UTC_HOUR,
                    order_quantity=0.0,
                ))
                st.total_count += 1
                st.batch = self._batch_persistence.maybe_flush(
                    st.batch, "hard_skip_utc_hour",
                )
                logger.info(
                    f"[205# §9.4] Hard skip: UTC {_utc_h}h is in "
                    f"hard_skip_utc_hours={self.config.hard_skip_utc_hours}"
                )
            self._update_lock_heartbeat()
            await self._effective_sleep()
            return True

        if self._in_hard_skip_hour:
            logger.info(f"[205# §9.4] Hard skip ended (UTC {_utc_h}h)")
            self._in_hard_skip_hour = False

        return False

    # ------------------------------------------------------------------
    # Phase 5: Phantom Position Guard
    # ------------------------------------------------------------------
    async def _process_phantom_guard(self) -> None:
        """237# Phantom Position Guard: status_unknown の遅延再照合.

        330# extract from run_continuous L578-L610.
        tick_veto + reconcile を実行。ファントム検出時は sleep 延長。
        """
        if self._phantom_guard is None:
            return

        self._phantom_guard.tick_veto()

        if not self._phantom_guard.has_pending:
            return

        try:
            _phantom_detections = await self._phantom_guard.reconcile(self.adapter)
            if _phantom_detections:
                self._inc_guard_fire("phantom_position_detected")
                for _pd in _phantom_detections:
                    logger.critical(
                        f"[237# PHANTOM] Inventory mismatch: "
                        f"{_pd.side} {_pd.quantity:.6f} BTC @ {_pd.price:.0f} "
                        f"(method={_pd.detection_method}) — "
                        f"cautious mode activated, side veto="
                        f"{self._phantom_guard._PHANTOM_VETO_CYCLES} cycles"
                    )
                await self._effective_sleep(
                    multiplier=self.config.phantom_detection_sleep_multiplier,
                )
        except Exception as _pg_err:
            logger.warning(f"[237# phantom_guard] Reconcile error: {_pg_err}")

    # ------------------------------------------------------------------
    # Phase 6: Side Veto 解決 (per-side DD + toxic + phantom)
    # ------------------------------------------------------------------
    async def _resolve_side_vetos(
        self, st: RunSessionState, ctx: CycleContext,
    ) -> bool:
        """205# §9.5 / §9.2 / 238#: Side veto 解決.

        330# extract from run_continuous L619-L695.
        per-side DD halt, toxic veto, phantom veto の順に評価。
        封鎖されたサイドを回避 → 反対 side に切替、
        両サイド封鎖 → True (continue)。
        """
        _halt_mult = self.config.halt_sleep_multiplier
        next_side = ctx.next_side

        # ── 205# §9.5: 片側 DD Halt チェック ──
        if self._daily_drawdown_guard.is_side_halted(next_side):
            _alt = self._opposite_side(next_side)
            if self._daily_drawdown_guard.is_side_halted(_alt):
                # 両サイド封鎖 → 集約 halt と同等扱い
                self._inc_guard_fire("per_side_dd_both_halt")
                await self._execute_skip(
                    st, side="none", cancel_reason=CR.PER_SIDE_DD_HALT,
                    flush_context="per_side_dd_both_halt",
                    heartbeat=True, multiplier=_halt_mult,
                )
                return True
            else:
                self._inc_guard_fire("per_side_halt_switch")
                logger.info(
                    f"[205# §9.5] Per-side DD halt: {next_side} blocked, "
                    f"switching to {_alt}"
                )
                next_side = _alt

        # ── 205# §9.2: Toxic Fill 同一サイド拒否 ──
        if self._toxic_veto and next_side in self._toxic_veto:
            _alt = self._opposite_side(next_side)
            _alt_blocked = (
                _alt in self._toxic_veto
                or self._daily_drawdown_guard.is_side_halted(_alt)
            )
            if _alt_blocked:
                self._inc_guard_fire("toxic_veto_block")
                self._tick_toxic_veto("both-blocked")
                await self._execute_skip(
                    st, side="none", cancel_reason=CR.TOXIC_FILL_SIDE_VETO,
                    flush_context="toxic_veto_both",
                )
                return True
            else:
                logger.info(
                    f"[205# §9.2] Toxic veto: {next_side} blocked "
                    f"(remaining={self._toxic_veto[next_side]}), "
                    f"switching to {_alt}"
                )
                next_side = _alt

        # ── 238# S-2: Phantom side veto ──
        if (
            self._phantom_guard is not None
            and self._phantom_guard.is_side_vetoed(next_side)
        ):
            _alt = self._opposite_side(next_side)
            if (
                self._phantom_guard.is_side_vetoed(_alt)
                or self._daily_drawdown_guard.is_side_halted(_alt)
            ):
                self._inc_guard_fire("phantom_veto_block")
                await self._execute_skip(
                    st, side="none", cancel_reason=CR.PHANTOM_SIDE_VETO,
                    flush_context="phantom_veto_both",
                )
                return True
            else:
                logger.info(
                    f"[238# phantom_veto] {next_side} blocked → "
                    f"switching to {_alt}"
                )
                next_side = _alt

        ctx.next_side = next_side
        return False

    # ------------------------------------------------------------------
    # Phase 7: Time Filter (086# deadlock 含む)
    # ------------------------------------------------------------------
    async def _apply_time_filter(
        self, st: RunSessionState, ctx: CycleContext,
    ) -> bool:
        """073# side 別時間帯フィルター + 086# deadlock 回避.

        330# extract from run_continuous L698-L871.
        両 side フィルタ時 → True (continue)。
        片側フィルタ → 反対 side に切替 (086# deadlock 考慮)。
        """
        next_side = ctx.next_side
        side_filtered = self._is_time_filtered(side=next_side)

        if not side_filtered:
            # フィルタ対象外 → 通過
            self._time_filter.on_exit()
            return False

        # 反対 side でもフィルタされるか確認
        alt_side = self._opposite_side(next_side)
        alt_filtered = self._is_time_filtered(side=alt_side)

        if alt_filtered:
            # 両 side ともフィルタ → スリープ
            self._inc_guard_fire("time_filter_both_sides")

            if not self._time_filter.in_filter:
                self._time_filter.on_enter()
                st.batch.append(self._make_loop_skip_record(
                    side=next_side,
                    cancel_reason=CR.TIME_FILTER_BOTH_SIDES,
                    order_quantity=0.0,
                ))
            else:
                # 079# heartbeat: 長時間抑制中にプロセス生存を定期ログ
                now_ts = time.time()
                if (
                    now_ts - self._time_filter.last_heartbeat_time
                    >= self.config.heartbeat_interval_sec
                ):
                    utc_h = current_utc_hour()
                    try:
                        import psutil  # lazy import
                        proc = psutil.Process()
                        mem_mb = proc.memory_info().rss / (1024 * 1024)
                        mem_info = f"mem={mem_mb:.1f}MB, "
                    except Exception as e:
                        logger.debug(
                            "psutil memory check unavailable: %s", e, exc_info=True,
                        )
                        mem_info = ""
                    logger.info(
                        f"[heartbeat] Still in time_filter zone "
                        f"(UTC {utc_h}h), "
                        f"{mem_info}"
                        f"unsaved_batch={len(st.batch)}, "
                        f"cycles={self._cycle_count}"
                    )
                    self._time_filter.last_heartbeat_time = now_ts
                    self._update_lock_heartbeat()
                st.batch = self._batch_persistence.maybe_flush(
                    st.batch, "time_filter",
                )
            await self._effective_sleep()
            return True

        # 反対 side は通過 → side 切り替え
        # 086# Bug: alt_side が _last_side と同じ場合、片側蓄積が発生する
        if alt_side == self._last_side:
            self._time_filter.consecutive_086_wait += 1
            max_wait = self.config.max_086_consecutive_wait
            utc_h = current_utc_hour()

            # 110# デッドロック解除: 連続待機が上限を超えたら alt_side を許可
            if (
                max_wait > 0
                and self._time_filter.consecutive_086_wait > max_wait
            ):
                logger.info(
                    f"[time_filter] 086# deadlock break: "
                    f"{self._time_filter.consecutive_086_wait} consecutive waits "
                    f"exceeded max={max_wait}, allowing {alt_side} "
                    f"(110# デッドロック解除)"
                )
                self._time_filter.consecutive_086_wait = 0
                ctx.next_side = alt_side
                self._time_filter.on_exit()
                return False  # 通常フローに合流
            else:
                logger.info(
                    f"[time_filter] {next_side} filtered at UTC {utc_h}h, "
                    f"alt={alt_side} would repeat last side → "
                    f"treating as both-filtered "
                    f"(086# 片側蓄積防止, "
                    f"wait={self._time_filter.consecutive_086_wait}/{max_wait})"
                )
                if not self._time_filter.in_filter:
                    self._time_filter.on_enter()
                    st.batch.append(self._make_loop_skip_record(
                        side=next_side,
                        cancel_reason=CR.TIME_FILTER_086_DEADLOCK,
                        order_quantity=0.0,
                    ))
                # 107# R1: 重複 flush → _maybe_flush_batch 統合
                st.batch = self._batch_persistence.maybe_flush(
                    st.batch, "alt_side==last_side wait",
                )
                await self._effective_sleep()
                return True
        else:
            # 086# ではない通常の side 切り替え → カウンタリセット
            self._time_filter.consecutive_086_wait = 0

        utc_h = current_utc_hour()
        logger.debug(
            f"[time_filter] {next_side} filtered at UTC {utc_h}h, "
            f"switching to {alt_side}"
        )
        ctx.next_side = alt_side
        self._time_filter.on_exit()
        return False

    # ------------------------------------------------------------------
    # 332# Phase 4: Alert mode チェック
    # ------------------------------------------------------------------
    async def _check_alert_mode(self, st: RunSessionState) -> bool:
        """215# P0-C: alert_mode.json オペレータ緊急介入チェック.

        332# extract from run_continuous.
        halt 時はスキップ→ True (continue)。
        非 halt 時は offset/lot/interval mult をインスタンス変数に保存→ False。
        """
        from scripts.v460.lib.alert_mode import load_alert_mode

        _alert = load_alert_mode(self._results_dir)
        if _alert.halt:
            await self._execute_skip(
                st, side="none", cancel_reason=CR.OPERATOR_HALT,
                heartbeat=True, multiplier=self.config.halt_sleep_multiplier,
            )
            return True
        # 非 halt オーバーライド → fill_cycle_executor から参照
        self._alert_offset_mult = _alert.offset_mult
        self._alert_lot_mult = _alert.lot_mult
        self._alert_interval_mult = _alert.interval_mult
        return False

    # ------------------------------------------------------------------
    # 332# Phase 4: CycleContext 初期化 + pre-execution セットアップ
    # ------------------------------------------------------------------
    def _prepare_cycle_context(self) -> CycleContext:
        """332# extract: tick_side_halt + toxic_veto + ctx + regime observability.

        run_continuous の while ループで CycleContext を構築する前の
        共通セットアップを一元化。
        """
        # 205# §9.5: 片側 DD Halt のサイクルカウンタ更新
        self._daily_drawdown_guard.tick_side_halt()

        # 205# §9.2: Toxic Fill 同一サイド拒否 — 初期化のみ
        if self._toxic_veto is None:
            self._toxic_veto = {}

        ctx = CycleContext(
            next_side=self._next_side(),
            regime_mult=self._regime_lot_multiplier(),
        )
        # 420# P1: SideSelector が返した初期 side を記録 (切替前)
        ctx.requested_side = ctx.next_side

        # 310# D: None regime observability (307# F5)
        self._total_regime_cycle_count += 1
        if self._current_regime_value() == "none":
            self._none_regime_cycle_count += 1

        return ctx

    # ------------------------------------------------------------------
    # 332# Phase 4: Regime detector fallback price update
    # ------------------------------------------------------------------
    def _update_regime_fallback(self) -> None:
        """158# §20-A: skip パスでも regime 遷移保証.

        332# extract from run_continuous.
        regime_detector に fallback price を投入し、遷移を検出する。
        """
        if self._regime_detector is None:
            return
        _fb_price, _fb_time = self._maker_price.get_fallback_price()
        if _fb_price is None:
            return
        _pre_regime = self._regime_detector.current_regime
        _regime_result = self._regime_detector.update(
            time.time(), _fb_price,
        )
        # 182# confidence キャッシュ (Trend Mode 厳格化)
        if self._cycle_strategy is not None:
            self._cycle_strategy.update_confidence(_regime_result.confidence)
        if _regime_result.regime != _pre_regime:
            logger.info(
                f"[158# §20-A] Regime transition in main loop: "
                f"{_pre_regime.value} → {_regime_result.regime.value} "
                f"(stability={_regime_result.stability}, "
                f"trend_pct={_regime_result.trend_pct:.4f})"
            )
