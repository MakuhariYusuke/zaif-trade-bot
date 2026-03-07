"""163# Mixin: FillLoopOrchestratorMixin -- run_continuous + ループ制御.

メインオーケストレーションループ: side 選択, skip chain, adaptation, 状態保存。

WARNING -- AI Coding Agent / 人間開発者への注意:
    このファイルは Mixin クラスであり、単独でインスタンス化しないこと。
    FillTestRunner.__init__ で生成される属性に依存する。
    責務: ループ制御 (side kill, time filter, balance forced, adaptation, cleanup)
    1 サイクルの実行ロジック (発注/約定/PnL) は fill_cycle_executor に属する。
    OB ラッパー / SkipGate 評価を追加しないこと。

市場理論的位置づけ (274#)
──────────────────────────
**Inventory Risk Management** (Stoll 1978, Ho & Stoll 1981):
    MM のオーケストレーションは「在庫リスクの動的管理」に帰結する。
    side 選択 (buy/sell alternation) は在庫中立化のための基本操作であり、
    balance_forced_switch は在庫偏重を強制的に修正するオーバーライド。
    Ho-Stoll の最適スプレッドモデルでは、在庫保有量に応じて
    bid/ask を非対称に調整する。本 orchestrator の side 選択は
    この理論の離散的近似に相当する。

**Liveness vs Safety トレードオフ**:
    Market Maker は流動性提供の義務 (liveness) と損失回避 (safety) の
    間でトレードオフに直面する。halt, kill, gate skip は safety 側の措置だが、
    過剰な safety は duty cycle を低下させ MM の存在意義を損なう。
    273# I3/I5/I6 はこのバランスを safety 過剰から liveness 側へ補正した。
    269# §3.2 Liveness Budget はこのトレードオフを明示的に管理する将来施策。

**Avellaneda-Stoikov (2008)** 最適マーケットメイク:
    reservation price + optimal spread モデルは本 bot の
    AS reservation, spread_offset_ratio, min_offset_jpy の理論的基盤。
    orchestrator は A-S の「状態 → 最適行動」写像を
    離散サイクルで逐次近似する。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.alert_mode import load_alert_mode
from scripts.v460.lib.micro_circuit_breaker import MCBLevel
from scripts.v460.lib.orchestrator_guards import OrchestratorGuardsMixin
from scripts.v460.lib.orchestrator_lifecycle import OrchestratorLifecycleMixin
from scripts.v460.lib.orchestrator_post_cycle import OrchestratorPostCycleMixin
from scripts.v460.lib.orchestrator_pre_cycle import CycleContext, OrchestratorPreCycleMixin
from scripts.v460.lib.spread_anomaly_detector import SADLevel

if TYPE_CHECKING:
    from scripts.v460.lib.micro_circuit_breaker import MicroCircuitBreaker
    from scripts.v460.lib.phantom_position_guard import PhantomPositionGuard
    from scripts.v460.lib.spread_anomaly_detector import SpreadAnomalyDetector
    from ztb.metrics.fill_quality import FillRecord
    from ztb.risk.sell_dynamic_kill import DynamicKillManager

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 265# RunSessionState: run_continuous ローカル変数のカプセル化
# ------------------------------------------------------------------
@dataclass
class RunSessionState:
    """run_continuous 内のループ間共有状態.

    265# extract: run_continuous の >20 ローカル変数を構造化し、
    extract method 間での受渡しを型安全に行う。
    """

    total_count: int = 0
    filled_count: int = 0
    cumulative_pnl_jpy: float = 0.0
    cumulative_btc_delta: float = 0.0
    cumulative_adverse_count: int = 0
    cumulative_adverse_bps: float = 0.0
    # 286# 283# P1-5: 強制買い KPI 分離トラッキング
    forced_buy_fill_count: int = 0
    forced_buy_pnl_sum_bps: float = 0.0
    normal_buy_fill_count: int = 0
    normal_buy_pnl_sum_bps: float = 0.0
    batch: list[FillRecord] = field(default_factory=list)
    batch_size: int = 10


class FillLoopOrchestratorMixin(
    OrchestratorGuardsMixin,
    OrchestratorLifecycleMixin,
    OrchestratorPostCycleMixin,
    OrchestratorPreCycleMixin,
):
    """run_continuous + side kill / filter / adaptation / cleanup (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: ループ制御 (run_continuous), skip chain 評価
      NG: 1 サイクル実行, OB 取得, SkipGate 評価, PnL 計測
    325# God Object 分割: Guard/Lifecycle/PostCycle を Mixin に抽出
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
    # 224# B1: halt解除後ソフトリカバリ lot 倍率 (orchestrator が設定、executor が参照)
    _halt_recovery_lot_mult: float = 1.0
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
    # 218# デッドロック検出: 連続ゲートブロックカウンタ
    _consecutive_gate_blocks: int = 0
    # 236# hasattr 排除: クラスレベルデフォルトで属性存在を保証
    _trending_sell_skip_count: int = 0
    _balance_forced_skip_count: int = 0
    # 234# 縮退清算モード duty cycle カウンタ
    _degraded_liquidation_duty_counter: int = 0
    # 269# Inventory Escape Mode duty cycle カウンタ
    _inventory_escape_duty_counter: int = 0
    # 234# one-sided エスカレーション: cooldown 残サイクル
    _one_sided_cooldown_remaining: int = 0
    _one_sided_freeze_remaining: int = 0
    # 286# 284# P1: 強制買い遅延実行 (Glosten-Milgrom 1985)
    _forced_buy_delay_remaining: int = 0
    # 294# P0: 連続ブロックカウンタ (デッドロック防止)
    _forced_buy_delay_consecutive: int = 0
    # 303# B: DD soft lot side 分離 — side 別 lot 倍率 (1.0 = 通常)
    _dd_soft_lot_scale_buy: float = 1.0
    _dd_soft_lot_scale_sell: float = 1.0
    # 310# D: None regime observability (307# F5)
    _none_regime_cycle_count: int = 0
    _total_regime_cycle_count: int = 0
    # 250# P1-4: freeze/cooldown が紐付いた side
    # — None 時は全 side スキップ (後方互換), side 指定時はその side のみ
    _one_sided_frozen_side: str | None = None
    # 223# skip-time state save: 最終 state save のモノトニック時刻
    _last_state_save_time: float = 0.0
    #: 223# skip パス中の state save 間隔 (秒)
    #: 277# 理論的導出: max(300, cycle_interval × 3) — 3 サイクル分の最低間隔。
    #: 過度な I/O を避けつつ再起動時の巻き戻しを 5 分以内に抑える。
    _STATE_SAVE_INTERVAL_SEC: float = 300.0
    # 228# H3: MCB/SAD/CycleStrategy class-level None defaults (hasattr 排除用)
    # 296# B-17: object → 具象型化 (TYPE_CHECKING)
    _mcb: MicroCircuitBreaker | None = None
    _sad: SpreadAnomalyDetector | None = None
    _cycle_strategy: object | None = None
    # 237# PhantomPositionGuard class-level default (hasattr 排除)
    # 238# C-1: object → PhantomPositionGuard 型安全化 (TYPE_CHECKING)
    _phantom_guard: PhantomPositionGuard | None = None
    # 254# _recent_records: _check_stop_conditions が参照。テスト注入用。
    # 256# list → deque(maxlen=200): batch save でリセットされない累積バッファ
    _recent_records: deque[FillRecord] = deque(maxlen=200)  # type: ignore[assignment]
    # 254# _heartbeat_task: run_continuous 内で代入、cleanup_heartbeat で参照
    _heartbeat_task: asyncio.Task[None] | None = None


    # ------------------------------------------------------------------
    # 179# S1: _effective_sleep — regime 応答サイクル間隔の一元化
    # ------------------------------------------------------------------
    async def _effective_sleep(
        self, *, multiplier: float = 1.0, max_override: float = 0.0,
    ) -> None:
        """179# CycleStrategy に委譲し、regime 別サイクル間隔で sleep.

        skip/halt/error continue 全パスがこのメソッドを経由する。
        - multiplier=1.0 : 通常スキップ
        - multiplier=config.halt_sleep_multiplier : halt (daily drawdown) — 276#
        - multiplier=config.phantom_detection_sleep_multiplier : phantom 検出 — 277#
        - max_override>0  : 242# quiescence 時の sleep 上限オーバーライド
        正常サイクル完了パスは rapid_exit ロジックを含むため直接呼ばない。
        200# P0-2: _soft_drawdown_interval_multiplier を追加乗算。
        """
        regime = self._current_regime_value()
        base = self._cycle_strategy.effective_interval(regime)
        # 200# P0-2: soft drawdown で lot 半減不可 → interval 延長
        soft_dd_mult = self._soft_drawdown_interval_multiplier
        # 217# fix: alert_mode の interval_mult をスキップ/halt パスにも適用
        alert_im = self._alert_interval_mult
        _raw = base * multiplier * soft_dd_mult * alert_im
        # 211#: max_cycle_sleep_sec キャップを _effective_sleep にも適用
        # 242#: max_override > 0 なら quiescence 用の拡大上限を使用
        _max = max_override if max_override > 0 else self.config.max_cycle_sleep_sec
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
    # 276# DRY: _execute_skip — skip ceremony 共通ヘルパー
    # ------------------------------------------------------------------
    async def _execute_skip(
        self,
        st: RunSessionState,
        *,
        side: str,
        cancel_reason: str,
        flush_context: str = "",
        order_quantity: float = 0.0,
        heartbeat: bool = False,
        state_save: bool = False,
        state_save_context: str = "",
        update_last_side: bool = False,
        sleep: bool = True,
        multiplier: float = 1.0,
        max_override: float = 0.0,
        **record_kwargs: object,
    ) -> None:
        """run_continuous skip パスの record → flush → sleep 一連処理を一元化.

        22 箇所の blocking decision point に共通する 5-7 行の skip ceremony
        (record 生成 → batch append → total_count++ → flush → heartbeat →
        state_save → last_side 更新 → sleep) を単一呼出に集約する。

        呼び出し側は ``await self._execute_skip(st, ...); continue`` のみ。

        理論的根拠:
          Skip ceremony はインフラ的処理 (observability / persistence /
          heartbeat) であり、blocking decision ロジック (Amihud 2002 非流動性
          コスト回避) とは直交する。重複は SRP 違反であり、変更時の一貫性
          リスク (268# incident の遠因) を生む。
        """
        record = self._make_loop_skip_record(
            side=side,
            cancel_reason=cancel_reason,
            order_quantity=order_quantity,
            **record_kwargs,
        )
        st.batch.append(record)
        st.total_count += 1
        st.batch = self._batch_persistence.maybe_flush(
            st.batch, flush_context or cancel_reason,
        )
        if heartbeat:
            self._update_lock_heartbeat()
        if state_save:
            self._maybe_skip_state_save(
                st, state_save_context or cancel_reason,
            )
        if update_last_side:
            self._last_side = side
        if sleep:
            await self._effective_sleep(
                multiplier=multiplier, max_override=max_override,
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


    async def run_continuous(self, hours: float) -> list[FillRecord]:
        """指定時間、連続してサイクルを実行.

        009# §4.4: 7 日間 (168h) の実測想定.
        中断→再開時は既存 fill_records を自動復元 (レジューム対応).

        024# R1-R4: 保存失敗耐性・例外分離・メモリ制御を強化.
        032# P0: 方策 A パラメータ適応統合.
        033# 方策 B: 動的ロットサイジング統合.
        033# F4: 累積 PnL 安全キャップ (000# §3.9).
        265# extract: _init_run_session / _process_post_cycle /
              _log_progress_and_adapt / _finalize_run に分割。
        """
        end_time = time.time() + hours * 3600

        # 265# extract: 初期化ロジック (~200行) を分離
        st = await self._init_run_session()

        # 148# P0: heartbeat 更新タスク — stale 誤判定防止
        async def _heartbeat_loop() -> None:
            """lock heartbeat を周期的に更新."""
            while not self._kill_switch.is_killed():
                self._update_lock_heartbeat()
                await asyncio.sleep(self.config.lock_heartbeat_period_sec)

        heartbeat_task = asyncio.create_task(_heartbeat_loop())
        self._heartbeat_task: asyncio.Task[None] | None = heartbeat_task  # 175# cleanup 用

        logger.info(f"Starting fill test: {hours}h, interval={self.config.cycle_interval_sec}s")

        # 223# skip-time state save: ループ開始時にタイムスタンプ初期化
        self._last_state_save_time = time.monotonic()

        while time.time() < end_time and not self._kill_switch.is_killed():
            # ── 330# Phase 1: 日替わりリセット ──
            self._process_daily_reset()

            # ── 330# Phase 2: DD halt チェック ──
            if await self._handle_dd_halt(st):
                continue

            # 215# P0-C: alert_mode.json — オペレータ緊急介入チェック
            _alert = load_alert_mode(self._results_dir)
            if _alert.halt:
                await self._execute_skip(
                    st, side="none", cancel_reason=CR.OPERATOR_HALT,
                    heartbeat=True, multiplier=self.config.halt_sleep_multiplier,
                )
                continue
            # 215# P0-C: 非 halt オーバーライドをインスタンス変数に保存
            # (fill_cycle_executor から参照)
            self._alert_offset_mult = _alert.offset_mult
            self._alert_lot_mult = _alert.lot_mult
            self._alert_interval_mult = _alert.interval_mult

            # ── 330# Phase 3: Circuit Breakers (MCB + SAD + Escalation) ──
            if await self._check_circuit_breakers(st):
                continue

            # ── 330# Phase 4: Hard Skip UTC ──
            if await self._check_hard_skip_utc(st):
                continue

            # ── 330# Phase 5: Phantom Guard ──
            await self._process_phantom_guard()

            # 205# §9.5: 片側 DD Halt のサイクルカウンタ更新
            self._daily_drawdown_guard.tick_side_halt()

            # 205# §9.2: Toxic Fill 同一サイド拒否 — 初期化のみ
            if self._toxic_veto is None:
                self._toxic_veto = {}

            # 330# CycleContext 初期化
            ctx = CycleContext(next_side=self._next_side())

            # 310# D: None regime observability (307# F5)
            self._total_regime_cycle_count += 1
            _current_regime_str = self._current_regime_value()
            if _current_regime_str == "none":
                self._none_regime_cycle_count += 1

            # ── 330# Phase 6: Side Veto 解決 ──
            if await self._resolve_side_vetos(st, ctx):
                continue

            # ── 330# Phase 7: Time Filter ──
            if await self._apply_time_filter(st, ctx):
                continue

            # 330# CycleContext → local vars (下流互換)
            next_side = ctx.next_side
            _balance_forced = False
            _is_rescue = False
            _one_sided_balance = False
            _inventory_escape = False

            # 158# §20-A: skip パスでも regime 遷移保証 (fallback price 投入)
            if self._regime_detector is not None:
                _fb_price, _fb_time = self._maker_price.get_fallback_price()
                if _fb_price is not None:
                    _pre_regime = self._regime_detector.current_regime
                    _regime_result = self._regime_detector.update(
                        time.time(), _fb_price
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

            # 041# 残高 pre-flight check: 不足サイドはスキップ
            # 145# §8-#1: レジーム倍率込みで残高判定 (preflight-lot alignment)
            _regime_mult = self._regime_lot_multiplier()
            if await self._check_balance_for_side(next_side, regime_mult=_regime_mult):
                # 091# 即座に反対 side を試す: time_filter との組合せで停滞するのを防止
                opposite = self._opposite_side(next_side)
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
                    # 223# P0: balance_forced 後に per-side halt を再チェック
                    # (222# 1.1 CRITICAL: halt 中の side を balance_forced で貫通するバグの修正)
                    if self._daily_drawdown_guard.is_side_halted(next_side):
                        # ────────────────────────────────────────────
                        # 269# P0: Inventory Escape Mode
                        # Codex 269# §4.1 / Gemini 270# Action A:
                        # balance_forced(= 元 side 残高不足) + per-side halt(= 切替先 halt)
                        # → 両 side ブロックのデッドロック。
                        # halt を一時的に貫通し、degraded liquidation パラメータで
                        # 縮退売却を実行して在庫を解消する。
                        # ────────────────────────────────────────────
                        _ie_enabled = self.config.inventory_escape_enabled
                        _ie_duty = max(self.config.inventory_escape_duty_cycle, 1)
                        # 281# fix: Inventory Escape を双方向化
                        # 269# 当初は sell 方向のみ (BTC 過剰在庫の縮退清算)
                        # しかし逆パターン (BTC=0 + buy halt) でもデッドロック発生。
                        # buy 方向でも degraded params で縮退取得を許可する。
                        if _ie_enabled:
                            self._inventory_escape_duty_counter += 1
                            if _ie_duty > 1 and (self._inventory_escape_duty_counter % _ie_duty) != 1:
                                # duty cycle スキップ: halt 貫通は控えめに
                                logger.info(
                                    f"[269#] Inventory escape duty skip: "
                                    f"cycle {self._inventory_escape_duty_counter}/{_ie_duty}"
                                )
                                self._inc_guard_fire("inventory_escape_duty_skip")
                            else:
                                # duty cycle 実行回: halt を貫通して縮退清算
                                logger.warning(
                                    f"[269#] INVENTORY ESCAPE: bypassing per-side halt "
                                    f"for {next_side} (balance_forced deadlock breakout, "
                                    f"cycle {self._inventory_escape_duty_counter}/{_ie_duty})"
                                )
                                self._inc_guard_fire("inventory_escape_active")
                                _inventory_escape = True
                                # toxic_veto の減算は通常同様に行う (226# S2)
                                self._tick_toxic_veto("inventory_escape")
                                # halt 貫通 → degraded liquidation として以降のパスに進む
                                # (ループの continue をスキップして実行パスへ fallthrough)
                        else:
                            _inventory_escape = False

                        if not _inventory_escape:
                            logger.warning(
                                f"[223#] balance_forced → {next_side} is per-side halted — "
                                f"refusing to bypass halt (safety > liveness)"
                            )
                            self._inc_guard_fire("balance_forced_halt_block")
                            # 281# fix: 273# I3 の untick_side_halt() を除去
                            # balance_forced + halt_block の組合せでは untick が
                            # halt カウントダウンを完全停止させ永久デッドロック化する。
                            # (実例: BTC=0 + buy halt → 8時間以上の完全停止)
                            # halt を自然にカウントダウンさせ、per_side_halt_cycles
                            # 経過後に解除。reanchor (269#) が再halt 基準をリセット。
                            # 226# S2: balance_forced + halt_block で continue する際、
                            # toxic_veto のカウンタも減算する。
                            self._tick_toxic_veto("halt_block")
                            await self._execute_skip(
                                st, side=next_side,
                                cancel_reason=CR.PER_SIDE_DD_HALT,
                                order_quantity=self._current_lot,
                                balance_forced_switch=True,
                                flush_context="balance_forced_halt_recheck",
                                state_save=True,
                                state_save_context="balance_forced_halt_block",
                                update_last_side=True,
                            )
                            continue
                    # 200# E: 時間ベース頻度検出 — 短時間で連続 balance_forced が発生 → 警告
                    _now = time.time()
                    _last_bf_time = self._last_balance_forced_time
                    _bf_cooldown = self.config.balance_forced_cooldown_sec
                    if _bf_cooldown > 0 and (_now - _last_bf_time) < _bf_cooldown:
                        _bf_freq_count = self._balance_forced_freq_count + 1
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
                    # 225# 5.2: fire count 記録
                    self._inc_guard_fire("preflight_insufficient")

                    # 140# §8.1-#2: preflight skip record 生成 (132# F4)
                    # 145# §9-#5: _make_skip_record DRY 化
                    st.batch.append(self._make_loop_skip_record(
                        side=next_side,
                        cancel_reason=CR.PREFLIGHT_INSUFFICIENT,
                        order_quantity=self._current_lot,
                    ))
                    # 107# R1: 重複 flush → _maybe_flush_batch 統合
                    st.batch = self._batch_persistence.maybe_flush(st.batch, "preflight skip")

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
                        st.batch.append(self._make_loop_skip_record(
                            timestamp=_pause_record_ts,
                            side="none",
                            cancel_reason=CR.PREFLIGHT_PAUSE,
                            cycle_id=(
                                f"preflight_pause_{self._preflight_pause_count}_"
                                f"{int(_pause_record_ts)}"
                            ),
                            order_quantity=0.0,
                        ))
                        st.batch = self._batch_persistence.maybe_flush(st.batch, "preflight_pause")
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

            # ────────────────────────────────────────────────────
            # 234# one-sided エスカレーション: cooldown / freeze 発動チェック
            # ────────────────────────────────────────────────────
            # 250# P1-4: freeze/cooldown は紐付いた side のみスキップ
            # — 反対 side は通常実行を許可 (247# §1.10)
            _frozen_side = self._one_sided_frozen_side  # None = all sides
            if self._one_sided_freeze_remaining > 0:
                if _frozen_side is None or _frozen_side == next_side:
                    self._one_sided_freeze_remaining -= 1
                    logger.info(
                        f"[234#] One-sided FREEZE active: skipping {next_side} "
                        f"(frozen_side={_frozen_side}, "
                        f"remaining={self._one_sided_freeze_remaining})"
                    )
                    self._inc_guard_fire("one_sided_freeze_skip")
                    await self._execute_skip(
                        st, side=next_side,
                        cancel_reason=CR.ONE_SIDED_FREEZE_SKIP,
                        order_quantity=self._current_lot,
                        update_last_side=True,
                    )
                    continue
                # 250#: frozen_side と異なる side → スキップせず通過
                logger.debug(
                    f"[250#] Freeze side={_frozen_side}, current={next_side} — pass through"
                )

            if self._one_sided_cooldown_remaining > 0:
                if _frozen_side is None or _frozen_side == next_side:
                    self._one_sided_cooldown_remaining -= 1
                    logger.info(
                        f"[234#] One-sided COOLDOWN skip: "
                        f"frozen_side={_frozen_side}, "
                        f"remaining={self._one_sided_cooldown_remaining}"
                    )
                    self._inc_guard_fire("one_sided_cooldown_skip")
                    await self._execute_skip(
                        st, side=next_side,
                        cancel_reason=CR.ONE_SIDED_COOLDOWN_SKIP,
                        order_quantity=self._current_lot,
                        update_last_side=True,
                    )
                    continue
                logger.debug(
                    f"[250#] Cooldown side={_frozen_side}, current={next_side} — pass through"
                )

            # 133# P0-08 / 154# C-1/C-2: balance_forced スキップ + deadlock 防止
            if _balance_forced and self.config.skip_balance_forced:
                # 154# C-1: 両側残高判定
                original_side = self._opposite_side(next_side)
                original_also_insufficient = await self._check_balance_for_side(
                    original_side, regime_mult=_regime_mult
                )
                # 154# C-2 + 182# regime 別緩和: trending 時は deadlock_limit 引き上げ
                _r = self._current_regime_value()
                _deadlock_limit = (
                    self._cycle_strategy.policy.deadlock_limit_trending
                    if _r and _r.startswith("trending") and self._cycle_strategy is not None
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
                    # 167# DL-5: _last_side を更新 (rescue=true 時は到達しないが防御的に)
                    await self._execute_skip(
                        st, side=next_side,
                        cancel_reason=CR.BALANCE_FORCED_SKIP,
                        order_quantity=self._current_lot,
                        balance_forced_switch=True,
                        balance_forced_consecutive=self._balance_forced_skip_count,
                        update_last_side=True,
                    )
                    continue

            # ════════════════════════════════════════════════════════════
            # 286# 284# P1: 強制買い遅延実行 (Glosten-Milgrom 1985)
            # balance_forced で buy 方向に切り替わった際、microprice が
            # 急落中なら N サイクル待機。逆選択リスクが高い局面での
            # 即時買いは損失を拡大するだけ (「待つ勇気」)。
            # ════════════════════════════════════════════════════════════
            if (
                _balance_forced
                and next_side == "buy"
                and self.config.forced_buy_delay_enabled
            ):
                _vel = self._maker_price.last_mid_trend_bps
                _thr = self.config.forced_buy_delay_velocity_threshold_bps
                # 292# P1: ranging/trending_down では緩い閾値を適用
                _ranging_thr = self.config.forced_buy_delay_velocity_threshold_ranging_bps
                if _ranging_thr is not None and self._regime_detector is not None:
                    _cur_regime = self._regime_detector.current_regime.value
                    if _cur_regime in ("ranging", "trending_down"):
                        _thr = _ranging_thr
                # 294# P0: 連続ブロック上限チェック — デッドロック防止
                _max_consec = self.config.forced_buy_delay_max_consecutive
                if (
                    _vel is not None
                    and _vel <= _thr
                    and self._forced_buy_delay_consecutive < _max_consec
                ):
                    self._forced_buy_delay_remaining = max(
                        self._forced_buy_delay_remaining,
                        self.config.forced_buy_delay_cycles,
                    )
                    logger.info(
                        f"[286# GM delay] Forced buy delayed: "
                        f"velocity={_vel:.2f}bps <= {_thr:.1f}bps, "
                        f"waiting {self._forced_buy_delay_remaining} cycles "
                        f"(consec={self._forced_buy_delay_consecutive}/{_max_consec})"
                    )
                elif self._forced_buy_delay_consecutive >= _max_consec:
                    # デッドロック突破: カウンタをリセットして通過させる
                    self._forced_buy_delay_remaining = 0
                    logger.warning(
                        f"[294# GM deadlock break] Forced buy delay exceeded "
                        f"max_consecutive={_max_consec}, forcing through. "
                        f"velocity={_vel}bps, regime={getattr(getattr(self._regime_detector, 'current_regime', None), 'value', 'N/A')}"
                    )

            if self._forced_buy_delay_remaining > 0 and next_side == "buy":
                self._forced_buy_delay_remaining -= 1
                self._forced_buy_delay_consecutive += 1
                self._inc_guard_fire("forced_buy_delay")
                await self._execute_skip(
                    st, side=next_side,
                    cancel_reason=CR.FORCED_BUY_DELAY,
                    order_quantity=self._current_lot,
                    balance_forced_switch=_balance_forced,
                    update_last_side=True,
                )
                continue
            else:
                # delay を通過 → 連続カウンタをリセット
                self._forced_buy_delay_consecutive = 0

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

            # 241# C-2 fix: toxicity 評価を check_kill() の前に実行
            # check_kill() は _cooldown をデクリメントする副作用があるため、
            # assess_toxicity() を先に呼ばないと最終 cooldown サイクルで
            # 状態が不整合になる (check_kill → killed=True, assess → GREEN)
            _buy_tox = self._assess_buy_toxicity()
            _sell_tox = self._assess_sell_toxicity()

            # 273# I6: halt 解除後の soft gate grace period
            # recovery 期間中はソフトゲート (trending_sell_skip, velocity_skip,
            # unknown_regime) を緩和して再参入速度を改善する
            _halt_recovery_active = self._daily_drawdown_guard.is_in_recovery(
                next_side
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
                is_buy_killed=self._is_side_killed("buy"),
                is_sell_killed=self._is_side_killed("sell"),
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
                # 240# Toxicity Budget (232# §2.2)
                buy_toxicity=_buy_tox,
                sell_toxicity=_sell_tox,
                # 273# I6: halt 解除後の soft gate grace period
                halt_recovery_active=_halt_recovery_active,
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

                # 276# DRY: record → flush → last_side を _execute_skip に委譲
                # sleep は quiescence / narrow_spread_pause 分岐があるため別途処理
                await self._execute_skip(
                    st, side=next_side,
                    cancel_reason=_gate_result.cancel_reason,
                    order_quantity=self._current_lot,
                    update_last_side=True, sleep=False,
                )

                # 224# guard_fire_counts: ゲートブロック理由を記録
                if _gate_result.blocking_reason:
                    self._inc_guard_fire(f"gate_{_gate_result.blocking_reason}")

                # 218#/242# 連続ゲートブロック検出 + quiescence 状態遷移
                self._consecutive_gate_blocks += 1
                _quiescence_th = self.config.quiescence_gate_blocks_threshold
                _in_quiescence = (
                    _quiescence_th > 0
                    and self._consecutive_gate_blocks >= _quiescence_th
                )
                # 277# gate block ログ間隔: quiescence 閾値の半分で導出 (min 5)
                _gate_log_interval = max(
                    5, self.config.quiescence_gate_blocks_threshold // 2,
                )
                if (
                    self._consecutive_gate_blocks >= _gate_log_interval
                    and self._consecutive_gate_blocks % _gate_log_interval == 0
                ):
                    if _in_quiescence:
                        # 242# quiescence: No Trade は正常系
                        self._inc_guard_fire("quiescence")
                        logger.info(
                            f"[242#] QUIESCENCE: {self._consecutive_gate_blocks} "
                            f"consecutive gate blocks — no-trade accepted as normal "
                            f"(reason={_gate_result.blocking_reason}, side={next_side}, "
                            f"sleep_cap={self.config.quiescence_sleep_sec:.0f}s)"
                        )
                    else:
                        logger.warning(
                            f"[218#] DEADLOCK WARNING: {self._consecutive_gate_blocks} "
                            f"consecutive gate blocks (reason={_gate_result.blocking_reason}, "
                            f"side={next_side})"
                        )

                # 223# skip-time lightweight state save: gate_block 連続中も
                # _STATE_SAVE_INTERVAL_SEC 経過ごとに state 保存して stale 防止
                self._maybe_skip_state_save(
                    st, f"gate_blocks={self._consecutive_gate_blocks}"
                )

                # 197# narrow_spread_pause: Gate 8 ブロック時は pause_sec 分待機
                # 242# quiescence: sleep 上限を引き上げ (max_cycle_sleep → quiescence_sleep)
                _q_sleep = (
                    self.config.quiescence_sleep_sec
                    if _in_quiescence and self.config.quiescence_sleep_sec > 0
                    else 0.0
                )
                if _gate_result.blocking_reason == "narrow_spread_pause":
                    await asyncio.sleep(self.config.narrow_spread_pause_sec)
                else:
                    await self._effective_sleep(max_override=_q_sleep)
                continue
            else:
                # ゲート通過 → カウンタリセット
                self._consecutive_gate_blocks = 0  # 218# デッドロック解消
                # 223# DUAL KILL bypass 発動時のメトリクス記録
                if _gate_result.dual_kill_bypassed:
                    self._inc_guard_fire("dual_kill_bypass")
                if (
                    self.config.skip_sell_trending
                    and next_side == "sell"
                    and self._regime_detector is not None
                    and self._regime_detector.current_regime.is_trending
                ):
                    self._trending_sell_skip_count = 0

            # ────────────────────────────────────────────────────
            # 240# Toxicity Budget: 確率的参加率チェック
            # ORANGE ゾーンでは 1/N の確率で参加 (Glosten-Milgrom)
            # ────────────────────────────────────────────────────
            if (
                not _gate_result.blocked
                and _gate_result.participation_rate < 1.0
                and random.random() > _gate_result.participation_rate
            ):
                self._inc_guard_fire("toxicity_participation_skip")
                logger.info(
                    f"[240#] Toxicity participation skip: "
                    f"rate={_gate_result.participation_rate:.2f}, "
                    f"offset_mult={_gate_result.toxicity_offset_mult:.2f} "
                    f"(side={next_side})"
                )
                await self._execute_skip(
                    st, side=next_side,
                    cancel_reason=CR.TOXICITY_PARTICIPATION_SKIP,
                    order_quantity=self._current_lot,
                    update_last_side=True,
                )
                continue

            # ────────────────────────────────────────────────────
            # 234# 縮退清算モード: balance_forced + Kill Gate blocked
            # → 完全 block ではなく min lot + wide offset + duty cycle で縮退実行
            # ────────────────────────────────────────────────────
            _degraded_liquidation = _gate_result.degraded_liquidation
            if _degraded_liquidation:
                self._degraded_liquidation_duty_counter += 1
                _duty = max(self.config.degraded_liquidation_duty_cycle, 1)
                # 235# duty_cycle=1 は「毎回実行」と同義 (skip なし)
                if _duty > 1 and (self._degraded_liquidation_duty_counter % _duty) != 1:
                    # duty cycle スキップ: N サイクルに 1 回のみ実行
                    logger.info(
                        f"[234#] Degraded liquidation duty skip: "
                        f"cycle {self._degraded_liquidation_duty_counter}/{_duty} "
                        f"(reason={_gate_result.degraded_reason})"
                    )
                    self._inc_guard_fire("degraded_liquidation_duty_skip")
                    await self._execute_skip(
                        st, side=next_side,
                        cancel_reason=CR.DEGRADED_LIQUIDATION_DUTY_SKIP,
                        order_quantity=self._current_lot,
                        update_last_side=True,
                    )
                    continue
                # duty cycle 実行回: 進む
                self._inc_guard_fire("degraded_liquidation_active")
                logger.warning(
                    f"[234#] Degraded liquidation ACTIVE: "
                    f"lot ×{self.config.degraded_liquidation_lot_mult:.1f}, "
                    f"offset ×{self.config.degraded_liquidation_offset_mult:.1f} "
                    f"(reason={_gate_result.degraded_reason})"
                )
            else:
                # 正常パスではカウンタリセット
                # 235# B-4 fix: トグル対策—転落先のカウンタを引き継ぎず完全リセット
                if self._degraded_liquidation_duty_counter > 0:
                    logger.info(
                        f"[235#] Degraded liquidation cleared after "
                        f"{self._degraded_liquidation_duty_counter} duty cycles"
                    )
                self._degraded_liquidation_duty_counter = 0
                # 269# Inventory Escape カウンタもリセット
                if self._inventory_escape_duty_counter > 0:
                    logger.info(
                        f"[269#] Inventory escape cleared after "
                        f"{self._inventory_escape_duty_counter} duty cycles"
                    )
                self._inventory_escape_duty_counter = 0

            try:
                # 224# B1: halt解除後ソフトリカバリ — lot 縮小倍率を算出
                _recovery_scale = self._daily_drawdown_guard.consume_recovery_cycle(
                    next_side
                )
                if _recovery_scale < 1.0:
                    self._inc_guard_fire("per_side_halt_recovery_active")
                    # 225# 市場理論補強: regime-aware recovery scaling
                    # Avellaneda-Stoikov 原理: 在庫リスクはボラティリティに比例
                    # trending regime → AS リスク残存 → さらに保守的 (×0.7)
                    # ranging regime → mean reversion 期待 → 通常スケール
                    if (
                        self._regime_detector is not None
                    ):
                        _regime = self._regime_detector.current_regime
                        if _regime is not None and _regime.is_trending:
                            _recovery_scale *= self.config.recovery_trending_penalty
                            logger.info(
                                f"[225#] Recovery penalty: trending regime → "
                                f"scale={_recovery_scale:.3f}"
                            )
                        elif _regime is not None and _regime.is_high_vol:
                            _recovery_scale *= self.config.recovery_high_vol_penalty
                            logger.info(
                                f"[225#] Recovery penalty: high_vol regime → "
                                f"scale={_recovery_scale:.3f}"
                            )
                self._halt_recovery_lot_mult = _recovery_scale

                record = await self.run_single_cycle(
                    side_override=next_side,
                    balance_forced_switch=_balance_forced,
                    balance_forced_rescue=_is_rescue,
                    one_sided_balance=_one_sided_balance,
                    trending_offset_mult=_gate_result.trending_offset_mult,
                    degraded_liquidation=_degraded_liquidation or _inventory_escape,
                    toxicity_offset_mult=_gate_result.toxicity_offset_mult,
                )
                # 154# C-2: 実サイクル実行 → forced skip カウンタリセット
                self._balance_forced_skip_count = 0
                # 158# §20-B: 実サイクル実行 → trending sell skip カウンタリセット
                self._trending_sell_skip_count = 0
                # 207# §4: one-sided 連続実行追跡 (205# §4.2 Codex)
                # 234# エスカレーション: limit → cooldown → freeze
                if _one_sided_balance:
                    self._one_sided_consecutive_count += 1
                    _os_limit = self.config.one_sided_consecutive_limit
                    if _os_limit > 0 and self._one_sided_consecutive_count >= _os_limit:
                        _over = self._one_sided_consecutive_count - _os_limit
                        # 234# Stage 3: freeze (limit + freeze_offset 以上)
                        _freeze_off = self.config.one_sided_escalation_freeze_offset
                        _cd_off = self.config.one_sided_escalation_cooldown_offset
                        if _freeze_off > 0 and _over >= _freeze_off:
                            _freeze_n = self.config.one_sided_escalation_freeze_cycles
                            self._one_sided_freeze_remaining = _freeze_n
                            # 250# P1-4: freeze を発動した side を記録
                            self._one_sided_frozen_side = next_side
                            # 235# B-5 fix: freeze 発動時にカウンタを limit まで巻き戻し
                            # freeze 消化後の即再発動を防止
                            self._one_sided_consecutive_count = _os_limit
                            self._inc_guard_fire("one_sided_freeze")
                            logger.warning(
                                f"[234#] One-sided FREEZE: "
                                f"{self._one_sided_consecutive_count}/{_os_limit} "
                                f"(+{_over}) → freezing {next_side} for {_freeze_n} cycles"
                            )
                        # 234# Stage 2: cooldown (limit + cooldown_offset 以上)
                        elif _cd_off > 0 and _over >= _cd_off:
                            _cd_n = self.config.one_sided_escalation_cooldown_cycles
                            self._one_sided_cooldown_remaining = _cd_n
                            # 250# P1-4: cooldown を発動した side を記録
                            self._one_sided_frozen_side = next_side
                            # 235# B-5 fix: cooldown 発動時もカウンタ巻き戻し
                            self._one_sided_consecutive_count = _os_limit
                            self._inc_guard_fire("one_sided_cooldown")
                            logger.warning(
                                f"[234#] One-sided COOLDOWN: "
                                f"{self._one_sided_consecutive_count}/{_os_limit} "
                                f"(+{_over}) → skip {_cd_n} cycles"
                            )
                        # 234# Stage 1: interval 延長 (既存)
                        else:
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
                    self._one_sided_cooldown_remaining = 0
                    self._one_sided_freeze_remaining = 0
                    self._one_sided_frozen_side = None  # 250# reset
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt — stopping gracefully")
                self._kill_switch.kill("keyboard_interrupt")
                break
            except Exception as e:
                # 024# R2: 例外分類 — サイクル実行エラーは継続可能
                logger.error(f"Cycle execution error: {e}", exc_info=True)
                # 225# 6.1: recovery counter が消費済みなのに lot 縮小が適用されなかった
                # → カウンタを復元して次サイクルで再適用
                if _recovery_scale < 1.0:
                    self._daily_drawdown_guard.restore_recovery_counter(next_side)
                    logger.info(
                        f"[225# 6.1] Recovery counter restored for {next_side} "
                        f"(cycle aborted by exception)"
                    )
                # 128# 例外時も dust sweep ロットを復元
                self._balance_checker.restore_lot_after_dust_sweep()
                # 166# SR-4: 例外 continue でも side 交互を保証
                self._last_side = next_side
                await self._effective_sleep()  # 179# S1
                continue

            # 128# dust sweep 後のロット復元 (サイクル完了ごとに確実に実行)
            self._balance_checker.restore_lot_after_dust_sweep()

            # 265# extract: post-cycle 処理 (~150行) を分離
            self._process_post_cycle(record, next_side, st)

            # 265# extract: 進捗ログ + state save + adaptation (~125行) を分離
            await self._log_progress_and_adapt(next_side, st)

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
                    # 306# L1: σ 連動 dynamic cycle interval
                    if self.config.dynamic_cycle_interval_enabled:
                        interval = self._compute_dynamic_interval(interval)
                # 200# P0-2: soft drawdown interval 延長
                soft_dd_mult = self._soft_drawdown_interval_multiplier
                # 202# A: 単一サイクル大損失クールダウン (1回適用で自動リセット)
                _loss_cd = self._loss_cooldown_mult
                self._loss_cooldown_mult = 1.0  # 次サイクルではリセット

                # 207# §3 / 275# DRY: Toxic veto カウンタ減算 (サイクル末尾)
                self._tick_toxic_veto("cycle_end")

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

        # 265# extract: 最終 cleanup (~35行) を分離
        return await self._finalize_run(st, heartbeat_task)

