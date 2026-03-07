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
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# 332# Phase 4: random, datetime, CR, load_alert_mode → Mixin に移管 (dead import 削除)
from scripts.v460.lib.orchestrator_balance import OrchestratorBalanceMixin
from scripts.v460.lib.orchestrator_guards import OrchestratorGuardsMixin
from scripts.v460.lib.orchestrator_lifecycle import OrchestratorLifecycleMixin
from scripts.v460.lib.orchestrator_mid_cycle import OrchestratorMidCycleMixin
from scripts.v460.lib.orchestrator_post_cycle import OrchestratorPostCycleMixin
from scripts.v460.lib.orchestrator_pre_cycle import OrchestratorPreCycleMixin
# 330# SADLevel は orchestrator_pre_cycle に移管済み (331# dead import 削除)

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
    OrchestratorBalanceMixin,
    OrchestratorGuardsMixin,
    OrchestratorLifecycleMixin,
    OrchestratorMidCycleMixin,
    OrchestratorPostCycleMixin,
    OrchestratorPreCycleMixin,
):
    """run_continuous + side kill / filter / adaptation / cleanup (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: ループ制御 (run_continuous), skip chain 評価
      NG: 1 サイクル実行, OB 取得, SkipGate 評価, PnL 計測
    325# God Object 分割: Guard/Lifecycle/PostCycle を Mixin に抽出
    332# Phase 4: Balance/MidCycle を追加 Mixin に抽出
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
        332# Phase 4: Balance/MidCycle Mixin 抽出 (908行→~60行)。
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
            # ── 330# Phase 1-2: 日替わりリセット + DD halt ──
            self._process_daily_reset()
            if await self._handle_dd_halt(st):
                continue

            # ── 332# Phase 4: Alert mode チェック ──
            if await self._check_alert_mode(st):
                continue

            # ── 330# Phase 3-5: Circuit Breakers + Hard Skip + Phantom ──
            if await self._check_circuit_breakers(st):
                continue
            if await self._check_hard_skip_utc(st):
                continue
            await self._process_phantom_guard()

            # ── 332# Phase 4: CycleContext 初期化 ──
            ctx = self._prepare_cycle_context()

            # ── 330# Phase 6-7: Side Veto + Time Filter ──
            if await self._resolve_side_vetos(st, ctx):
                continue
            if await self._apply_time_filter(st, ctx):
                continue

            # ── 332# Phase 4: Regime fallback + Balance 解決 ──
            self._update_regime_fallback()
            if await self._resolve_balance_and_preflight(st, ctx):
                continue

            # ── 332# Phase 4: Mid-cycle 判定 ──
            if await self._handle_one_sided_skip(st, ctx):
                continue
            if await self._handle_balance_forced_skip(st, ctx):
                continue
            if await self._handle_forced_buy_delay(st, ctx):
                continue

            # ── 332# Phase 4: Gate 評価 ──
            gate_result = await self._evaluate_and_handle_cycle_gate(st, ctx)
            if gate_result is None:
                continue  # blocked
            if await self._handle_toxicity_skip(st, ctx, gate_result):
                continue
            if await self._handle_degraded_liquidation(st, ctx, gate_result):
                continue

            # ── 332# Phase 4: Cycle 実行 ──
            await self._execute_and_track_cycle(st, ctx, gate_result)

            # ── 332# Phase 4: Post-cycle sleep ──
            if time.time() < end_time and not self._kill_switch.is_killed():
                await self._post_cycle_sleep(ctx)

        # 265# extract: 最終 cleanup (~35行) を分離
        return await self._finalize_run(st, heartbeat_task)
