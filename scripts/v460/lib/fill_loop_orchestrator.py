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
from scripts.v460.lib.spread_anomaly_detector import SADLevel

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig
    from scripts.v460.lib.phantom_position_guard import PhantomPositionGuard
    from ztb.metrics.fill_quality import FillRecord
    from ztb.risk.sell_dynamic_kill import DynamicKillManager, ToxicityAssessment

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
    batch: list[FillRecord] = field(default_factory=list)
    batch_size: int = 10


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
    # 250# P1-4: freeze/cooldown が紐付いた side
    # — None 時は全 side スキップ (後方互換), side 指定時はその side のみ
    _one_sided_frozen_side: str | None = None
    # 223# skip-time state save: 最終 state save のモノトニック時刻
    _last_state_save_time: float = 0.0
    #: 223# skip パス中の state save 間隔 (秒)
    _STATE_SAVE_INTERVAL_SEC: float = 300.0
    # 228# H3: MCB/SAD/CycleStrategy class-level None defaults (hasattr 排除用)
    _mcb: object | None = None
    _sad: object | None = None
    _cycle_strategy: object | None = None
    # 237# PhantomPositionGuard class-level default (hasattr 排除)
    # 238# C-1: object → PhantomPositionGuard 型安全化 (TYPE_CHECKING)
    _phantom_guard: PhantomPositionGuard | None = None
    # 254# _recent_records: _check_stop_conditions が参照。テスト注入用。
    # 256# list → deque(maxlen=200): batch save でリセットされない累積バッファ
    _recent_records: deque[FillRecord] = deque(maxlen=200)  # type: ignore[assignment]
    # 254# _heartbeat_task: run_continuous 内で代入、cleanup_heartbeat で参照
    _heartbeat_task: asyncio.Task[None] | None = None

    def _is_sell_killed(self) -> bool:
        """133# P0-10 / 136# P1-03: sell 動的 kill 判定 — SellDynamicKillManager に委譲.

        §9 #3: 現在レジームを check_kill() に渡し regime_thresholds を有効化。
        """
        regime: str | None = None
        if self._regime_detector is not None:
            regime = self._regime_detector.current_regime.value
        killed, telemetry = self._sell_kill_mgr.check_kill(regime=regime)
        # 223# probe/force_release メトリクス
        if telemetry.probe_fired:
            self._inc_guard_fire("dynamic_kill_probe_sell")
        if telemetry.force_release_fired:
            self._inc_guard_fire("dynamic_kill_force_release_sell")
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
        # 223# probe/force_release メトリクス
        if telemetry.probe_fired:
            self._inc_guard_fire("dynamic_kill_probe_buy")
        if telemetry.force_release_fired:
            self._inc_guard_fire("dynamic_kill_force_release_buy")
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
    # 240# Toxicity Budget — assess_toxicity (副作用なし)
    # 241# C-3/S-1/S-2 fix: DRY 統一 + 型安全化 + getattr 除去
    # ------------------------------------------------------------------
    def _assess_toxicity(
        self, mgr: "DynamicKillManager",
    ) -> "ToxicityAssessment | None":
        """240# toxicity budget 評価 (副作用なし).

        Args:
            mgr: buy or sell の DynamicKillManager

        Returns:
            ToxicityAssessment or None (budget 無効時)
        """
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

    def _warmup_daily_drawdown_from_records(
        self, records: list["FillRecord"],
    ) -> None:
        """203# F: fill records から当日分の PnL を DD guard に投入.

        state file が stale/missing の場合のセーフティネット。
        import_state が skip された後に呼ばれ、DD guard を正確な状態に復元する。
        """
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
                guard.state.halt_triggered_at = time.time()
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

        225# F1: 当日分のみ replay — B2 日替わり kill reset との矛盾を防止。
        前日以前のデータを注入すると日替わり reset() の効果が無効化されるため、
        DD warmup と同じ日付フィルタを適用する。
        """
        utc_today = datetime.now(timezone.utc).strftime("%Y%m%d")
        sell_count = 0
        buy_count = 0
        skipped_old = 0
        for r in records:
            if not r.filled or r.post_fill_30s_pnl is None:
                continue
            # 225# F1: 当日分のみ — 前日以前は skip
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
            # 224#: soft drawdown interval 乗数永続化
            soft_drawdown_interval_multiplier=self._soft_drawdown_interval_multiplier,
            # 216# E: Guard 発火カウンタ永続化
            guard_fire_counts=dict(self._guard_fire_counts) if self._guard_fire_counts else None,
            # 244# Guard reason category totals
            guard_category_totals=self._guard_category_totals(),
            # 209# H4: DynamicKillManager 状態永続化
            sell_kill_state=self._sell_kill_mgr.export_state(),
            buy_kill_state=self._buy_kill_mgr.export_state(),
            # 225# MCB/SAD 状態永続化 (228# H3: hasattr → class-level None default)
            mcb_state=(
                self._mcb.export_state() if self._mcb is not None else None
            ),
            sad_state=(
                self._sad.export_state() if self._sad is not None else None
            ),
            # 236# エスカレーション・縮退カウンタ永続化
            degraded_liquidation_duty_counter=self._degraded_liquidation_duty_counter,
            # 269# Inventory Escape Mode カウンタ永続化
            inventory_escape_duty_counter=self._inventory_escape_duty_counter,
            one_sided_cooldown_remaining=self._one_sided_cooldown_remaining,
            one_sided_freeze_remaining=self._one_sided_freeze_remaining,
            # 254# 250# P1-4 永続化漏れ修正: freeze/cooldown の対象 side
            one_sided_frozen_side=self._one_sided_frozen_side,
            consecutive_no_feasible=(
                dict(self._consecutive_no_feasible)
                if self._consecutive_no_feasible
                else None
            ),
            # 237# phantom position guard メトリクス
            phantom_guard_metrics=(
                self._phantom_guard.get_metrics()
                if self._phantom_guard is not None
                else None
            ),
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

    def _guard_category_totals(self) -> dict[str, int] | None:
        """244# guard_fire_counts のカテゴリ別合計を返す."""
        if not self._guard_fire_counts:
            return None
        from scripts.v460.lib.guard_reason_classifier import guard_category_totals
        return guard_category_totals(self._guard_fire_counts)

    # ------------------------------------------------------------------
    # 272# DRY: toxic_veto 減算ヘルパー (3 箇所の共通ロジック抽出)
    # ------------------------------------------------------------------
    def _tick_toxic_veto(self, context: str) -> None:
        """全 toxic_veto カウンタを 1 減算し、期限切れを除去する.

        L1518 (both-blocked), L1741 (inventory_escape), L1763 (halt_block) の
        3 箇所で同一ロジックが重複していたものを一元化。
        """
        if not self._toxic_veto:
            return
        for _vs in list(self._toxic_veto.keys()):
            self._toxic_veto[_vs] -= 1
            if self._toxic_veto[_vs] <= 0:
                del self._toxic_veto[_vs]
                logger.info(f"[226# S2] Toxic veto expired ({context}): {_vs}")

    # ------------------------------------------------------------------
    # 272# DRY: skip-path state save ヘルパー (3 箇所の共通ロジック抽出)
    # ------------------------------------------------------------------
    def _maybe_skip_state_save(
        self,
        st: "RunSessionState",
        context: str,
    ) -> None:
        """_STATE_SAVE_INTERVAL_SEC 経過時のみ state 保存する.

        gate_block (L2156), halt_block (L1782), 等の skip パスで
        同一ロジックが重複していたものを一元化。
        """
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
    # 272# DRY: MCB/SAD フィードヘルパー (halt loop / 将来の skip loop 共通)
    # ------------------------------------------------------------------
    def _feed_mcb_sad(self) -> None:
        """MCB/SAD に最新の mid_price / spread をフィードする.

        halt 中など check() を呼ばずにモデル更新だけを行うパスで使用。
        halt 解除直後に陳腐化した σ で誤判定するのを防止する。
        """
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
        # 224#: soft drawdown interval 乗数復元
        # 254# getattr → 直接参照 (FillTestState にフィールド存在)
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
        # 225# MCB/SAD 状態復元 (228# H3: hasattr → class-level None default)
        # 254# getattr → 直接参照 (FillTestState にフィールド存在)
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
        # 254# getattr → 直接参照 (FillTestState にフィールド存在)
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
        # 254# getattr → 直接参照
        _fs = saved_state.one_sided_frozen_side
        if _fs is not None:
            self._one_sided_frozen_side = _fs
            logger.info(f"[254#] One-sided frozen side restored: {_fs}")
        _cnf = saved_state.consecutive_no_feasible
        if _cnf:
            self._consecutive_no_feasible = dict(_cnf)
            logger.info(f"[236#] Consecutive no-feasible restored: {_cnf}")

    # ------------------------------------------------------------------
    # 179# S1: _effective_sleep — regime 応答サイクル間隔の一元化
    # ------------------------------------------------------------------
    async def _effective_sleep(
        self, *, multiplier: float = 1.0, max_override: float = 0.0,
    ) -> None:
        """179# CycleStrategy に委譲し、regime 別サイクル間隔で sleep.

        skip/halt/error continue 全パスがこのメソッドを経由する。
        - multiplier=1.0 : 通常スキップ
        - multiplier=5.0 : halt (daily drawdown)
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
        # 256# deque 化: スライス不可のため list comprehension → スライス
        pnls = [
            r.post_fill_30s_pnl for r in self._recent_records
            if r.filled and r.post_fill_30s_pnl is not None
        ][-100:]
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

    # ------------------------------------------------------------------
    # 265# extract: run_continuous 初期化 → _init_run_session
    # ------------------------------------------------------------------
    async def _init_run_session(self) -> RunSessionState:
        """run_continuous の初期化フェーズ.

        265# extract method: lock 取得, trades health check, レジューム復元,
        state/regime/DD warmup, PnL 累積計算。
        ~200 行の初期化ロジックを run_continuous から分離。

        Returns:
            RunSessionState: ループ間共有状態 (total/filled count, PnL 累積等).
        """
        from scripts.v460.lib.event_logger import log_event as _log_event
        from ztb.data.trades_health import check_trades_health
        from ztb.metrics.fill_quality import (
            compute_record_pnl_jpy,
            filter_clean_records,
        )

        # 044# 単一起動ロック取得
        self._acquire_lock()

        # 135# P2-09→P1: trades データ健全性チェック
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
    # 265# extract: post-cycle 処理 → _process_post_cycle
    # ------------------------------------------------------------------
    def _process_post_cycle(
        self,
        record: FillRecord,
        next_side: str,
        st: RunSessionState,
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
            self._track_sell_pnl(record)
            self._track_buy_pnl(record)
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
            # 168# §4.1: daily drawdown PnL update
            if record.post_fill_30s_pnl is not None:
                dd_result = self._daily_drawdown_guard.update_pnl(
                    record.post_fill_30s_pnl,
                    side=next_side,
                )
                if dd_result.get("soft_triggered"):
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
    # 265# extract: progress log + state save + adaptation → _finalize_cycle
    # ------------------------------------------------------------------
    async def _log_progress_and_adapt(
        self,
        next_side: str,
        st: RunSessionState,
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

        # 181# 停止条件モニター
        if (
            self._cycle_strategy is not None
            and self._cycle_count > 0
            and self._cycle_count % 30 == 0
        ):
            self._check_regime_stop_conditions(st.filled_count, st.total_count)

    # ------------------------------------------------------------------
    # 265# extract: final cleanup → _finalize_run
    # ------------------------------------------------------------------
    async def _finalize_run(
        self,
        st: RunSessionState,
        heartbeat_task: asyncio.Task[None],
    ) -> list[FillRecord]:
        """run_continuous の最終クリーンアップ.

        265# extract method: 残バッチ保存, 最終 state 保存, heartbeat 停止。

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
            # 200# 10-A: 日替わり時に soft_drawdown_interval_multiplier をリセット
            # P0-2 で追加した multiplier が日次境界で reset されないバグの修正
            if self._daily_drawdown_guard.maybe_reset_day():
                _old_mult = self._soft_drawdown_interval_multiplier
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

            # 168# §4.1 #3: 日次ドローダウンガード — halt 中はスキップ
            if self._daily_drawdown_guard.is_halted():
                # 日次 PnL 超過 → UTC 日替わりまでスキップ
                # 200# K: halt record 削減 — 開始/終了 + N回毎のみ記録
                # 203# G: _halt_iter_count で正確にカウント (_cycle_count は halt 中不変)
                _halt_entering = self._halt_start_cycle is None
                if _halt_entering:
                    self._inc_guard_fire("dd_halt")
                    self._halt_start_cycle = self._cycle_count
                    self._halt_iter_count = 0
                else:
                    self._halt_iter_count = self._halt_iter_count + 1
                # 211# fix: halt 中は progress_log_interval(50) ではなく
                # 専用間隔 _HALT_PERSIST_INTERVAL(10) で state/record を保存。
                # 600s sleep × 50 = 8.3h は長すぎ、再起動時に進捗が大幅に巻き戻る。
                _HALT_PERSIST_INTERVAL = 10
                _should_record_halt = (
                    _halt_entering  # 開始時
                    or self._halt_iter_count % _HALT_PERSIST_INTERVAL == 0
                )
                if _should_record_halt:
                    st.batch.append(self._make_loop_skip_record(
                        side="none",
                        cancel_reason=CR.DAILY_DRAWDOWN_HALT,
                        order_quantity=0.0,
                    ))
                    st.total_count += 1
                    st.batch = self._batch_persistence.maybe_flush(st.batch, "daily_drawdown_halt")
                self._update_lock_heartbeat()
                # 203# E: HALT 開始時は必ず state 保存 + 以降は N iter 毎
                # 211# fix: halt 専用間隔に統一 (旧: progress_log_interval=50 → 8.3h)
                if _should_record_halt:
                    self._state_persistence.save(self._build_state_snapshot(
                        total_count=st.total_count,
                        filled_count=st.filled_count,
                        cumulative_pnl_jpy=st.cumulative_pnl_jpy,
                    ))
                    self._last_state_save_time = time.monotonic()  # 223#
                # 211#: halt サイクル可視化ログ (entering + _HALT_PERSIST_INTERVAL 毎)
                if _should_record_halt:
                    logger.info(
                        f"[daily_drawdown] Halt cycle #{self._halt_iter_count}"
                        f" (next log @+{_HALT_PERSIST_INTERVAL} iters)"
                    )
                # 226# S5: halt 中も MCB/SAD に price/spread をフィードし続ける。
                # halt 解除直後に陳腐化した σ で誤判定するのを防止。
                # ※ check() は呼ばない (halt 中の二重ガードは不要)。
                self._feed_mcb_sad()
                await self._effective_sleep(multiplier=5.0)  # 179# S1: halt 中は 5x 間隔
                continue

            # 200# K: halt 終了時の記録 (前サイクルが halt だった場合)
            if self._halt_start_cycle is not None:
                _halt_iters = self._halt_iter_count
                logger.info(
                    f"[daily_drawdown] Halt ended after {_halt_iters} iterations"
                )
                self._halt_start_cycle = None
                self._halt_iter_count = 0

            # 215# P0-C: alert_mode.json — オペレータ緊急介入チェック
            _alert = load_alert_mode(self._results_dir)
            if _alert.halt:
                st.batch.append(self._make_loop_skip_record(
                    side="none",
                    cancel_reason=CR.OPERATOR_HALT,
                    order_quantity=0.0,
                ))
                st.total_count += 1
                st.batch = self._batch_persistence.maybe_flush(st.batch, "operator_halt")
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
            if self._mcb is not None and self._mcb.config.enabled:
                _mcb_mid = self._maker_price.last_mid_price
                if _mcb_mid is not None and _mcb_mid > 0:
                    self._mcb.update(_mcb_mid, time.time())
                _mcb_result = self._mcb.check(time.time())
                if _mcb_result.level == MCBLevel.HALT:
                    self._inc_guard_fire("mcb_halt")
                    st.batch.append(self._make_loop_skip_record(
                        side="none",
                        cancel_reason=CR.MCB_HALT,
                        order_quantity=0.0,
                    ))
                    st.total_count += 1
                    st.batch = self._batch_persistence.maybe_flush(st.batch, "mcb_halt")
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
            if self._sad is not None and self._sad.config.enabled:
                # 217# fix: last_spread は 60s staleness guard 付き (210# M5)。
                # cycle 間隔 120s では常に stale → None になるため、
                # staleness guard なしの last_spread_raw を使用する。
                _sad_spread = self._maker_price.last_spread_raw
                if _sad_spread is not None and _sad_spread > 0:
                    self._sad.update(_sad_spread, time.time())
                _sad_result = self._sad.check(time.time())
                if _sad_result.level == SADLevel.FROZEN:
                    self._inc_guard_fire("sad_frozen")
                    st.batch.append(self._make_loop_skip_record(
                        side="none",
                        cancel_reason=CR.SAD_FROZEN,
                        order_quantity=0.0,
                    ))
                    st.total_count += 1
                    st.batch = self._batch_persistence.maybe_flush(st.batch, "sad_frozen")
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
                st.batch.append(self._make_loop_skip_record(
                    side="none",
                    cancel_reason=CR.MCB_SAD_ESCALATION,
                    order_quantity=0.0,
                ))
                st.total_count += 1
                st.batch = self._batch_persistence.maybe_flush(
                    st.batch, "mcb_sad_escalation"
                )
                self._update_lock_heartbeat()
                await self._effective_sleep(multiplier=5.0)
                continue

            # 205# §9.4: 時間帯 Hard Skip (Kyle proxy)
            # soft offset (158# P1-6) では抑制不十分な最悪時間帯はサイクル全停止
            if self.config.hard_skip_utc_hours:
                _utc_h = datetime.now(timezone.utc).hour
                if _utc_h in self.config.hard_skip_utc_hours:
                    # 初回のみ skip record を記録
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
                        st.batch = self._batch_persistence.maybe_flush(st.batch, "hard_skip_utc_hour")
                        logger.info(
                            f"[205# §9.4] Hard skip: UTC {_utc_h}h is in "
                            f"hard_skip_utc_hours={self.config.hard_skip_utc_hours}"
                        )
                    self._update_lock_heartbeat()
                    await self._effective_sleep()
                    continue
                else:
                    if self._in_hard_skip_hour:
                        logger.info(f"[205# §9.4] Hard skip ended (UTC {_utc_h}h)")
                        self._in_hard_skip_hour = False

            # ────────────────────────────────────────────────────
            # 237# Phantom Position Guard: 前サイクルの status_unknown を遅延再照合
            # 238# S-2: side veto カウンタ tick + reconcile
            # ────────────────────────────────────────────────────
            if self._phantom_guard is not None:
                self._phantom_guard.tick_veto()  # 238# S-2: veto デクリメント
                if self._phantom_guard.has_pending:
                    try:
                        _phantom_detections = await self._phantom_guard.reconcile(self.adapter)
                        if _phantom_detections:
                            self._inc_guard_fire("phantom_position_detected")
                            for _pd in _phantom_detections:
                                logger.critical(
                                    f"[237# PHANTOM] Inventory mismatch: "
                                    f"{_pd.side} {_pd.quantity:.6f} BTC @ {_pd.price:.0f} "
                                    f"(method={_pd.detection_method}) — "
                                    f"cautious mode activated, side veto={self._phantom_guard._PHANTOM_VETO_CYCLES} cycles"
                                )
                            # ファントム検出時: 安全側バイアス — interval 延長
                            await self._effective_sleep(multiplier=3.0)
                    except Exception as _pg_err:
                        logger.warning(f"[237# phantom_guard] Reconcile error: {_pg_err}")

            # 205# §9.5: 片側 DD Halt のサイクルカウンタ更新
            self._daily_drawdown_guard.tick_side_halt()

            # 205# §9.2: Toxic Fill 同一サイド拒否 — 初期化のみ (デクリメントはサイクル末尾)
            if self._toxic_veto is None:
                self._toxic_veto = {}

            # 129# D.2: 残高制約による side 強制切替追跡
            _balance_forced = False
            _is_rescue = False  # 158# P1-1: balance_forced rescue フラグ
            _one_sided_balance = False  # 190# B: 片側 balance フラグ (ev_weighted threshold 緩和用)
            _inventory_escape = False  # 269# P0: Inventory Escape Mode
            # 073# side 別時間帯フィルター: side 決定後にフィルタリング
            # side 別リスト未設定時はグローバルリスト (041# 互換)
            next_side = self._next_side()

            # 205# §9.5: 片側 DD Halt チェック — 封鎖されたサイドは回避
            if self._daily_drawdown_guard.is_side_halted(next_side):
                _alt = self._opposite_side(next_side)
                if self._daily_drawdown_guard.is_side_halted(_alt):
                    # 両サイド封鎖 → 集約 halt と同等扱い
                    self._inc_guard_fire("per_side_dd_both_halt")  # 223#
                    # 273# I3: 両サイド封鎖の空サイクルも halt カウントから除外
                    self._daily_drawdown_guard.untick_side_halt()
                    st.batch.append(self._make_loop_skip_record(
                        side="none",
                        cancel_reason=CR.PER_SIDE_DD_HALT,
                        order_quantity=0.0,
                    ))
                    st.total_count += 1
                    st.batch = self._batch_persistence.maybe_flush(st.batch, "per_side_dd_both_halt")
                    self._update_lock_heartbeat()
                    await self._effective_sleep(multiplier=5.0)
                    continue
                else:
                    # 223# P0: per-side halt switch を guard_fire_counts に記録
                    self._inc_guard_fire("per_side_halt_switch")
                    logger.info(
                        f"[205# §9.5] Per-side DD halt: {next_side} blocked, "
                        f"switching to {_alt}"
                    )
                    next_side = _alt

            # 205# §9.2: Toxic Fill 同一サイド拒否 — 封鎖されたサイドは反対に切替
            # 207# §5b: alt_side が per_side_dd で封鎖されている場合も考慮
            if self._toxic_veto and next_side in self._toxic_veto:
                _alt = self._opposite_side(next_side)
                _alt_blocked = (
                    _alt in self._toxic_veto
                    or self._daily_drawdown_guard.is_side_halted(_alt)
                )
                if _alt_blocked:
                    # 両サイド封鎖 (veto + per_side_dd 含む) → スキップ
                    # 209# H-1: デッドロック防止 — skip 時も veto カウンタを減算
                    self._inc_guard_fire("toxic_veto_block")
                    self._tick_toxic_veto("both-blocked")
                    st.batch.append(self._make_loop_skip_record(
                        side="none",
                        cancel_reason=CR.TOXIC_FILL_SIDE_VETO,
                        order_quantity=0.0,
                    ))
                    st.total_count += 1
                    st.batch = self._batch_persistence.maybe_flush(st.batch, "toxic_veto_both")
                    await self._effective_sleep()
                    continue
                else:
                    logger.info(
                        f"[205# §9.2] Toxic veto: {next_side} blocked "
                        f"(remaining={self._toxic_veto[next_side]}), "
                        f"switching to {_alt}"
                    )
                    next_side = _alt

            # 238# S-2: Phantom side veto — phantom 検出後の同 side 一時拒否
            if (
                self._phantom_guard is not None
                and self._phantom_guard.is_side_vetoed(next_side)
            ):
                _alt = self._opposite_side(next_side)
                if (
                    self._phantom_guard.is_side_vetoed(_alt)
                    or self._daily_drawdown_guard.is_side_halted(_alt)
                ):
                    # 両サイド封鎖 → スキップ
                    self._inc_guard_fire("phantom_veto_block")
                    st.batch.append(self._make_loop_skip_record(
                        side="none",
                        cancel_reason=CR.PHANTOM_SIDE_VETO,
                        order_quantity=0.0,
                    ))
                    st.total_count += 1
                    st.batch = self._batch_persistence.maybe_flush(st.batch, "phantom_veto_both")
                    await self._effective_sleep()
                    continue
                else:
                    logger.info(
                        f"[238# phantom_veto] {next_side} blocked → "
                        f"switching to {_alt}"
                    )
                    next_side = _alt

            # side 別チェック (073#): side固有リストがあれば side 別判定
            side_filtered = self._is_time_filtered(side=next_side)
            if side_filtered:
                # 反対 side でもフィルタされるか確認
                alt_side = self._opposite_side(next_side)
                alt_filtered = self._is_time_filtered(side=alt_side)
                if alt_filtered:
                    # 両 side ともフィルタ → スリープ
                    # 225# 5.1: fire count 記録
                    self._inc_guard_fire("time_filter_both_sides")
                    # 140# §8.1-#2: skip record を生成し可観測性確保 (132# F4)
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
                        if now_ts - self._time_filter.last_heartbeat_time >= self.config.heartbeat_interval_sec:
                            utc_h = datetime.now(timezone.utc).hour
                            try:
                                import psutil  # lazy import
                                proc = psutil.Process()
                                mem_mb = proc.memory_info().rss / (1024 * 1024)
                                mem_info = f"mem={mem_mb:.1f}MB, "
                            except Exception:
                                # 254# bare except → debug log (psutil 不在/権限エラー可観測化)
                                logger.debug("psutil memory check unavailable", exc_info=True)
                                mem_info = ""
                            logger.info(
                                f"[heartbeat] Still in time_filter zone "
                                f"(UTC {utc_h}h), "
                                f"{mem_info}"
                                f"unsaved_batch={len(st.batch)}, "
                                f"cycles={self._cycle_count}"
                            )
                            self._time_filter.last_heartbeat_time = now_ts
                            # 129# lock heartbeat 更新
                            self._update_lock_heartbeat()
                        # 107# R1: 重複 flush → _maybe_flush_batch 統合
                        st.batch = self._batch_persistence.maybe_flush(st.batch, "time_filter")
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
                                st.batch.append(self._make_loop_skip_record(
                                    side=next_side,
                                    cancel_reason=CR.TIME_FILTER_086_DEADLOCK,
                                    order_quantity=0.0,
                                ))
                            # 107# R1: 重複 flush → _maybe_flush_batch 統合
                            st.batch = self._batch_persistence.maybe_flush(st.batch, "alt_side==last_side wait")
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
                        if _ie_enabled and next_side == "sell":
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
                            # 273# I3: 空サイクルの halt カウント除外
                            # この continue パスでは実質的な取引試行がないため
                            # tick_side_halt() のデクリメントを補償する
                            self._daily_drawdown_guard.untick_side_halt()
                            # 226# S2: balance_forced + halt_block で continue する際、
                            # toxic_veto のカウンタも減算する。
                            self._tick_toxic_veto("halt_block")
                            st.batch.append(self._make_loop_skip_record(
                                side=next_side,
                                cancel_reason=CR.PER_SIDE_DD_HALT,
                                order_quantity=self._current_lot,
                                balance_forced_switch=True,
                            ))
                            st.total_count += 1
                            st.batch = self._batch_persistence.maybe_flush(
                                st.batch, "balance_forced_halt_recheck",
                            )
                            # 269# P0-b: state save — halt_block 長期化に対する stale 防止
                            self._maybe_skip_state_save(st, "balance_forced_halt_block")
                            self._last_side = next_side
                            await self._effective_sleep()
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
                    _skip_record = self._make_loop_skip_record(
                        side=next_side,
                        cancel_reason="one_sided_freeze_skip",
                        order_quantity=self._current_lot,
                    )
                    st.batch.append(_skip_record)
                    st.total_count += 1
                    st.batch = self._batch_persistence.maybe_flush(st.batch, "one_sided_freeze_skip")
                    self._last_side = next_side
                    await self._effective_sleep()
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
                    _skip_record = self._make_loop_skip_record(
                        side=next_side,
                        cancel_reason="one_sided_cooldown_skip",
                        order_quantity=self._current_lot,
                    )
                    st.batch.append(_skip_record)
                    st.total_count += 1
                    st.batch = self._batch_persistence.maybe_flush(st.batch, "one_sided_cooldown_skip")
                    self._last_side = next_side
                    await self._effective_sleep()
                    continue
                logger.debug(
                    f"[250#] Cooldown side={_frozen_side}, current={next_side} — pass through"
                )

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
                    _skip_record = self._make_loop_skip_record(
                        side=next_side,
                        cancel_reason=CR.BALANCE_FORCED_SKIP,
                        order_quantity=self._current_lot,
                        balance_forced_switch=True,
                        balance_forced_consecutive=self._balance_forced_skip_count,
                    )
                    st.batch.append(_skip_record)
                    st.total_count += 1
                    st.batch = self._batch_persistence.maybe_flush(st.batch, "balance_forced_skip")
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

                _skip_record = self._make_loop_skip_record(
                    side=next_side,
                    cancel_reason=_gate_result.cancel_reason,
                    order_quantity=self._current_lot,
                )
                st.batch.append(_skip_record)
                st.total_count += 1
                st.batch = self._batch_persistence.maybe_flush(
                    st.batch, _gate_result.cancel_reason,
                )
                self._last_side = next_side

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
                if self._consecutive_gate_blocks >= 10 and self._consecutive_gate_blocks % 10 == 0:
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
                _skip_record = self._make_loop_skip_record(
                    side=next_side,
                    cancel_reason="toxicity_participation_skip",
                    order_quantity=self._current_lot,
                )
                st.batch.append(_skip_record)
                st.total_count += 1
                st.batch = self._batch_persistence.maybe_flush(
                    st.batch, "toxicity_participation_skip",
                )
                self._last_side = next_side
                await self._effective_sleep()
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
                    _skip_record = self._make_loop_skip_record(
                        side=next_side,
                        cancel_reason="degraded_liquidation_duty_skip",
                        order_quantity=self._current_lot,
                    )
                    st.batch.append(_skip_record)
                    st.total_count += 1
                    st.batch = self._batch_persistence.maybe_flush(
                        st.batch, "degraded_liquidation_duty_skip",
                    )
                    self._last_side = next_side
                    await self._effective_sleep()
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
                # 200# P0-2: soft drawdown interval 延長
                soft_dd_mult = self._soft_drawdown_interval_multiplier
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

        # 265# extract: 最終 cleanup (~35行) を分離
        return await self._finalize_run(st, heartbeat_task)

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
