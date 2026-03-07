"""332# orchestrator_balance — Balance / Preflight 解決 Mixin.

328# God Object Split Phase 4:
run_continuous (~800 行) のうち ~230 行の balance preflight ロジックを
OrchestratorBalanceMixin に分離。

責務:
  - 残高 pre-flight check (current side + opposite side 試行)
  - balance_forced スイッチ (129# D.2)
  - Inventory Escape Mode (269#)
  - Preflight failure handling (balance_shrink, pause, safe_stop)
  - Balance forced frequency detection (200# E)

市場理論的位置づけ:
  **Inventory Risk Management** (Stoll 1978, Ho & Stoll 1981):
    残高確認は在庫管理の第一段階。資金不足による片側偏重は
    マーケットメイカーの基本原則に反する。balance_forced は
    在庫中立化のための強制修正操作。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from scripts.v460.lib import cancel_reasons as CR

if TYPE_CHECKING:
    from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
    from scripts.v460.lib.orchestrator_pre_cycle import CycleContext

logger = logging.getLogger(__name__)


class OrchestratorBalanceMixin:
    """332# Balance / Preflight 解決 Mixin.

    run_continuous の残高チェック〜preflight 失敗ハンドリングを分離。
    戻り値 True = 呼び出し元で continue (サイクルスキップ)。
    戻り値 False = 実行パスへ進む (preflight 成功)。
    """

    # ------------------------------------------------------------------
    # 332# extract: balance / preflight resolution
    # ------------------------------------------------------------------
    async def _resolve_balance_and_preflight(
        self, st: RunSessionState, ctx: CycleContext,
    ) -> bool:
        """041# 残高 pre-flight check + opposite side 試行 + preflight 失敗処理.

        332# extract from run_continuous L435-L636.
        True = サイクルスキップ (caller: continue)
        False = preflight 成功、実行パスに進む

        ctx.next_side, ctx.balance_forced, ctx.inventory_escape が更新される。
        """
        next_side = ctx.next_side
        _regime_mult = ctx.regime_mult

        # 041# / 145# §8-#1: レジーム倍率込みで残高判定
        if not await self._check_balance_for_side(next_side, regime_mult=_regime_mult):
            # 残高 OK → preflight 成功
            self._preflight_skip_count = 0
            self._balance_checker.restore_lot_on_success()
            self._side_selector.unfreeze_side()
            return False

        # ── 残高不足: 反対 side を試す ──
        opposite = self._opposite_side(next_side)
        tried_opposite = False

        if not await self._check_balance_for_side(opposite, regime_mult=_regime_mult):
            # 反対 side は残高 OK → 即座に切替
            logger.info(
                f"[balance] {next_side} insufficient, "
                f"switching to {opposite} immediately (091#)"
            )
            self._side_selector.freeze_side(
                next_side, cycles=self.config.balance_freeze_cycles,
            )
            ctx.next_side = opposite
            next_side = opposite
            self._last_side = opposite
            self._preflight_skip_count = 0
            tried_opposite = True
            ctx.balance_forced = True

            # 223# P0: balance_forced 後に per-side halt を再チェック
            if self._daily_drawdown_guard.is_side_halted(next_side):
                skip = await self._handle_inventory_escape_or_halt(
                    st, ctx, next_side,
                )
                if skip:
                    return True
                # skip=False → inventory_escape 成功、実行パスへ fallthrough

            # 200# E: balance_forced 頻度検出
            self._track_balance_forced_frequency()

        if not tried_opposite:
            # 両 side とも残高不足 → preflight failure handling
            return await self._handle_preflight_failure(st, ctx)

        # preflight 成功 (opposite side に切替済み)
        self._preflight_skip_count = 0
        self._balance_checker.restore_lot_on_success()
        self._side_selector.unfreeze_side()
        return False

    # ------------------------------------------------------------------
    # 269# Inventory Escape Mode handling
    # ------------------------------------------------------------------
    async def _handle_inventory_escape_or_halt(
        self,
        st: RunSessionState,
        ctx: CycleContext,
        next_side: str,
    ) -> bool:
        """269# balance_forced + per-side halt → Inventory Escape or halt block.

        True = サイクルスキップ (caller: continue)
        False = inventory_escape 成功、実行パスへ進む
        """
        _ie_enabled = self.config.inventory_escape_enabled
        _ie_duty = max(self.config.inventory_escape_duty_cycle, 1)
        _inventory_escape = False

        if _ie_enabled:
            self._inventory_escape_duty_counter += 1
            if _ie_duty > 1 and (self._inventory_escape_duty_counter % _ie_duty) != 1:
                logger.info(
                    f"[269#] Inventory escape duty skip: "
                    f"cycle {self._inventory_escape_duty_counter}/{_ie_duty}"
                )
                self._inc_guard_fire("inventory_escape_duty_skip")
            else:
                logger.warning(
                    f"[269#] INVENTORY ESCAPE: bypassing per-side halt "
                    f"for {next_side} (balance_forced deadlock breakout, "
                    f"cycle {self._inventory_escape_duty_counter}/{_ie_duty})"
                )
                self._inc_guard_fire("inventory_escape_active")
                _inventory_escape = True
                self._tick_toxic_veto("inventory_escape")

        if _inventory_escape:
            ctx.inventory_escape = True
            return False  # 実行パスへ進む

        # halt 貫通不可 → スキップ
        logger.warning(
            f"[223#] balance_forced → {next_side} is per-side halted — "
            f"refusing to bypass halt (safety > liveness)"
        )
        self._inc_guard_fire("balance_forced_halt_block")
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
        return True

    # ------------------------------------------------------------------
    # 200# E: balance_forced 頻度検出
    # ------------------------------------------------------------------
    def _track_balance_forced_frequency(self) -> None:
        """200# E: 短時間で連続 balance_forced が発生した場合の警告."""
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

    # ------------------------------------------------------------------
    # Preflight failure handling (both sides insufficient)
    # ------------------------------------------------------------------
    async def _handle_preflight_failure(
        self, st: RunSessionState, ctx: CycleContext,
    ) -> bool:
        """両 side とも残高不足時の処理: shrink / pause / safe_stop.

        332# extract from run_continuous.
        常に True を返す (caller: continue) — break 時は kill_switch で停止。
        """
        next_side = ctx.next_side
        self._last_side = next_side
        self._preflight_skip_count += 1
        self._inc_guard_fire("preflight_insufficient")

        st.batch.append(self._make_loop_skip_record(
            side=next_side,
            cancel_reason=CR.PREFLIGHT_INSUFFICIENT,
            order_quantity=self._current_lot,
        ))
        st.batch = self._batch_persistence.maybe_flush(st.batch, "preflight skip")

        # 051# P2-3: Balance auto-shrink
        min_lot = max(self.config.order_quantity, self.config.min_order_btc)
        if (
            self._preflight_skip_count >= self.config.balance_shrink_consecutive
            and not self._balance_checker.balance_shrink_active
            and self._current_lot > min_lot
        ):
            old_lot = self._current_lot
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
            self._preflight_skip_count = 0
            await self._effective_sleep()
            return True

        # 138# P1-10: preflight pause
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
            return True

        # 044# F8: 連続 preflight 失敗上限 → SAFE_STOP
        if self._preflight_skip_count >= self.config.max_preflight_skip:
            logger.error(
                f"SAFE_STOP: 連続 preflight スキップ {self._preflight_skip_count} 回 "
                f"(上限 {self.config.max_preflight_skip}). "
                f"buy/sell 両方で残高不足の可能性. 停止します."
            )
            self._kill_switch.kill("preflight_skip_exceeded")
            return True  # kill_switch で while ループが終了する

        await self._effective_sleep()
        return True
