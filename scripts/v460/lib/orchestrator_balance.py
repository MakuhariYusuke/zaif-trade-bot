"""332# orchestrator_balance — Balance / Preflight 解決 Mixin.

328# God Object Split Phase 4:
run_continuous (~800 行) のうち ~230 行の balance preflight ロジックを
OrchestratorBalanceMixin に分離。

責務:
  - 残高 pre-flight check (current side のみ — 反対 side 強制なし)
  - Preflight failure handling (balance_shrink, pause, safe_stop)
  - 372# dust buy-to-clear: micro-dust 検出時の buy 準備

522# balance_forced 完全撤廃:
  348# で balance_forced 概念を全廃したが、balance_switch / recovery_skew /
  inventory_escape が実質的に同じ機能を果たしていた。522# でこれらを
  全て撤廃。残高不足時はサイクルをスキップし、side_selector の freeze を
  通じて次サイクルで自然に反対 side が選択されるようにする。
  No Trade = Normal (250# 原則)。
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
        """522# 残高 pre-flight check (balance-forcing 完全撤廃).

        332# extract from run_continuous L435-L636.
        True = サイクルスキップ (caller: continue)
        False = preflight 成功、実行パスに進む

        522# 変更: balance_switch / recovery_skew / inventory_escape を全廃。
        残高不足時は side を freeze して次サイクルで自然に反対 side が選ばれるようにする。
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

        # ── 522# 残高不足: 反対 side の状態を確認 (安全弁) ──
        opposite = self._opposite_side(next_side)

        if await self._check_balance_for_side(opposite, regime_mult=_regime_mult):
            # 両 side とも残高不足 → preflight failure handling
            return await self._handle_preflight_failure(st, ctx)

        # ── 522# 片側のみ残高不足 → side を freeze してスキップ ──
        # balance_switch/recovery_skew/inventory_escape は全廃。
        # freeze により次サイクルで自然に opposite が選択される。
        logger.info(
            f"[522#] {next_side} insufficient, {opposite} available — "
            f"freezing {next_side} and skipping (no forced switching)"
        )
        self._side_selector.freeze_side(
            next_side, cycles=self.config.balance_freeze_cycles,
        )
        self._inc_guard_fire("balance_insufficient_skip")

        # 372# dust buy-to-clear: micro-dust 検出時は buy ロット準備
        if self._balance_checker.dust_buy_pending:
            self._balance_checker.prepare_dust_buy()

        await self._execute_skip(
            st, side=next_side,
            cancel_reason=CR.PREFLIGHT_INSUFFICIENT,
            order_quantity=self._current_lot,
            flush_context="balance_insufficient",
            state_save=True,
            state_save_context="balance_insufficient",
            update_last_side=True,
        )
        return True

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

        # 604# preflight 連続失敗毎に残高コンテキストを出力
        _bc = self._balance_checker
        logger.warning(
            "[preflight_skip] count=%d/%d "
            "btc_free=%s, btc_locked=%s, jpy_free=%s, jpy_locked=%s",
            self._preflight_skip_count,
            self.config.max_preflight_skip,
            f"{_bc.last_btc_free:.8f}" if _bc.last_btc_free is not None else "?",
            f"{_bc.last_btc_locked:.8f}" if _bc.last_btc_locked is not None else "?",
            f"{_bc.last_jpy_free:.2f}" if _bc.last_jpy_free is not None else "?",
            f"{_bc.last_jpy_locked:.2f}" if _bc.last_jpy_locked is not None else "?",
        )

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
                f"ロット縮小: {old_lot:.8f} → {self._current_lot:.8f} BTC "
                f"(min_lot={min_lot:.8f})"
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
            # 459# preflight_pause 中も config hot-reload を検出
            self._config_reloader.maybe_reload(self)
            await asyncio.sleep(pause_sec)
            return True

        # 602# open order recovery: SAFE_STOP 直前に滞留注文をキャンセル
        # btc_reserved が open order に拘束され sell 不可 → 両側膠着のパターン
        if self._preflight_skip_count >= self.config.max_preflight_skip:
            try:
                open_orders = await self.adapter.get_open_orders(
                    self.config.symbol,
                )
                if open_orders:
                    cancelled = 0
                    for order in open_orders:
                        try:
                            await self.adapter.cancel_order(order.order_id)
                            cancelled += 1
                            logger.warning(
                                f"[602# preflight_recovery] Cancelled stale "
                                f"order: id={order.order_id}, side={order.side}"
                                f", price={order.price}, qty={order.quantity}"
                            )
                        except Exception as e:
                            logger.error(
                                f"[602# preflight_recovery] Failed to cancel "
                                f"order {order.order_id}: {e}",
                                exc_info=True,
                            )
                    if cancelled > 0:
                        logger.warning(
                            f"[602# preflight_recovery] Cancelled "
                            f"{cancelled}/{len(open_orders)} stale orders. "
                            f"Resetting preflight counter to retry."
                        )
                        self._preflight_skip_count = 0
                        await self._effective_sleep()
                        return True
            except Exception as e:
                logger.error(
                    f"[602# preflight_recovery] Open order check failed: {e}",
                    exc_info=True,
                )

            # 044# F8: 連続 preflight 失敗上限 → SAFE_STOP
            logger.error(
                f"SAFE_STOP: 連続 preflight スキップ {self._preflight_skip_count} 回 "
                f"(上限 {self.config.max_preflight_skip}). "
                f"buy/sell 両方で残高不足の可能性. 停止します."
            )
            self._kill_switch.kill("preflight_skip_exceeded")
            return True  # kill_switch で while ループが終了する

        await self._effective_sleep()
        return True
