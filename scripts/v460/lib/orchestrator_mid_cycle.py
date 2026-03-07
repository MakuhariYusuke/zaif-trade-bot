"""332# orchestrator_mid_cycle — Mid-cycle 判定 + 実行ラッパー Mixin.

328# God Object Split Phase 4:
run_continuous (~800 行) のうち mid-cycle 判定ロジック + 実行ラッパーを
OrchestratorMidCycleMixin に分離。

責務:
  - One-sided freeze/cooldown skip (234#)
  - Balance forced skip 判定 (133#/154#)
  - Forced buy delay (286# Glosten-Milgrom)
  - CycleGateAggregator 評価 + ブロック処理 (194#)
  - Toxicity participation skip (240#)
  - Degraded liquidation setup (234#)
  - Cycle 実行 + one-sided tracking + 例外ハンドリング
  - Post-cycle sleep (regime + config reload)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import TYPE_CHECKING

from scripts.v460.lib import cancel_reasons as CR

if TYPE_CHECKING:
    from scripts.v460.lib.cycle_gate_aggregator import CycleGateResult
    from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
    from scripts.v460.lib.orchestrator_pre_cycle import CycleContext

logger = logging.getLogger(__name__)


class OrchestratorMidCycleMixin:
    """332# Mid-cycle 判定 + 実行ラッパー Mixin.

    各メソッドは bool を返し、True = 呼び出し元で continue (サイクルスキップ)。
    _execute_and_track_cycle は void (例外時も continue で処理)。
    """

    # ------------------------------------------------------------------
    # 234# One-sided freeze / cooldown skip
    # ------------------------------------------------------------------
    async def _handle_one_sided_skip(
        self, st: RunSessionState, ctx: CycleContext,
    ) -> bool:
        """234# one-sided エスカレーション: freeze/cooldown 中のスキップ.

        332# extract from run_continuous.
        True = スキップ (caller: continue)
        """
        next_side = ctx.next_side
        _frozen_side = self._one_sided_frozen_side

        # 250# P1-4: freeze は紐付いた side のみスキップ
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
                return True
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
                return True
            logger.debug(
                f"[250#] Cooldown side={_frozen_side}, current={next_side} — pass through"
            )

        return False

    # ------------------------------------------------------------------
    # 133# / 154# Balance forced skip 判定
    # ------------------------------------------------------------------
    async def _handle_balance_forced_skip(
        self, st: RunSessionState, ctx: CycleContext,
    ) -> bool:
        """133# P0-08 / 154# C-1/C-2: balance_forced skip + deadlock 防止.

        332# extract from run_continuous.
        True = スキップ (caller: continue)
        False = 実行パスへ進む (rescue / one_sided_balance / skip 不要)
        ctx.is_rescue, ctx.one_sided_balance が更新される。
        """
        if not ctx.balance_forced or not self.config.skip_balance_forced:
            return False

        next_side = ctx.next_side
        _regime_mult = ctx.regime_mult

        # 154# C-1: 両側残高判定
        original_side = self._opposite_side(next_side)
        original_also_insufficient = await self._check_balance_for_side(
            original_side, regime_mult=_regime_mult,
        )
        # 154# C-2 + 182# regime 別緩和
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
            # 片側のみ or デッドロック上限超過 → 実行許可
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
            ctx.one_sided_balance = original_also_insufficient
            # 202# B: 片側残高枯渇時は rescue offset で保護
            if (
                original_also_insufficient
                and self.config.one_sided_balance_rescue_offset
            ):
                ctx.is_rescue = True
                logger.info(
                    f"[202# B] one_sided_balance rescue: offset ×"
                    f"{self.config.balance_forced_rescue_offset_mult:.1f}"
                )
            return False

        if self.config.balance_forced_rescue_enabled:
            # 158# P1-1: rescue モード
            _prev_skip_count = self._balance_forced_skip_count
            self._balance_forced_skip_count = 0
            ctx.is_rescue = True
            logger.info(
                f"[158# P1-1] balance_forced rescue mode: "
                f"executing {next_side} with offset ×"
                f"{self.config.balance_forced_rescue_offset_mult:.1f} "
                f"(was consecutive skip={_prev_skip_count})"
            )
            return False

        # 両方残高 OK → スキップ (forced switch は損失回避のため)
        self._balance_forced_skip_count += 1
        logger.info(
            f"[133# P0-08] Skipping cycle — balance_forced_switch=True. "
            f"side={next_side}, "
            f"consecutive={self._balance_forced_skip_count}"
        )
        await self._execute_skip(
            st, side=next_side,
            cancel_reason=CR.BALANCE_FORCED_SKIP,
            order_quantity=self._current_lot,
            balance_forced_switch=True,
            balance_forced_consecutive=self._balance_forced_skip_count,
            update_last_side=True,
        )
        return True

    # ------------------------------------------------------------------
    # 286# Forced buy delay (Glosten-Milgrom 1985)
    # ------------------------------------------------------------------
    async def _handle_forced_buy_delay(
        self, st: RunSessionState, ctx: CycleContext,
    ) -> bool:
        """286# 284# P1: 強制買い遅延実行.

        332# extract from run_continuous.
        balance_forced で buy 方向に切り替わった際、microprice が
        急落中なら N サイクル待機。逆選択リスクが高い局面での
        即時買いは損失を拡大するだけ (「待つ勇気」)。
        """
        next_side = ctx.next_side

        # delay 評価: balance_forced + buy + 設定有効時のみ
        if (
            ctx.balance_forced
            and next_side == "buy"
            and self.config.forced_buy_delay_enabled
        ):
            _vel = self._maker_price.last_mid_trend_bps
            _thr = self.config.forced_buy_delay_velocity_threshold_bps
            # 292# P1: ranging/trending_down では緩い閾値
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
                self._forced_buy_delay_remaining = 0
                logger.warning(
                    f"[294# GM deadlock break] Forced buy delay exceeded "
                    f"max_consecutive={_max_consec}, forcing through. "
                    f"velocity={_vel}bps, regime="
                    f"{getattr(getattr(self._regime_detector, 'current_regime', None), 'value', 'N/A')}"
                )

        # delay 消化
        if self._forced_buy_delay_remaining > 0 and next_side == "buy":
            self._forced_buy_delay_remaining -= 1
            self._forced_buy_delay_consecutive += 1
            self._inc_guard_fire("forced_buy_delay")
            await self._execute_skip(
                st, side=next_side,
                cancel_reason=CR.FORCED_BUY_DELAY,
                order_quantity=self._current_lot,
                balance_forced_switch=ctx.balance_forced,
                update_last_side=True,
            )
            return True

        # delay 通過 → 連続カウンタリセット
        self._forced_buy_delay_consecutive = 0
        return False

    # ------------------------------------------------------------------
    # 194# CycleGateAggregator 評価 + ブロック処理
    # ------------------------------------------------------------------
    async def _evaluate_and_handle_cycle_gate(
        self, st: RunSessionState, ctx: CycleContext,
    ) -> CycleGateResult | None:
        """194# CycleGateAggregator 評価.

        332# extract from run_continuous.
        ブロック時は内部で skip 処理して None を返す。
        通過時は CycleGateResult を返す。
        """
        next_side = ctx.next_side

        # HF4 安全弁: trending_sell のための buy 残高チェック
        _buy_side_insufficient = False
        if (
            self.config.skip_sell_trending
            and next_side == "sell"
            and not ctx.balance_forced
            and self._regime_detector is not None
            and self._regime_detector.current_regime.is_trending
        ):
            _buy_side_insufficient = await self._check_balance_for_side(
                "buy", regime_mult=ctx.regime_mult,
            )

        # 241# C-2: toxicity 評価を check_kill() の前に実行
        _buy_tox = self._assess_buy_toxicity()
        _sell_tox = self._assess_sell_toxicity()

        # 273# I6: halt 解除後の soft gate grace period
        _halt_recovery_active = self._daily_drawdown_guard.is_in_recovery(
            next_side,
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
            balance_forced=ctx.balance_forced,
            inv_net_imbalance=self._maker_price.inv_net_imbalance,
            is_buy_killed=self._is_side_killed("buy"),
            is_sell_killed=self._is_side_killed("sell"),
            spread_jpy=self._maker_price.last_spread,
            mid_price=self._maker_price.last_mid_price,
            price_velocity_bps=self._maker_price.last_mid_trend_bps,
            trending_sell_skip_count=self._trending_sell_skip_count,
            buy_side_insufficient=_buy_side_insufficient,
            buy_toxicity=_buy_tox,
            sell_toxicity=_sell_tox,
            halt_recovery_active=_halt_recovery_active,
        )

        if _gate_result.blocked:
            await self._handle_gate_block(st, ctx, _gate_result)
            return None

        # ゲート通過 → カウンタリセット
        self._consecutive_gate_blocks = 0
        if _gate_result.dual_kill_bypassed:
            self._inc_guard_fire("dual_kill_bypass")
        if (
            self.config.skip_sell_trending
            and next_side == "sell"
            and self._regime_detector is not None
            and self._regime_detector.current_regime.is_trending
        ):
            self._trending_sell_skip_count = 0

        return _gate_result

    async def _handle_gate_block(
        self,
        st: RunSessionState,
        ctx: CycleContext,
        gate_result: CycleGateResult,
    ) -> None:
        """194# ゲートブロック時の処理: record, quiescence, sleep."""
        next_side = ctx.next_side

        # カウンタ管理
        if gate_result.blocking_reason == "trending_sell_skip":
            self._trending_sell_skip_count += 1
            _max_c = self.config.max_consecutive_trending_sell_skip
            logger.info(
                f"[194#] {gate_result.blocking_reason} "
                f"[consecutive={self._trending_sell_skip_count}"
                f"/{_max_c if _max_c > 0 else '∞'}] "
                f"[{gate_result.audit_summary}]"
            )
        else:
            logger.info(
                f"[194#] Cycle gate blocked: {gate_result.blocking_reason} "
                f"[{gate_result.audit_summary}]"
            )

        await self._execute_skip(
            st, side=next_side,
            cancel_reason=gate_result.cancel_reason,
            order_quantity=self._current_lot,
            update_last_side=True, sleep=False,
        )

        if gate_result.blocking_reason:
            self._inc_guard_fire(f"gate_{gate_result.blocking_reason}")

        # 218#/242# 連続ゲートブロック + quiescence
        self._consecutive_gate_blocks += 1
        _quiescence_th = self.config.quiescence_gate_blocks_threshold
        _in_quiescence = (
            _quiescence_th > 0
            and self._consecutive_gate_blocks >= _quiescence_th
        )
        _gate_log_interval = max(
            5, self.config.quiescence_gate_blocks_threshold // 2,
        )
        if (
            self._consecutive_gate_blocks >= _gate_log_interval
            and self._consecutive_gate_blocks % _gate_log_interval == 0
        ):
            if _in_quiescence:
                self._inc_guard_fire("quiescence")
                logger.info(
                    f"[242#] QUIESCENCE: {self._consecutive_gate_blocks} "
                    f"consecutive gate blocks — no-trade accepted as normal "
                    f"(reason={gate_result.blocking_reason}, side={next_side}, "
                    f"sleep_cap={self.config.quiescence_sleep_sec:.0f}s)"
                )
            else:
                logger.warning(
                    f"[218#] DEADLOCK WARNING: {self._consecutive_gate_blocks} "
                    f"consecutive gate blocks (reason={gate_result.blocking_reason}, "
                    f"side={next_side})"
                )

        self._maybe_skip_state_save(
            st, f"gate_blocks={self._consecutive_gate_blocks}",
        )

        _q_sleep = (
            self.config.quiescence_sleep_sec
            if _in_quiescence and self.config.quiescence_sleep_sec > 0
            else 0.0
        )
        if gate_result.blocking_reason == "narrow_spread_pause":
            await asyncio.sleep(self.config.narrow_spread_pause_sec)
        else:
            await self._effective_sleep(max_override=_q_sleep)

    # ------------------------------------------------------------------
    # 240# Toxicity participation skip
    # ------------------------------------------------------------------
    async def _handle_toxicity_skip(
        self,
        st: RunSessionState,
        ctx: CycleContext,
        gate_result: CycleGateResult,
    ) -> bool:
        """240# Toxicity Budget: 確率的参加率チェック.

        ORANGE ゾーンでは 1/N の確率で参加 (Glosten-Milgrom)。
        True = スキップ (caller: continue)
        """
        if (
            gate_result.participation_rate < 1.0
            and random.random() > gate_result.participation_rate
        ):
            self._inc_guard_fire("toxicity_participation_skip")
            logger.info(
                f"[240#] Toxicity participation skip: "
                f"rate={gate_result.participation_rate:.2f}, "
                f"offset_mult={gate_result.toxicity_offset_mult:.2f} "
                f"(side={ctx.next_side})"
            )
            await self._execute_skip(
                st, side=ctx.next_side,
                cancel_reason=CR.TOXICITY_PARTICIPATION_SKIP,
                order_quantity=self._current_lot,
                update_last_side=True,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # 234# Degraded liquidation setup
    # ------------------------------------------------------------------
    async def _handle_degraded_liquidation(
        self,
        st: RunSessionState,
        ctx: CycleContext,
        gate_result: CycleGateResult,
    ) -> bool:
        """234# 縮退清算モードの duty cycle 管理.

        True = duty cycle スキップ (caller: continue)
        False = 実行可 (degraded or normal)
        gate_result.degraded_liquidation のフラグに基づく。
        """
        _degraded = gate_result.degraded_liquidation
        if _degraded:
            self._degraded_liquidation_duty_counter += 1
            _duty = max(self.config.degraded_liquidation_duty_cycle, 1)
            if _duty > 1 and (self._degraded_liquidation_duty_counter % _duty) != 1:
                logger.info(
                    f"[234#] Degraded liquidation duty skip: "
                    f"cycle {self._degraded_liquidation_duty_counter}/{_duty} "
                    f"(reason={gate_result.degraded_reason})"
                )
                self._inc_guard_fire("degraded_liquidation_duty_skip")
                await self._execute_skip(
                    st, side=ctx.next_side,
                    cancel_reason=CR.DEGRADED_LIQUIDATION_DUTY_SKIP,
                    order_quantity=self._current_lot,
                    update_last_side=True,
                )
                return True
            self._inc_guard_fire("degraded_liquidation_active")
            logger.warning(
                f"[234#] Degraded liquidation ACTIVE: "
                f"lot ×{self.config.degraded_liquidation_lot_mult:.1f}, "
                f"offset ×{self.config.degraded_liquidation_offset_mult:.1f} "
                f"(reason={gate_result.degraded_reason})"
            )
        else:
            if self._degraded_liquidation_duty_counter > 0:
                logger.info(
                    f"[235#] Degraded liquidation cleared after "
                    f"{self._degraded_liquidation_duty_counter} duty cycles"
                )
            self._degraded_liquidation_duty_counter = 0
            if self._inventory_escape_duty_counter > 0:
                logger.info(
                    f"[269#] Inventory escape cleared after "
                    f"{self._inventory_escape_duty_counter} duty cycles"
                )
            self._inventory_escape_duty_counter = 0
        return False

    # ------------------------------------------------------------------
    # 332# Cycle 実行 + one-sided tracking + 例外ハンドリング
    # ------------------------------------------------------------------
    async def _execute_and_track_cycle(
        self,
        st: RunSessionState,
        ctx: CycleContext,
        gate_result: CycleGateResult,
    ) -> None:
        """run_single_cycle 呼出し + one-sided エスカレーション + 例外処理.

        332# extract from run_continuous.
        例外時は continue 相当 (caller の while loop が継続)。
        """
        next_side = ctx.next_side
        _recovery_scale = 1.0

        try:
            # 224# B1: halt解除後ソフトリカバリ
            _recovery_scale = self._daily_drawdown_guard.consume_recovery_cycle(
                next_side,
            )
            if _recovery_scale < 1.0:
                self._inc_guard_fire("per_side_halt_recovery_active")
                if self._regime_detector is not None:
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
                balance_forced_switch=ctx.balance_forced,
                balance_forced_rescue=ctx.is_rescue,
                one_sided_balance=ctx.one_sided_balance,
                trending_offset_mult=gate_result.trending_offset_mult,
                degraded_liquidation=(
                    gate_result.degraded_liquidation or ctx.inventory_escape
                ),
                toxicity_offset_mult=gate_result.toxicity_offset_mult,
            )
            # 154# C-2: 実サイクル実行 → forced skip カウンタリセット
            self._balance_forced_skip_count = 0
            self._trending_sell_skip_count = 0

            # 207# §4: one-sided 連続実行追跡 + 234# エスカレーション
            self._track_one_sided_escalation(ctx, next_side)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping gracefully")
            self._kill_switch.kill("keyboard_interrupt")
            return
        except Exception as e:
            logger.error(f"Cycle execution error: {e}", exc_info=True)
            if _recovery_scale < 1.0:
                self._daily_drawdown_guard.restore_recovery_counter(next_side)
                logger.info(
                    f"[225# 6.1] Recovery counter restored for {next_side} "
                    f"(cycle aborted by exception)"
                )
            self._balance_checker.restore_lot_after_dust_sweep()
            self._last_side = next_side
            await self._effective_sleep()
            return

        self._balance_checker.restore_lot_after_dust_sweep()

        # 265# extract: post-cycle + adaptation
        self._process_post_cycle(record, next_side, st)
        await self._log_progress_and_adapt(next_side, st)

    def _track_one_sided_escalation(
        self, ctx: CycleContext, next_side: str,
    ) -> None:
        """207# §4 / 234#: one-sided 連続実行追跡 + freeze/cooldown エスカレーション."""
        if ctx.one_sided_balance:
            self._one_sided_consecutive_count += 1
            _os_limit = self.config.one_sided_consecutive_limit
            if _os_limit > 0 and self._one_sided_consecutive_count >= _os_limit:
                _over = self._one_sided_consecutive_count - _os_limit
                _freeze_off = self.config.one_sided_escalation_freeze_offset
                _cd_off = self.config.one_sided_escalation_cooldown_offset

                if _freeze_off > 0 and _over >= _freeze_off:
                    # Stage 3: freeze
                    _freeze_n = self.config.one_sided_escalation_freeze_cycles
                    self._one_sided_freeze_remaining = _freeze_n
                    self._one_sided_frozen_side = next_side
                    self._one_sided_consecutive_count = _os_limit
                    self._inc_guard_fire("one_sided_freeze")
                    logger.warning(
                        f"[234#] One-sided FREEZE: "
                        f"{self._one_sided_consecutive_count}/{_os_limit} "
                        f"(+{_over}) → freezing {next_side} for {_freeze_n} cycles"
                    )
                elif _cd_off > 0 and _over >= _cd_off:
                    # Stage 2: cooldown
                    _cd_n = self.config.one_sided_escalation_cooldown_cycles
                    self._one_sided_cooldown_remaining = _cd_n
                    self._one_sided_frozen_side = next_side
                    self._one_sided_consecutive_count = _os_limit
                    self._inc_guard_fire("one_sided_cooldown")
                    logger.warning(
                        f"[234#] One-sided COOLDOWN: "
                        f"{self._one_sided_consecutive_count}/{_os_limit} "
                        f"(+{_over}) → skip {_cd_n} cycles"
                    )
                else:
                    # Stage 1: interval 延長
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
            self._one_sided_frozen_side = None

    # ------------------------------------------------------------------
    # 332# Post-cycle sleep
    # ------------------------------------------------------------------
    async def _post_cycle_sleep(self, ctx: CycleContext) -> None:
        """サイクル完了後の sleep: regime 別 interval + config reload.

        332# extract from run_continuous.
        """
        # 169# Config Hot-Reload
        self._config_reloader.maybe_reload(self)

        if self._side_selector.rapid_exit_side is not None:
            interval = self.config.early_exit_rapid_interval_sec
            logger.info(
                f"[early_exit] Rapid exit: interval shortened to "
                f"{interval:.0f}s (next side={self._side_selector.rapid_exit_side})"
            )
        else:
            regime = self._current_regime_value()
            interval = self._cycle_strategy.effective_interval(regime)
            # 306# L1: σ 連動 dynamic cycle interval
            if self.config.dynamic_cycle_interval_enabled:
                interval = self._compute_dynamic_interval(interval)

        # 200# P0-2: soft drawdown interval 延長
        soft_dd_mult = self._soft_drawdown_interval_multiplier
        # 202# A: 単一サイクル大損失クールダウン
        _loss_cd = self._loss_cooldown_mult
        self._loss_cooldown_mult = 1.0

        # 207# §3 / 275# DRY: Toxic veto カウンタ減算
        self._tick_toxic_veto("cycle_end")

        # 207# §4: one-sided interval 延長
        _os_limit = self.config.one_sided_consecutive_limit
        _os_mult = 1.0
        if _os_limit > 0 and self._one_sided_consecutive_count >= _os_limit:
            _os_mult = self.config.one_sided_consecutive_interval_mult

        # 209# M4 / 215# P0-C: sleep 計算
        _alert_im = self._alert_interval_mult
        _raw_sleep = interval * soft_dd_mult * _loss_cd * _os_mult * _alert_im
        _max_sleep = self.config.max_cycle_sleep_sec
        _clamped = min(_raw_sleep, _max_sleep) if _max_sleep > 0 else _raw_sleep
        await asyncio.sleep(_clamped)
