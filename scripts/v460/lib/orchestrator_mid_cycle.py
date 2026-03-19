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

        # 372# F1 Gap-1: SAC sidecar signal を読み込み gate に注入
        # 487# P0: signal_status も取得して attribution 可観測性を確保
        from scripts.v460.lib.sidecar_signal_io import (
            read_sidecar_signal_with_status,
        )

        _sidecar_signal, _sidecar_signal_status = read_sidecar_signal_with_status()

        # 487# P2: sidecar activity tracking (progress log 用)
        if _sidecar_signal_status == "fresh":
            st.sidecar_fresh_count += 1
        elif _sidecar_signal_status == "stale":
            st.sidecar_stale_count += 1
        else:
            st.sidecar_missing_count += 1

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
            sidecar_signal=_sidecar_signal,
            recovery_skew=ctx.recovery_skew,
        )

        # 487# P0: sidecar attribution を gate_result に転記
        _gate_result.sidecar_signal_status = _sidecar_signal_status
        if _sidecar_signal is not None:
            _gate_result.sidecar_confidence = _sidecar_signal.confidence
            _gate_result.sidecar_model_version = _sidecar_signal.model_version

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

        # 487# P2: cancel_reason distribution tracking
        _cr = gate_result.cancel_reason or gate_result.blocking_reason or "unknown"
        st.cancel_reason_counts[_cr] = st.cancel_reason_counts.get(_cr, 0) + 1

        if gate_result.blocking_reason:
            self._inc_guard_fire(f"gate_{gate_result.blocking_reason}")
        # 451# P1-2: compound suppression カウンタ
        for spec in gate_result.speculative_checks:
            if spec.blocked:
                self._inc_guard_fire(f"compound_{spec.gate_name}")

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
            # 459# narrow_spread_pause も _effective_sleep 経由に統一
            await self._effective_sleep(
                max_override=self.config.narrow_spread_pause_sec,
            )
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
                one_sided_balance=ctx.one_sided_balance,
                trending_offset_mult=gate_result.trending_offset_mult,
                degraded_liquidation=(
                    gate_result.degraded_liquidation or ctx.inventory_escape
                ),
                toxicity_offset_mult=gate_result.toxicity_offset_mult,
                sidecar_offset_bps=gate_result.sidecar_offset_bps,
                sidecar_bias=gate_result.sidecar_bias if gate_result.sidecar_bias != 0.0 else None,
                # 487# P0: sidecar attribution 可観測性
                sidecar_confidence=gate_result.sidecar_confidence,
                sidecar_model_version=gate_result.sidecar_model_version,
                sidecar_signal_status=gate_result.sidecar_signal_status,
            )
            # 420# P1: Side 切替可観測性 — CycleContext の情報を FillRecord に転記
            record.requested_side = ctx.requested_side
            record.resolved_side_reason = ctx.resolved_side_reason
            # 465# balance_forced_switch 一貫性: resolved_side_reason から導出
            if ctx.resolved_side_reason == "balance_switch":
                record.balance_forced_switch = True
            # 154# C-2: 実サイクル実行 → skip カウンタリセット
            self._trending_sell_skip_count = 0

            # 207# §4: one-sided 連続実行追跡 + 234# エスカレーション
            self._track_one_sided_escalation(ctx, next_side)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping gracefully")
            self._kill_switch.kill("keyboard_interrupt")
            return
        except (ConnectionError, TimeoutError, OSError) as e:
            # 488# P1-1: ネットワーク系は WARNING (頻発するため)
            logger.warning("Cycle execution network error: %s", e)
            if _recovery_scale < 1.0:
                self._daily_drawdown_guard.restore_recovery_counter(next_side)
                logger.info(
                    "[225# 6.1] Recovery counter restored for %s "
                    "(cycle aborted by network error)", next_side,
                )
            self._balance_checker.restore_lot_after_dust_sweep()
            self._last_side = next_side
            await self._effective_sleep()
            return
        except Exception as e:
            logger.error("Cycle execution error: %s", e, exc_info=True)
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

        # 372# dust buy-to-clear: buy 完了 → 次サイクルで sell 強制
        if self._balance_checker.dust_buy_pending and next_side == "buy":
            self._balance_checker.clear_dust_buy_pending()
            self._side_selector.rapid_exit_side = "sell"
            logger.info(
                "[dust_sweep] Buy-to-clear completed. "
                "Forcing sell next cycle for dust sweep."
            )

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
        # 475# メモリリーク防止: _effective_sleep を経由しない正常パスでも GC カウンタ進行
        self._gc_cycle_counter += 1
        if self._gc_cycle_counter >= self._GC_INTERVAL_CYCLES:
            import gc
            self._gc_cycle_counter = 0
            collected = gc.collect()
            if collected > 0:
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    f"[475# GC] post-cycle collected {collected} objects"
                )
        await asyncio.sleep(_clamped)
