"""120# OrderMonitor — 約定ポーリング監視モジュール.

run_fill_test.py FillTestRunner からの God Object 分割:
- _monitor_fill_polling (267L) → monitor()
- stale order 検出 & cancel-replace (094#)
- SkipGate reprice ガード (100# P0-6)

型安全: OrderState enum 活用 (ztb.trading.orders.state_machine)、
        order パラメータの Any 型排除。
メモリ: __slots__ 適用。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Final, Protocol, cast

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.fill_config import FillMonitorResult, FillTestConfig
from ztb.ml.skip_gate_contracts import SkipDecisionLike as _SkipDecisionLike
from ztb.ml.skip_gate_contracts import SkipGateLike as _SkipGateLike
from ztb.trading.execution.contracts import ExchangeAdapter, OrderLike, OrderStatusLike
from ztb.trading.execution.stale_order_policy import (
    CancelFillCheck as _CancelFillCheck,
    ORDER_STATE_CANCELLED as _STATE_CANCELLED,
    ORDER_STATE_CONFIRMED as _STATE_CONFIRMED,
    ORDER_STATE_FILLED as _STATE_FILLED,
    ORDER_STATE_PARTIAL as _STATE_PARTIAL,
    ORDER_STATE_PENDING as _STATE_PENDING,
    ORDER_STATE_REJECTED as _STATE_REJECTED,
    parse_order_state as _parse_order_state,
)
from ztb.trading.signal.regime.regime_detector import RegimeDetectorLike

logger = logging.getLogger(__name__)

from scripts.v460.lib.constants import BPS_FACTOR as _BPS_FACTOR


class _KillSwitchLike(Protocol):
    """175# shutdown_check の型安全 Protocol."""

    def is_killed(self) -> bool: ...

class OrderMonitor:
    """約定ポーリング監視 — FillTestRunner から分割.

    __slots__ でメモリフットプリントを制御。
    """

    __slots__ = ("_config",)

    def __init__(self, config: FillTestConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # 262# DRY: cancel → fill recheck ヘルパー (3箇所の重複を統合)
    # ------------------------------------------------------------------
    @staticmethod
    async def _try_cancel_with_fill_recheck(
        adapter: ExchangeAdapter,
        order_id: str,
        fallback_price: float,
    ) -> _CancelFillCheck:
        """注文キャンセルを試行し、失敗時に fill 済みかを再確認する.

        cancel_order() → 例外 → "Failed to cancel" / "not found" 文字列
        パターンで約定済み判定 → get_order_status で fill 確認。

        Returns:
            _CancelFillCheck: was_filled=True なら約定済み検出。
        """
        try:
            await adapter.cancel_order(order_id)
            return _CancelFillCheck()
        except Exception as cancel_err:
            cancel_msg = str(cancel_err)
            cancel_lower = cancel_msg.lower()
            if "failed to cancel" not in cancel_lower and "not found" not in cancel_lower:
                # 予期しないキャンセルエラー → 呼び出し側で処理
                logger.warning(f"Cancel order unexpected error: {cancel_err}")
                return _CancelFillCheck(cancel_succeeded=False)

            # "Failed to cancel" / "not found" → 約定済みの可能性 → recheck
            try:
                recheck = await adapter.get_order_status(order_id)
                if (
                    recheck is not None
                    and _parse_order_state(recheck.status) == _STATE_FILLED
                ):
                    price = recheck.price if recheck.price else fallback_price
                    logger.info(
                        f"[cancel_recheck] Order actually filled during cancel "
                        f"@ {price:.0f} JPY"
                    )
                    return _CancelFillCheck(
                        was_filled=True,
                        fill_price=price,
                        t_fill=time.time(),
                    )
            except Exception as exc:
                logger.debug("Recheck after cancel-fail raised: %s", exc)

            logger.warning(f"Cancel failed (order may be gone): {cancel_err}")
            return _CancelFillCheck(cancel_succeeded=False)

    @staticmethod
    def _resolve_regime_name(regime_detector: RegimeDetectorLike | None) -> str | None:
        """257# regime detector から現在レジーム名を型安全に取得する.

        Protocol 化により duck-typing の動的属性検査を排除。
        """
        if regime_detector is None:
            return None
        return regime_detector.current_regime.value

    def _should_block_reprice_with_skip_gate(
        self,
        *,
        skip_gate: _SkipGateLike | None,
        side: str,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        regime_detector: RegimeDetectorLike | None,
        market_timestamp: float,
    ) -> bool:
        """reprice 前の SkipGate ガードを共通処理する."""
        if skip_gate is None:
            return False
        try:
            from ztb.ml.skip_gate import build_features_from_market_state

            rp_features = build_features_from_market_state(
                side=side,
                spread_jpy=spread_at_order or 0.0,
                offset_ratio=effective_offset_ratio,
                regime=self._resolve_regime_name(regime_detector) or "unknown",
                recent_trades=None,
                market_timestamp=market_timestamp,
            )
            # 255# getattr → 直接参照 (FillTestConfig.stale_reprice_skip_gate_offset: float = 0.0)
            threshold_offset = -self._config.stale_reprice_skip_gate_offset
            rp_decision = cast(
                _SkipDecisionLike,
                skip_gate.evaluate(
                    rp_features,
                    side=side,
                    threshold_offset=threshold_offset,
                ),
            )
            if not rp_decision.should_skip:
                return False

            as_prob = rp_decision.as_probability
            threshold_used = rp_decision.threshold_used
            if as_prob is not None and threshold_used is not None:
                logger.info(
                    "[stale_order] SkipGate blocked reprice: P(AS)=%.3f >= %.3f",
                    as_prob,
                    threshold_used,
                )
            else:
                logger.info("[stale_order] SkipGate blocked reprice")
            return True
        except Exception as sg_err:
            logger.debug("[stale_order] SkipGate check failed: %s", sg_err)
            return False

    async def monitor(
        self,
        adapter: ExchangeAdapter,
        order: OrderLike,
        order_price: float,
        side: str,
        t_submit: float,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        *,
        shutdown_check: _KillSwitchLike,
        pending_order_setter: Callable[[str | None], None],
        get_mid_price: Callable[[], Awaitable[float]],
        compute_maker_price: Callable[[str], Awaitable[tuple[float, float, float]]],
        skip_gate: _SkipGateLike | None = None,
        regime_detector: RegimeDetectorLike | None = None,
        current_lot: float = 0.001,
        chase_drift_bps_override: float | None = None,      # 179# Chase
        chase_max_reprice_override: int | None = None,       # 179# Chase
    ) -> FillMonitorResult:
        """約定ポーリング監視 + stale order cancel-replace.

        113# R1: run_single_cycle Phase 3-4.
        120# 型安全: order: Any → OrderLike Protocol.
        120# ztb 活用: OrderState enum で状態比較。
        """
        import asyncio

        cfg = self._config
        filled = False
        fill_price: float | None = None
        t_fill: float | None = None
        cancel_reason_poll: str | None = None
        _cancel_failed_likely_filled = False  # 166# C.7
        elapsed = 0.0
        reprice_count = 0
        cumulative_drift_bps = 0.0  # 158# P1-3: 累積 drift
        min_order_btc: Final[float] = cfg.min_order_btc

        # ------ 144# R-1c/R-1d: レジーム別 reprice/timeout 調整 ------
        _regime_name = self._resolve_regime_name(regime_detector)
        # R-1d: effective timeout (base × regime multiplier)
        # 155# S-3: sell 側は専用 timeout を使用 (速い撤退が有利)
        _base_timeout = (
            cfg.order_timeout_sec_sell
            if side == "sell" and cfg.order_timeout_sec_sell is not None
            else cfg.order_timeout_sec
        )
        _timeout_mult = 1.0
        if _regime_name is not None:
            _timeout_mult = cfg.regime_timeout_multipliers.get(_regime_name, 1.0)
        _effective_timeout = _base_timeout * _timeout_mult
        # 506# P0: sell age cap — sell 注文の最大滞留時間を制限
        # ranging 30–50s バケットに -158.73 JPY 集中 → 25s キャップで回避
        if (
            side == "sell"
            and cfg.sell_age_cap_sec is not None
            and cfg.sell_age_cap_sec > 0
        ):
            _pre_cap = _effective_timeout
            _effective_timeout = min(_effective_timeout, cfg.sell_age_cap_sec)
            if _effective_timeout < _pre_cap:
                logger.info(
                    "[506#] sell_age_cap enforced: %.0fs → %.0fs (cap=%ds)",
                    _pre_cap,
                    _effective_timeout,
                    cfg.sell_age_cap_sec,
                )
        if _timeout_mult != 1.0:
            logger.info(
                f"[regime_timeout] {_regime_name} → timeout "
                f"{_base_timeout:.0f}s × {_timeout_mult:.2f} = {_effective_timeout:.0f}s"
            )
        # R-1c: reprice offset (applied after side-specific resolution)
        _regime_reprice_offset = (
            cfg.regime_reprice_adjustments.get(_regime_name, 0)
            if _regime_name is not None
            else 0
        )

        # 094# 発注時 mid price を stale 判定の基準にする
        mid_at_order: float | None = None
        if cfg.stale_order_enabled:
            try:
                mid_at_order = await get_mid_price()
            except Exception as e:
                logger.debug(f"[stale_order] mid_at_order 取得失敗 (stale 検出無効化): {e}")
        last_reprice_time = t_submit
        _consecutive_poll_errors = 0  # 373# F9: 連続 poll エラーカウンタ

        while elapsed < _effective_timeout and not shutdown_check.is_killed():
            await asyncio.sleep(cfg.poll_interval_sec)
            elapsed = time.time() - t_submit

            try:
                status_order = await adapter.get_order_status(order.order_id)
                if status_order is None:
                    _retry_delays = cfg.status_unknown_retry_delays
                    _recovered = False
                    for _retry_i, _delay in enumerate(_retry_delays):
                        logger.warning(
                            f"Order {order.order_id} not found — "
                            f"retry {_retry_i + 1}/{len(_retry_delays)} after {_delay}s"
                        )
                        await asyncio.sleep(_delay)
                        status_order = await adapter.get_order_status(order.order_id)
                        if status_order is not None:
                            state = _parse_order_state(status_order.status)
                            if state == _STATE_FILLED:
                                filled = True
                                fill_price = (
                                    status_order.price
                                    if status_order.price
                                    else order_price
                                )
                                t_fill = time.time()
                                logger.info(
                                    f"Order confirmed filled on retry {_retry_i + 1} @ "
                                    f"{fill_price:.0f} JPY"
                                )
                                _recovered = True
                                break
                            else:
                                _recovered = True
                                break
                    if _recovered and filled:
                        break
                    if _recovered and status_order is not None:
                        state = _parse_order_state(status_order.status)
                        if state in (_STATE_CANCELLED, _STATE_REJECTED):
                            cancel_reason_poll = f"exchange_{status_order.status}"
                            logger.info(f"Order {status_order.status}: {order.order_id}")
                            break
                        continue
                    # 122# E12 Fix: postonly_reject 推定の精度向上
                    # elapsed だけでなく spread_at_order も条件に含めて
                    # status_unknown との誤分類を低減
                    # 156# §10 #3: 定数経由で統一 (旧 "postonly_reject" → "post_only_reject")
                    is_fast_cancel = elapsed < cfg.poll_interval_sec * 3
                    is_narrow_spread = (
                        spread_at_order is not None
                        and cfg.min_spread_jpy > 0
                        and spread_at_order < cfg.min_spread_jpy * 2
                    )
                    if is_fast_cancel and is_narrow_spread:
                        reason = "post_only_reject"
                    elif is_fast_cancel:
                        reason = "status_unknown_fast"
                    else:
                        reason = "status_unknown"
                    logger.warning(
                        f"Order {order.order_id} status unknown after "
                        f"{len(_retry_delays)} retries "
                        f"— treating as cancelled ({reason}, elapsed={elapsed:.1f}s)"
                    )
                    cancel_reason_poll = reason
                    break

                state = _parse_order_state(status_order.status)
                if state == _STATE_FILLED:
                    filled = True
                    fill_price = (
                        status_order.price if status_order.price else order_price
                    )
                    t_fill = time.time()
                    logger.info(
                        f"Order filled @ {fill_price:.0f} JPY, "
                        f"wait={elapsed:.1f}s"
                    )
                    break
                elif state in (_STATE_CANCELLED, _STATE_REJECTED):
                    cancel_reason_poll = f"exchange_{status_order.status}"
                    logger.info(f"Order {status_order.status}: {order.order_id}")
                    break
            except Exception as e:
                _consecutive_poll_errors += 1
                logger.warning(f"Poll error ({_consecutive_poll_errors}): {e}")
                # 373# F9: 連続 poll エラーが閾値超過 → ループ脱出して cancel
                if _consecutive_poll_errors >= 5:
                    logger.error(
                        f"[373# F9] {_consecutive_poll_errors} consecutive poll errors "
                        f"— giving up monitoring, will cancel order"
                    )
                    cancel_reason_poll = CR.POLL_ERROR_LIMIT
                    break

            # --- 094# stale order 検出 & cancel-replace ---
            # 200# 10-B: 冗長 ternary 解消 — side 別値を先に解決
            _side_check_sec = cfg.stale_check_after_sec_buy if side == "buy" else cfg.stale_check_after_sec_sell
            _stale_check_sec = _side_check_sec if _side_check_sec is not None else cfg.stale_check_after_sec

            _side_drift = cfg.stale_drift_bps_buy if side == "buy" else cfg.stale_drift_bps_sell
            _stale_drift = _side_drift if _side_drift is not None else cfg.stale_drift_bps

            _side_max_rp = cfg.stale_max_reprice_buy if side == "buy" else cfg.stale_max_reprice_sell
            _stale_max_rp_base = _side_max_rp if _side_max_rp is not None else cfg.stale_max_reprice
            # 179# Chase: trending 時は低い drift 閾値 & 高い reprice 上限を適用
            if chase_drift_bps_override is not None:
                _stale_drift = chase_drift_bps_override
            if chase_max_reprice_override is not None:
                _stale_max_rp_base = chase_max_reprice_override
            # 144# R-1c: レジーム別 reprice オフセット (最低0でクランプ)
            _stale_max_rp = max(0, _stale_max_rp_base + _regime_reprice_offset)
            if (
                cfg.stale_order_enabled
                and not filled
                and mid_at_order is not None
                and elapsed >= _stale_check_sec
                and reprice_count < _stale_max_rp
                and (time.time() - last_reprice_time) >= cfg.stale_cooldown_sec
            ):
                try:
                    current_mid = await get_mid_price()
                    drift_bps = abs(current_mid - mid_at_order) / mid_at_order * _BPS_FACTOR
                    # 200# P0-1: adverse drift = cancel-only (逆選択回避)
                    # buy で mid↑ / sell で mid↓ は不利方向 → 追わない
                    is_adverse_drift = (
                        (side == "buy" and current_mid > mid_at_order)
                        or (side == "sell" and current_mid < mid_at_order)
                    )
                    # 順方向 drift (buy で mid↓ / sell で mid↑) は有利 → reprice ok
                    is_favorable_drift = (
                        (side == "buy" and current_mid < mid_at_order)
                        or (side == "sell" and current_mid > mid_at_order)
                    )
                    if drift_bps >= _stale_drift and is_adverse_drift:
                        # 200# 不利方向: cancel-only で撤退 (MM理論: 逆選択特攻阻止)
                        logger.info(
                            f"[stale_order] Adverse drift {drift_bps:.1f}bps "
                            f"({side}: mid {mid_at_order:.0f}→{current_mid:.0f}). "
                            f"Cancel-only — not chasing adverse direction"
                        )
                        # 262# DRY: cancel-recheck ヘルパー
                        chk = await self._try_cancel_with_fill_recheck(
                            adapter, order.order_id, order_price,
                        )
                        if chk.was_filled:
                            filled = True
                            fill_price = chk.fill_price
                            t_fill = chk.t_fill
                            _cancel_failed_likely_filled = True
                        if not filled:
                            cancel_reason_poll = CR.STALE_ADVERSE_DRIFT
                        break
                    if drift_bps >= _stale_drift and is_favorable_drift:
                        # 509# 残時間チェック: reprice cycle (cancel+place) に 3s 以上必要
                        _remaining = _effective_timeout - elapsed
                        if _remaining < 3.0:
                            logger.info(
                                "[509#] Reprice skipped: %.1fs remaining < 3s min "
                                "(elapsed=%.1fs, timeout=%.0fs)",
                                _remaining, elapsed, _effective_timeout,
                            )
                            continue
                        # 292# BS-1: cancel 前に deadband 判定 — queue position 保護
                        # compute_maker_price を先に呼び、価格差が小さければ
                        # cancel せずに既存注文を維持 (queue priority 保全)
                        _pre_cancel_new_price: float | None = None
                        _deadband_skip = False
                        _min_delta = cfg.stale_reprice_min_delta_jpy
                        if _min_delta > 0:
                            try:
                                _pre_result = await compute_maker_price(side)
                                _pre_cancel_new_price = _pre_result[0]
                                # 158# P1-2: reprice offset tightening (事前計算)
                                _tighten = cfg.stale_reprice_tighten
                                if _tighten != 1.0 and current_mid > 0:
                                    _gap = abs(_pre_cancel_new_price - current_mid)
                                    _tightened_gap = _gap * _tighten
                                    if side == "buy":
                                        _pre_cancel_new_price = round(current_mid - _tightened_gap)
                                    else:
                                        _pre_cancel_new_price = round(current_mid + _tightened_gap)
                                if abs(_pre_cancel_new_price - order_price) < _min_delta:
                                    _deadband_skip = True
                                    logger.info(
                                        f"[stale_order] Reprice deadband (pre-cancel): "
                                        f"|{_pre_cancel_new_price:.0f} - {order_price:.0f}| = "
                                        f"{abs(_pre_cancel_new_price - order_price):.0f} < "
                                        f"min_delta={_min_delta:.0f} JPY. "
                                        f"Keeping existing order to protect queue position"
                                    )
                            except Exception as pre_err:
                                logger.debug(f"[stale_order] Pre-cancel price check failed: {pre_err}")

                        if _deadband_skip:
                            continue

                        logger.info(
                            f"[stale_order] Favorable drift {drift_bps:.1f}bps "
                            f"({side}: mid {mid_at_order:.0f}→{current_mid:.0f}). "
                            f"Cancelling & repricing (reprice #{reprice_count + 1})"
                        )
                        # 1) 262# DRY: cancel-recheck ヘルパー
                        chk = await self._try_cancel_with_fill_recheck(
                            adapter, order.order_id, order_price,
                        )
                        if chk.was_filled:
                            filled = True
                            fill_price = chk.fill_price
                            t_fill = chk.t_fill
                            _cancel_failed_likely_filled = True  # 166# C.7
                            break
                        if not chk.cancel_succeeded:
                            continue

                        # 2) 100# P0-6: SkipGate による reprice ガード
                        reprice_check_ts = time.time()
                        reprice_gate_skipped = self._should_block_reprice_with_skip_gate(
                            skip_gate=skip_gate,
                            side=side,
                            spread_at_order=spread_at_order,
                            effective_offset_ratio=effective_offset_ratio,
                            regime_detector=regime_detector,
                            market_timestamp=reprice_check_ts,
                        )

                        if reprice_gate_skipped:
                            cancel_reason_poll = CR.STALE_SKIP_GATE_BLOCKED
                            break

                        try:
                            result = await compute_maker_price(side)
                            new_price = result[0]
                            # 158# P1-2: reprice offset tightening
                            _tighten = cfg.stale_reprice_tighten
                            if _tighten != 1.0 and current_mid > 0:
                                _gap = abs(new_price - current_mid)
                                _tightened_gap = _gap * _tighten
                                if side == "buy":
                                    new_price = round(current_mid - _tightened_gap)
                                else:
                                    new_price = round(current_mid + _tightened_gap)
                                logger.debug(
                                    f"[stale_order] Offset tightened: "
                                    f"gap {_gap:.0f}→{_tightened_gap:.0f} "
                                    f"(factor={_tighten})"
                                )
                            # 476#: Coincheck は satoshi 精度 — 0.001 単位切り捨て廃止
                            reprice_lot = max(
                                min_order_btc,
                                round(current_lot, 8),
                            )
                            new_order = await adapter.place_order(
                                symbol=cfg.symbol,
                                side=side,
                                quantity=reprice_lot,
                                price=new_price,
                                order_type="limit",
                            )
                            order = new_order
                            order_price = new_price
                            mid_at_order = current_mid
                            last_reprice_time = reprice_check_ts
                            reprice_count += 1
                            cumulative_drift_bps += drift_bps  # 158# P1-3
                            pending_order_setter(order.order_id)
                            logger.info(
                                f"[stale_order] Repriced {side} @ {new_price:.0f} JPY "
                                f"(id={order.order_id}, reprice #{reprice_count})"
                            )
                        except Exception as place_err:
                            logger.warning(
                                f"[stale_order] Reprice failed: {place_err}. "
                                f"Treating as cancelled."
                            )
                            cancel_reason_poll = CR.STALE_REPRICE_FAILED
                            break
                except Exception as stale_err:
                    logger.debug(f"[stale_order] Check failed (non-fatal): {stale_err}")

        # 4. 未約定 → キャンセル
        # 117# B-fix: stale_skip_gate_blocked / stale_reprice_failed は既にキャンセル済み
        # 200# P0-1: stale_adverse_drift も既にキャンセル済み
        _already_cancelled = cancel_reason_poll in (
            CR.STALE_SKIP_GATE_BLOCKED,
            CR.STALE_REPRICE_FAILED,
            CR.STALE_ADVERSE_DRIFT,
        )
        if not filled and not _already_cancelled:
            # 262# DRY: cancel-recheck ヘルパー
            chk = await self._try_cancel_with_fill_recheck(
                adapter, order.order_id, order_price,
            )
            if chk.was_filled:
                filled = True
                fill_price = chk.fill_price
                t_fill = chk.t_fill
                cancel_reason_poll = None
                _cancel_failed_likely_filled = True  # 166# C.7
            elif chk.cancel_succeeded:
                logger.info(f"Cancelled unfilled order after {elapsed:.1f}s")

        pending_order_setter(None)

        # 237# phantom position guard: status_unknown 時の注文 ID を記録
        _order_id_for_reconciliation: str | None = None
        if (
            not filled
            and cancel_reason_poll is not None
            and cancel_reason_poll.startswith("status_unknown")
        ):
            _order_id_for_reconciliation = order.order_id

        return FillMonitorResult(
            filled=filled,
            fill_price=fill_price,
            t_fill=t_fill,
            cancel_reason=cancel_reason_poll,
            queue_wait=elapsed,
            reprice_count=reprice_count,
            reprice_drift_bps=cumulative_drift_bps,  # 158# P1-3
            final_order_price=order_price,
            effective_timeout=_effective_timeout,
            cancel_failed_likely_filled=_cancel_failed_likely_filled,  # 166# C.7
            order_id_for_reconciliation=_order_id_for_reconciliation,  # 237#
        )
