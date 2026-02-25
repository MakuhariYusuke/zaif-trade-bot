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
from typing import Final, Optional, Protocol, runtime_checkable

from ztb.trading.orders.state_machine import OrderState

from scripts.v460.lib.fill_config import FillMonitorResult, FillTestConfig

logger = logging.getLogger(__name__)

# 定数
_BPS_FACTOR: Final[int] = 10_000


@runtime_checkable
class OrderLike(Protocol):
    """注文オブジェクトの型安全プロトコル (Any 排除)."""

    @property
    def order_id(self) -> str: ...


@runtime_checkable
class OrderStatusLike(Protocol):
    """注文ステータスオブジェクトのプロトコル."""

    @property
    def status(self) -> str: ...

    @property
    def price(self) -> Optional[float]: ...


class ExchangeAdapter(Protocol):
    """OrderMonitor が必要とする adapter メソッド群."""

    async def get_order_status(self, order_id: str) -> Optional[OrderStatusLike]: ...
    async def cancel_order(self, order_id: str) -> None: ...
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "limit",
    ) -> OrderLike: ...
    async def get_orderbook(self, symbol: str, depth: int = 1) -> object: ...


# ztb OrderState ↔ exchange status string マッピング
_STATUS_TO_STATE: dict[str, OrderState] = {
    "filled": OrderState.FILLED,
    "cancelled": OrderState.CANCELLED,
    "rejected": OrderState.REJECTED,
    "pending": OrderState.PENDING,
    "confirmed": OrderState.CONFIRMED,
    "partial": OrderState.PARTIAL,
}


def _parse_order_state(status_str: str) -> OrderState:
    """exchange status 文字列 → OrderState enum (型安全変換)."""
    return _STATUS_TO_STATE.get(status_str, OrderState.PENDING)


class OrderMonitor:
    """約定ポーリング監視 — FillTestRunner から分割.

    __slots__ でメモリフットプリントを制御。
    """

    __slots__ = ("_config",)

    def __init__(self, config: FillTestConfig) -> None:
        self._config = config

    async def monitor(
        self,
        adapter: ExchangeAdapter,
        order: OrderLike,
        order_price: float,
        side: str,
        t_submit: float,
        spread_at_order: Optional[float],
        effective_offset_ratio: float,
        *,
        shutdown_check: object,  # KillSwitch
        pending_order_setter: object,  # Callable[[str | None], None]
        get_mid_price: object,  # Callable[[], Awaitable[float]]
        compute_maker_price: object,  # Callable[[str], Awaitable[tuple]]
        skip_gate: object | None = None,
        regime_detector: object | None = None,
        current_lot: float = 0.001,
    ) -> FillMonitorResult:
        """約定ポーリング監視 + stale order cancel-replace.

        113# R1: run_single_cycle Phase 3-4.
        120# 型安全: order: Any → OrderLike Protocol.
        120# ztb 活用: OrderState enum で状態比較。
        """
        import asyncio

        cfg = self._config
        filled = False
        fill_price: Optional[float] = None
        t_fill: Optional[float] = None
        cancel_reason_poll: Optional[str] = None
        _cancel_failed_likely_filled = False  # 166# C.7
        elapsed = 0.0
        reprice_count = 0
        cumulative_drift_bps = 0.0  # 158# P1-3: 累積 drift
        min_order_btc: Final[float] = cfg.min_order_btc

        # ------ 144# R-1c/R-1d: レジーム別 reprice/timeout 調整 ------
        _regime_name: str | None = None
        if regime_detector is not None and hasattr(regime_detector, "current_regime"):
            _rr = regime_detector.current_regime  # type: ignore[union-attr]
            if _rr is not None:
                _regime_name = _rr.value
        # R-1d: effective timeout (base × regime multiplier)
        # 155# S-3: sell 側は専用 timeout を使用 (速い撤退が有利)
        _base_timeout = (
            cfg.order_timeout_sec_sell
            if side == "sell" and cfg.order_timeout_sec_sell is not None
            else cfg.order_timeout_sec
        )
        _timeout_mult = cfg.regime_timeout_multipliers.get(_regime_name, 1.0) if _regime_name else 1.0  # type: ignore[arg-type]
        _effective_timeout = _base_timeout * _timeout_mult
        if _timeout_mult != 1.0:
            logger.debug(
                f"[regime_timeout] {_regime_name} → timeout "
                f"{_base_timeout:.0f}s × {_timeout_mult:.2f} = {_effective_timeout:.0f}s"
            )
        # R-1c: reprice offset (applied after side-specific resolution)
        _regime_reprice_offset = cfg.regime_reprice_adjustments.get(_regime_name, 0) if _regime_name else 0  # type: ignore[arg-type]

        # 094# 発注時 mid price を stale 判定の基準にする
        mid_at_order: Optional[float] = None
        if cfg.stale_order_enabled:
            try:
                mid_at_order = await get_mid_price()  # type: ignore[operator]
            except Exception as e:
                logger.debug(f"[stale_order] mid_at_order 取得失敗 (stale 検出無効化): {e}")
        last_reprice_time = t_submit

        while elapsed < _effective_timeout and not shutdown_check.is_killed():  # type: ignore[union-attr]
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
                            if state == OrderState.FILLED:
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
                        if state in (OrderState.CANCELLED, OrderState.REJECTED):
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
                if state == OrderState.FILLED:
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
                elif state in (OrderState.CANCELLED, OrderState.REJECTED):
                    cancel_reason_poll = f"exchange_{status_order.status}"
                    logger.info(f"Order {status_order.status}: {order.order_id}")
                    break
            except Exception as e:
                logger.warning(f"Poll error: {e}")

            # --- 094# stale order 検出 & cancel-replace ---
            _stale_check_sec = (
                (cfg.stale_check_after_sec_buy if side == "buy" else cfg.stale_check_after_sec_sell)
                if (cfg.stale_check_after_sec_buy if side == "buy" else cfg.stale_check_after_sec_sell) is not None
                else cfg.stale_check_after_sec
            )
            _stale_drift = (
                (cfg.stale_drift_bps_buy if side == "buy" else cfg.stale_drift_bps_sell)
                if (cfg.stale_drift_bps_buy if side == "buy" else cfg.stale_drift_bps_sell) is not None
                else cfg.stale_drift_bps
            )
            _stale_max_rp_base = (
                (cfg.stale_max_reprice_buy if side == "buy" else cfg.stale_max_reprice_sell)
                if (cfg.stale_max_reprice_buy if side == "buy" else cfg.stale_max_reprice_sell) is not None
                else cfg.stale_max_reprice
            )
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
                    current_mid = await get_mid_price()  # type: ignore[operator]
                    drift_bps = abs(current_mid - mid_at_order) / mid_at_order * _BPS_FACTOR
                    is_drifting_away = (
                        (side == "buy" and current_mid > mid_at_order)
                        or (side == "sell" and current_mid < mid_at_order)
                    )
                    if drift_bps >= _stale_drift and is_drifting_away:
                        logger.info(
                            f"[stale_order] Price drifted {drift_bps:.1f}bps "
                            f"({side}: mid {mid_at_order:.0f}→{current_mid:.0f}). "
                            f"Cancelling & repricing (reprice #{reprice_count + 1})"
                        )
                        # 1) 既存注文キャンセル
                        try:
                            await adapter.cancel_order(order.order_id)
                        except Exception as cancel_err:
                            if "Failed to cancel" in str(cancel_err) or "not found" in str(cancel_err).lower():
                                try:
                                    recheck = await adapter.get_order_status(order.order_id)
                                    if recheck is not None and _parse_order_state(recheck.status) == OrderState.FILLED:
                                        filled = True
                                        fill_price = recheck.price if recheck.price else order_price
                                        t_fill = time.time()
                                        _cancel_failed_likely_filled = True  # 166# C.7
                                        logger.info(
                                            f"[stale_order] Order actually filled during cancel @ "
                                            f"{fill_price:.0f} JPY"
                                        )
                                except Exception:
                                    pass
                            if filled:
                                break
                            logger.warning(f"[stale_order] Cancel failed: {cancel_err}")
                            continue

                        # 2) 100# P0-6: SkipGate による reprice ガード
                        reprice_gate_skipped = False
                        if skip_gate is not None:
                            try:
                                from scripts.v460.ml.skip_gate import build_features_from_market_state
                                sg_regime = None
                                if regime_detector is not None and hasattr(regime_detector, "current_regime"):
                                    sg_regime = regime_detector.current_regime.value
                                rp_features = build_features_from_market_state(
                                    side=side,
                                    spread_jpy=spread_at_order or 0.0,
                                    offset_ratio=effective_offset_ratio,
                                    regime=sg_regime,
                                    recent_trades=None,
                                    market_timestamp=time.time(),
                                )
                                rp_decision = skip_gate.evaluate(rp_features, side=side)  # type: ignore[union-attr]
                                if rp_decision.should_skip:
                                    reprice_gate_skipped = True
                                    logger.info(
                                        f"[stale_order] SkipGate blocked reprice: "
                                        f"P(AS)={rp_decision.as_probability:.3f} "
                                        f">= {rp_decision.threshold_used:.3f}"
                                    )
                            except Exception as sg_err:
                                logger.debug(f"[stale_order] SkipGate check failed: {sg_err}")

                        if reprice_gate_skipped:
                            cancel_reason_poll = "stale_skip_gate_blocked"
                            break

                        try:
                            result = await compute_maker_price(side)  # type: ignore[operator]
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
                            reprice_lot = max(
                                min_order_btc,
                                int(current_lot / min_order_btc) * min_order_btc,
                            )
                            new_order = await adapter.place_order(
                                symbol=cfg.symbol,
                                side=side,
                                quantity=reprice_lot,
                                price=new_price,
                                order_type="limit",
                            )
                            order = new_order  # type: ignore[assignment]
                            order_price = new_price
                            mid_at_order = current_mid
                            last_reprice_time = time.time()
                            reprice_count += 1
                            cumulative_drift_bps += drift_bps  # 158# P1-3
                            pending_order_setter(order.order_id)  # type: ignore[operator]
                            logger.info(
                                f"[stale_order] Repriced {side} @ {new_price:.0f} JPY "
                                f"(id={order.order_id}, reprice #{reprice_count})"
                            )
                        except Exception as place_err:
                            logger.warning(
                                f"[stale_order] Reprice failed: {place_err}. "
                                f"Treating as cancelled."
                            )
                            cancel_reason_poll = "stale_reprice_failed"
                            break
                except Exception as stale_err:
                    logger.debug(f"[stale_order] Check failed (non-fatal): {stale_err}")

        # 4. 未約定 → キャンセル
        # 117# B-fix: stale_skip_gate_blocked / stale_reprice_failed は既にキャンセル済み
        _already_cancelled = cancel_reason_poll in (
            "stale_skip_gate_blocked",
            "stale_reprice_failed",
        )
        if not filled and not _already_cancelled:
            try:
                await adapter.cancel_order(order.order_id)
                logger.info(f"Cancelled unfilled order after {elapsed:.1f}s")
            except Exception as e:
                logger.warning(f"Cancel failed: {e}")
                if "Failed to cancel" in str(e) or "not found" in str(e).lower():
                    try:
                        recheck = await adapter.get_order_status(order.order_id)
                        if recheck is not None and _parse_order_state(recheck.status) == OrderState.FILLED:
                            filled = True
                            fill_price = (
                                recheck.price if recheck.price else order_price
                            )
                            t_fill = time.time()
                            cancel_reason_poll = None
                            _cancel_failed_likely_filled = True  # 166# C.7
                            logger.info(
                                f"[Bug11] Order was actually filled @ "
                                f"{fill_price:.0f} JPY (detected on cancel failure)"
                            )
                        else:
                            logger.info(
                                f"[Bug11] Recheck: order not found in transactions either"
                            )
                    except Exception as recheck_err:
                        logger.warning(f"[Bug11] Recheck failed: {recheck_err}")

        pending_order_setter(None)  # type: ignore[operator]

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
        )
