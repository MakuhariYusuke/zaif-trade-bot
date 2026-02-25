"""163# Mixin: FillCycleExecutorMixin -- run_single_cycle + 直接依存メソッド.

1 サイクル: 発注 → 約定監視 → PnL 計測 → FillRecord 構築。

WARNING -- AI Coding Agent / 人間開発者への注意:
    このファイルは Mixin クラスであり、単独でインスタンス化しないこと。
    FillTestRunner.__init__ で生成される属性に依存する。
    責務: 1 取引サイクルの実行 (OB取得, SkipGate, 発注, 約定監視, PnL計測)
    run_continuous のループ制御ロジックを追加しないこと。
    side kill / time filter / balance forced 判定は fill_loop_orchestrator に属する。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.fill_config import (
    SkipGateResult as _SkipGateResult,
    FillMonitorResult as _FillMonitorResult,
    PnlMeasurement as _PnlMeasurement,
)
from scripts.v460.lib.resilience import CircuitState
from ztb.metrics.fill_quality import FillRecord

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig

logger = logging.getLogger(__name__)


class FillCycleExecutorMixin:
    """run_single_cycle + OB/SkipGate/Fill/PnL ヘルパー (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: 1 取引サイクル実行, OB ラッパー, SkipGate, Fill 監視, PnL 計測
      NG: ループ制御, side kill, time filter, balance forced
    MAX LINES: 700 (超えたら run_single_cycle 内のフェーズを分割せよ)
    ────────────────────────────────────────────────────
    """

    async def _compute_orderbook_imbalance(self, depth: int = 5) -> tuple[float, float, float]:
        """054# S1: 板不均衡を計算 — 120# MakerPriceCalculator に委譲."""
        r = await self._maker_price.compute_imbalance(self.adapter, self.config.symbol, depth=depth)
        return r.imbalance, r.bid_total, r.ask_total

    async def _get_mid_price(self) -> float:
        """板の best bid/ask から mid price を算出 — 120# MakerPriceCalculator に委譲."""
        return await self._maker_price.get_mid_price(self.adapter, self.config.symbol)

    async def _compute_maker_price(self, side: str) -> tuple[float, float, float]:
        """maker limit 価格を算出 — 120# MakerPriceCalculator に委譲."""
        r = await self._maker_price.compute(side, self.adapter, self.config.symbol)
        return r.price, r.spread, r.effective_offset_ratio

    # ==================================================================
    # 113# R1: run_single_cycle から抽出したサブメソッド
    # ==================================================================

    async def _evaluate_skip_gate(
        self,
        side: str,
        cycle_id: str,
        order_price: float,
        spread_at_order: Optional[float],
        effective_offset_ratio: float,
        *,
        order_lot: float | None = None,
    ) -> _SkipGateResult:
        """SkipGate ML 判定 — 121# SkipGateEvaluator に委譲.

        145# §9-#4: order_lot を渡してレジーム倍率適用後のロットで記録.
        """
        regime_value = (
            self._regime_detector.current_regime.value
            if self._regime_detector is not None
            else None
        )
        _lot = order_lot if order_lot is not None else self._current_lot
        return await self._skip_gate_evaluator.evaluate(
            side=side,
            cycle_id=cycle_id,
            order_price=order_price,
            spread_at_order=spread_at_order,
            effective_offset_ratio=effective_offset_ratio,
            adapter=self.adapter,
            symbol=self.config.symbol,
            current_lot=_lot,
            run_id=self._run_id,
            git_sha=self._git_sha,
            regime_value=regime_value,
            last_imbalance=self._maker_price._last_imbalance,
            last_bid_depth=self._maker_price._last_bid_depth,
            last_ask_depth=self._maker_price._last_ask_depth,
            imbalance_enabled=self.config.imbalance_enabled,
            maker_price_vpin_setter=lambda v: setattr(self._maker_price, '_last_vpin', v),
        )

    async def _monitor_fill_polling(
        self,
        order: object,
        order_price: float,
        side: str,
        t_submit: float,
        spread_at_order: Optional[float],
        effective_offset_ratio: float,
        *,
        order_lot: float | None = None,
    ) -> _FillMonitorResult:
        """約定ポーリング監視 — 120# OrderMonitor に委譲.

        120# 型安全: order: Any → object (OrderLike Protocol 準拠)。
        145# fix §9-#1: current_lot → order_lot (regime 調整済みの正しいロット)。
        """
        _lot = order_lot if order_lot is not None else self._current_lot

        def _set_pending(oid: str | None) -> None:
            self._pending_order_id = oid

        return await self._order_monitor.monitor(
            adapter=self.adapter,
            order=order,  # type: ignore[arg-type]
            order_price=order_price,
            side=side,
            t_submit=t_submit,
            spread_at_order=spread_at_order,
            effective_offset_ratio=effective_offset_ratio,
            shutdown_check=self._kill_switch,
            pending_order_setter=_set_pending,
            get_mid_price=self._get_mid_price,
            compute_maker_price=self._compute_maker_price,
            skip_gate=self._skip_gate,
            regime_detector=self._regime_detector,
            current_lot=_lot,
        )

    async def _measure_post_fill_pnl(
        self,
        filled: bool,
        fill_price: Optional[float],
        side: str,
    ) -> _PnlMeasurement:
        """約定後 PnL 計測 — 120# PnlMeasurer に委譲."""
        pnl = await self._pnl_measurer.measure(
            filled=filled,
            fill_price=fill_price,
            side=side,
            get_mid_price=self._get_mid_price,
        )
        # 054# S3: early exit → rapid exit フラグ
        if pnl.early_exit_triggered:
            self._side_selector.set_rapid_exit(side)
        return pnl

    async def run_single_cycle(
        self,
        side_override: str | None = None,
        balance_forced_switch: bool = False,
        balance_forced_rescue: bool = False,
    ) -> FillRecord:
        """1 サイクル: 発注 → 監視 → 結果記録.

        009# §4.2 の流れに準拠.
        041# 時間帯フィルター・残高チェック追加.
        055# Fix: side 決定前に最新 imbalance を取得.
        075# Fix: side_override で run_continuous() が決定した side を強制適用.
        129# D.2: balance_forced_switch フラグを FillRecord に記録.
        158# P1-1: balance_forced_rescue — offset 倍増で安全にポジション解消.
        """
        self._cycle_count += 1
        cycle_id = self._new_cycle_id()

        # 113# resilience: CircuitBreaker ガード — OPEN 中は API 呼出しを回避
        if self._circuit_breaker.state == CircuitState.OPEN:
            if not self._circuit_breaker.should_attempt_reset():
                logger.warning(
                    f"[circuit_breaker] OPEN — skipping cycle {self._cycle_count} "
                    f"(recovery in {self._circuit_breaker.config.recovery_timeout}s)"
                )
                return self._make_skip_record(
                    side=side_override or "buy",
                    cancel_reason=CR.CIRCUIT_BREAKER_OPEN,
                    cycle_id=cycle_id,
                    regime=self._current_regime_value(),  # 160#
                )

        # 055# Fix #2: Smart Side 判定用に最新板 imbalance を事前取得
        # (_compute_maker_price 内での取得では side 決定後 → 1サイクル遅延)
        # 122# §7.3 方法 2: OB データ記録のため常時計算 (smart_side 無効時もデータ蓄積)
        try:
            imb, bid_d, ask_d = await self._compute_orderbook_imbalance(
                depth=self.config.imbalance_depth,
            )
            self._maker_price._last_imbalance = imb
            self._maker_price._last_bid_depth = bid_d
            self._maker_price._last_ask_depth = ask_d
            # 129# OB recorder: サイクルごとに板スナップショットを記録
            ob = self._maker_price._last_ob_snapshot
            if ob is not None:
                self._ob_recorder.record(ob.bids, ob.asks, ob.timestamp)
        except Exception as e:
            logger.warning(f"[ob_prefetch] Pre-fetch imbalance failed, using last: {e}")
            # フォールバック: 前回値を維持

        # 135# P0-04: trades recorder — OB とは独立した try で障害分離 (§9.1 #3)
        try:
            recent = await self.adapter.get_recent_trades(
                self.config.symbol, limit=self.config.trades_recorder_fetch_limit,
            )
            self._trades_recorder.record_from_adapter(recent)
        except Exception as te:
            logger.debug(f"Trades fetch for recording skipped: {te}")

        # 075# Fix: side_override があればそれを使い、_next_side() 二重呼出を防止
        if side_override is not None:
            side = side_override
        else:
            side = self._next_side()
        # 054# S2: 連続同 side カウンタ更新 — 121# SideSelector に委譲
        self._side_selector.update_after_decision(side)

        logger.info(f"=== Cycle {self._cycle_count} ({side}) ===")

        # 1. maker limit 価格算出
        spread_at_order: Optional[float] = None
        effective_offset_ratio: float = self.config.spread_offset_ratio
        try:
            order_price, spread_at_order, effective_offset_ratio = await self._compute_maker_price(side)
        except Exception as e:
            # 130# orderbook_error 細分化
            err_msg = str(e).lower()
            if "timeout" in err_msg or "timed out" in err_msg:
                ob_cancel_reason = "orderbook_timeout"
                logger.error(f"Failed to compute maker price: {e}")
            elif "rate" in err_msg or "limit" in err_msg or "too many" in err_msg:
                ob_cancel_reason = "orderbook_rate_limit"
                logger.error(f"Failed to compute maker price: {e}")
            elif "empty" in err_msg or "no bid" in err_msg or "no ask" in err_msg:
                ob_cancel_reason = "orderbook_empty"
                logger.error(f"Failed to compute maker price: {e}")
            elif "sell_guard" in err_msg:
                ob_cancel_reason = "sell_guard_reject"
                logger.warning(f"Maker price rejected: {e}")
            elif "spread too narrow" in err_msg:
                # 158# §20-D: spread_too_narrow を専用分類 — ERROR→INFO 降格
                ob_cancel_reason = "spread_too_narrow"
                logger.info(f"[158# §20-D] {e}")
            else:
                ob_cancel_reason = "orderbook_error"
                logger.error(f"Failed to compute maker price: {e}")
            # 155# §9.5 #3: orderbook_error 時に前回 mid_price をフォールバック
            # 156# §10 #5: 鮮度判定 — 閾値超は stale とみなす
            # 156# §16 review: _prev_mid_* 直接アクセス → 公開メソッド化
            _fb_price, _fb_time = self._maker_price.get_fallback_price()
            _fallback_price = _fb_price or 0.0
            _fallback_age: float | None = None
            _fallback_stale = False
            _stale_sec = self.config.fallback_stale_sec
            if _fallback_price > 0 and _fb_time is not None:
                _fallback_age = time.time() - _fb_time
                _fallback_stale = _fallback_age > _stale_sec
                logger.info(
                    f"[155# ob_fallback] Using last mid_price={_fallback_price:.0f} "
                    f"age={_fallback_age:.1f}s stale={_fallback_stale} "
                    f"as reference for skip record"
                )
            elif _fallback_price > 0:
                # no timestamp → stale 扱い (安全策)
                _fallback_stale = True
                logger.info(
                    f"[155# ob_fallback] Using last mid_price={_fallback_price:.0f} "
                    f"(no timestamp, treated as stale) as reference for skip record"
                )
            return self._make_skip_record(
                side=side,
                cancel_reason=ob_cancel_reason,
                cycle_id=cycle_id,
                order_price=_fallback_price if not _fallback_stale else 0.0,
                spread_offset_ratio=self._maker_price.base_offset_ratio,
                error_message=(
                    f"{e} [fallback_age={_fallback_age:.1f}s stale={_fallback_stale}]"
                    if _fallback_age is not None else str(e)
                ),
                regime=self._current_regime_value(),  # 160#
            )

        # 113# R1: SkipGate 判定を _evaluate_skip_gate() に委譲
        # 158# P1-1: balance_forced rescue → offset 倍増で安全にポジション解消
        if balance_forced_rescue and effective_offset_ratio > 0:
            _rescue_mult = self.config.balance_forced_rescue_offset_mult
            _pre_rescue_offset = effective_offset_ratio
            effective_offset_ratio = min(
                effective_offset_ratio * _rescue_mult,
                self.config.max_offset_ratio,
            )
            # 価格を rescue offset で再計算
            if spread_at_order is not None and spread_at_order > 0:
                mid_est = order_price + (spread_at_order * _pre_rescue_offset / 2 if side == "buy"
                                         else -spread_at_order * _pre_rescue_offset / 2)
                if side == "buy":
                    order_price = mid_est - spread_at_order * effective_offset_ratio / 2
                else:
                    order_price = mid_est + spread_at_order * effective_offset_ratio / 2
                order_price = round(order_price)
            logger.info(
                f"[158# P1-1] balance_forced_rescue: offset "
                f"{_pre_rescue_offset:.4f}→{effective_offset_ratio:.4f} "
                f"(×{_rescue_mult:.1f}), price={order_price:.0f}"
            )

        # 137# P1-08: spread 狭小時の「休む」判定
        if (
            self.config.narrow_spread_pause_enabled
            and spread_at_order is not None
            and order_price > 0
        ):
            mid_est = order_price  # 近似: maker price ≈ mid
            spread_bps_val = spread_at_order / mid_est * self._BPS_FACTOR if mid_est > 0 else 0.0
            if spread_bps_val < self.config.narrow_spread_pause_bps:
                self._narrow_spread_consecutive += 1
                if self._narrow_spread_consecutive <= self.config.narrow_spread_pause_max_consecutive:
                    pause_sec = self.config.narrow_spread_pause_sec
                    logger.info(
                        f"[137# P1-08] Spread too narrow ({spread_bps_val:.1f}bps "
                        f"< {self.config.narrow_spread_pause_bps}bps). "
                        f"Pausing {pause_sec}s "
                        f"({self._narrow_spread_consecutive}/{self.config.narrow_spread_pause_max_consecutive})"
                    )
                    # 139# §9-#3: 実際に待機してから FillRecord を返す
                    await asyncio.sleep(pause_sec)
                    return self._make_skip_record(
                        side=side,
                        cancel_reason=CR.NARROW_SPREAD_PAUSE,
                        cycle_id=cycle_id,
                        order_price=order_price,
                        spread_at_order=spread_at_order,
                        spread_offset_ratio=effective_offset_ratio,
                        regime=self._current_regime_value(),  # 160#
                    )
            else:
                self._narrow_spread_consecutive = 0

        # 151# §10 #4: regime_lot を1回だけ算出し、SkipGate/発注/記録へ共通引き回し
        _regime_lot = self._regime_adjusted_lot()

        sg = await self._evaluate_skip_gate(
            side, cycle_id, order_price, spread_at_order, effective_offset_ratio,
            order_lot=_regime_lot,
        )
        skip_gate_skipped = sg.skipped
        skip_gate_score = sg.score
        skip_gate_reason = sg.reason
        skip_gate_model_used = sg.model_used
        skip_gate_as_prob = sg.as_prob
        skip_gate_threshold_used = sg.threshold_used
        skip_gate_hour_offset = sg.hour_offset if sg.hour_offset != 0.0 else None
        _sg_velocity_60s = sg.price_velocity_60s  # 165# AS-R1
        if sg.early_return_record is not None:
            return sg.early_return_record

        # 2. 発注 (CM-2: リトライ付き)
        t_submit = time.time()
        order = None
        last_error: Optional[str] = None
        cancel_reason: str = "unknown"  # 032# #6: ループ未実行時の NameError 防止

        # 105#: lot floor guard — 121# BalanceChecker に委譲
        self._balance_checker.apply_lot_floor()

        # 143# R-1b + 151# P3-03: regime × confidence (per-cycle, _current_lot には永続化しない)
        # 145# fix: §8-#2 乗法的複利と §8-#3 片側更新を修正
        # §10 #5: dust_sweep 時は confidence_factor=1.0
        _order_lot, _confidence_factor = self._effective_order_lot(
            _regime_lot,
            as_prob=skip_gate_as_prob,
            dust_sweep_active=self._balance_checker.dust_sweep_active,
        )

        for attempt in range(1 + self.config.max_order_retries):
            try:
                # 130# E1: postonly 二重確認 — 発注直前に mid price を再取得し
                # テイカー側になっていないか確認 (postonly_reject 低減)
                try:
                    _pre_ob = await self.adapter.get_orderbook(self.config.symbol, depth=1)
                    if _pre_ob and _pre_ob.bids and _pre_ob.asks:
                        from scripts.v460.lib.ob_utils import best_bid_ask
                        _pre_best_bid, _pre_best_ask = best_bid_ask(_pre_ob)
                        # buy の指値が best_ask 以上 → テイカー側
                        if side == "buy" and order_price >= _pre_best_ask:
                            _safe_price = _pre_best_bid
                            logger.info(
                                f"[postonly_guard] 130# buy price {order_price:.0f} >= best_ask "
                                f"{_pre_best_ask:.0f}, adjusted to best_bid {_safe_price:.0f}"
                            )
                            order_price = _safe_price
                        # sell の指値が best_bid 以下 → テイカー側
                        elif side == "sell" and order_price <= _pre_best_bid:
                            _safe_price = _pre_best_ask
                            logger.info(
                                f"[postonly_guard] 130# sell price {order_price:.0f} <= best_bid "
                                f"{_pre_best_bid:.0f}, adjusted to best_ask {_safe_price:.0f}"
                            )
                            order_price = _safe_price
                except Exception as _pre_e:
                    logger.debug(f"[postonly_guard] Pre-check failed (non-fatal): {_pre_e}")

                order = await self.adapter.place_order(
                    symbol=self.config.symbol,
                    side=side,
                    quantity=_order_lot,
                    price=order_price,
                    order_type="limit",
                )
                self._pending_order_id = order.order_id
                logger.info(
                    f"Placed {side} limit @ {order_price:.0f} JPY, "
                    f"qty={_order_lot}, id={order.order_id}"
                    + (f" (retry {attempt})" if attempt > 0 else "")
                )
                break
            except Exception as e:
                last_error = str(e)
                # CM-2: エラー分類
                err_lower = last_error.lower()
                if "post_only" in err_lower or "taker" in err_lower:
                    cancel_reason = "post_only_reject"
                elif (
                    "insufficient" in err_lower
                    or "balance" in err_lower
                    # 042# Coincheck の日本語エラーメッセージ対応 — 121# YAML 外部化
                    or any(p in last_error for p in self.config.insufficient_funds_patterns)
                ):
                    cancel_reason = "insufficient_funds"
                elif "minimum" in err_lower or "size" in err_lower:
                    cancel_reason = "minimum_size"
                else:
                    cancel_reason = "api_error"

                logger.warning(
                    f"Order attempt {attempt + 1} failed ({cancel_reason}): {e}"
                )

                # 046# Bug10: 残高不足はリトライ不要 (2s 待っても残高は回復しない)
                # 084# post_only_reject もリトライ不要 (価格がスプレッド交差済み)
                _non_retriable = {"insufficient_funds", "post_only_reject", "minimum_size"}
                if cancel_reason in _non_retriable:
                    logger.info(
                        f"[Bug10] Skipping retry — {cancel_reason} is not retriable"
                    )
                    break

                if attempt < self.config.max_order_retries:
                    # 084# 指数バックオフ: 2s → 4s → 8s (rate-limit 緩和) — 121# YAML 外部化
                    _backoff = self.config.retry_delay_sec * (self.config.retry_backoff_base ** attempt)
                    # rate-limit 検出時はさらに延長
                    if "rate" in err_lower or "limit" in err_lower or "too many" in err_lower:
                        _backoff = max(_backoff, self.config.rate_limit_min_backoff_sec)
                        logger.warning(f"Rate-limit detected, extended backoff: {_backoff:.1f}s")
                    else:
                        logger.info(f"Retry backoff: {_backoff:.1f}s")
                    await asyncio.sleep(_backoff)
                    try:
                        ob = await self.adapter.get_orderbook(self.config.symbol, depth=1)
                        if ob.bids and ob.asks:
                            # 保守的価格: best_bid/best_ask そのまま (確実に maker)
                            order_price = ob.bids[0][0] if side == "buy" else ob.asks[0][0]
                            logger.info(f"Retry with conservative price: {order_price:.0f}")
                    except Exception:
                        pass  # 板取得失敗時は前回価格でリトライ

        if order is None:
            logger.error(f"All order attempts failed: {last_error}")
            # 113# resilience: API 失敗を CircuitBreaker に記録
            await self._circuit_breaker.async_on_failure()
            return FillRecord(
                cycle_id=cycle_id,
                timestamp=t_submit,
                side=side,
                order_price=order_price,
                order_quantity=_order_lot,
                cancelled=True,
                cancel_reason=cancel_reason,
                error_message=last_error,  # 031# エラー詳細を記録
                spread_at_order=spread_at_order,
                spread_offset_ratio=effective_offset_ratio,  # 096# 計算済み実効値
                run_id=self._run_id,       # 088# データ品質: エラー時も必須
                git_sha=self._git_sha,     # 088# quarantine 防止
                regime=self._current_regime_value(),  # 160#
            )

        # 113# R1: ポーリング監視 + 未約定キャンセルを _monitor_fill_polling() に委譲
        monitor = await self._monitor_fill_polling(
            order, order_price, side, t_submit, spread_at_order, effective_offset_ratio,
            order_lot=_order_lot,
        )
        filled = monitor.filled
        fill_price = monitor.fill_price
        queue_wait = monitor.queue_wait
        cancel_reason_poll = monitor.cancel_reason
        reprice_count = monitor.reprice_count
        reprice_drift_bps = monitor.reprice_drift_bps  # 158# P1-3
        order_price = monitor.final_order_price  # stale reprice で変更される場合
        _effective_timeout = monitor.effective_timeout  # 145# §9-#2

        # 113# R1: PnL 計測を _measure_post_fill_pnl() に委譲
        pnl = await self._measure_post_fill_pnl(filled, fill_price, side)
        mid_at_fill = pnl.mid_at_fill
        mid_30s_after = pnl.mid_30s_after
        mid_60s_after = pnl.mid_60s_after
        mid_120s_after = pnl.mid_120s_after
        post_fill_pnl = pnl.post_fill_pnl
        post_fill_60s_pnl = pnl.post_fill_60s_pnl
        post_fill_120s_pnl = pnl.post_fill_120s_pnl
        adverse_selected = pnl.adverse_selected
        adverse_selected_raw = pnl.adverse_selected_raw
        actual_measurement_sec = pnl.actual_measurement_sec

        # 162# Inventory Skewing: fill 成功時に在庫偏重を更新
        if filled:
            self._maker_price.update_inventory(side)

        # 037# レジーム検知更新 (035# §7 Week 1)
        regime_str: Optional[str] = None
        regime_conf: Optional[float] = None
        regime_stab: Optional[int] = None
        # 156# §18: データシンク解消 — trend_pct/volatility_ratio を FillRecord へ
        regime_trend_pct: Optional[float] = None
        regime_vol_ratio: Optional[float] = None
        if self._regime_detector is not None:
            # 100# P1-6 fix: unfilled 時は order_price (offset 込み) ではなく
            # 直近の真の mid price を使用。order_price は offset を含むため
            # regime 検知のノイズ源となる。
            if mid_at_fill is not None:
                regime_price = mid_at_fill
            else:
                _fb_price, _ = self._maker_price.get_fallback_price()
                regime_price = _fb_price  # None if unavailable

            if regime_price is not None:
                regime_result = self._regime_detector.update(t_submit, regime_price)
                regime_str = regime_result.regime.value
                regime_conf = regime_result.confidence
                regime_stab = regime_result.stability
                regime_trend_pct = regime_result.trend_pct
                regime_vol_ratio = regime_result.volatility_ratio

        record = FillRecord(
            cycle_id=cycle_id,
            timestamp=t_submit,
            side=side,
            order_price=order_price,
            order_quantity=_order_lot,
            fill_price=fill_price,
            filled=filled,
            cancelled=not filled,
            queue_wait_sec=queue_wait,
            mid_at_fill=mid_at_fill,
            mid_30s_after=mid_30s_after,
            mid_60s_after=mid_60s_after,
            mid_120s_after=mid_120s_after,
            post_fill_30s_pnl=post_fill_pnl,
            post_fill_60s_pnl=post_fill_60s_pnl,
            post_fill_120s_pnl=post_fill_120s_pnl,
            adverse_selected=adverse_selected,
            adverse_selected_raw=adverse_selected_raw,
            cancel_reason=(
                cancel_reason_poll
                if cancel_reason_poll
                else (
                    "timeout"
                    # 145# §9-#2: regime 調整済みの effective_timeout を使用
                    if (not filled and queue_wait >= (_effective_timeout or self.config.order_timeout_sec))
                    else ("unknown" if not filled else None)  # 117# C-fix: None 防止
                )
            ),
            run_id=self._run_id,
            git_sha=self._git_sha,
            # 031# 追加フィールド
            spread_at_order=spread_at_order,
            spread_offset_ratio=effective_offset_ratio,  # 050# Bug#3 fix: 実効値を記録
            # 037# レジーム情報
            regime=regime_str,
            regime_confidence=regime_conf,
            regime_stability=regime_stab,
            # 156# §18: データシンク解消
            regime_trend_pct=regime_trend_pct,
            regime_volatility_ratio=regime_vol_ratio,
            # 054# S5: AS 予測データ基盤
            # 122# R5/§7.3 方法 2: OB 記録を imbalance_enabled と独立させ常時記録
            orderbook_imbalance=self._maker_price._last_imbalance,
            bid_depth_total=self._maker_price._last_bid_depth,
            ask_depth_total=self._maker_price._last_ask_depth,
            mid_price_trend_5s=self._maker_price._last_mid_trend_bps,
            spread_bps=(
                (spread_at_order / mid_at_fill * self._BPS_FACTOR)
                if spread_at_order is not None and mid_at_fill is not None and mid_at_fill > 0
                else None
            ),
            effective_offset_used=effective_offset_ratio,
            # 062# SkipGate 判定情報 (PASS 時も記録 → 後続分析用)
            skip_gate_skipped=skip_gate_skipped,
            skip_gate_score=skip_gate_score,
            skip_gate_reason=skip_gate_reason,
            skip_gate_model_used=skip_gate_model_used,
            # 084# P(AS) 可観測性改善
            skip_gate_as_prob=skip_gate_as_prob,
            skip_gate_threshold_used=skip_gate_threshold_used,
            # 158# P1-6: 時間帯別 skip_gate 閾値調整のオフセット
            skip_gate_hour_offset=skip_gate_hour_offset,
            # 094# stale order cancel-replace 追跡
            reprice_count=reprice_count,
            # 158# P1-3: reprice 累積 drift (bps)
            reprice_drift_bps=reprice_drift_bps if reprice_count > 0 else None,
            # 100# P1-4: 実際の PnL 計測経過秒数
            actual_measurement_sec=actual_measurement_sec if filled else None,
            # 120# A4: Early Exit 明示フラグ
            early_exit_triggered=pnl.early_exit_triggered if filled else None,
            # 120# A4-2: EE 中断時点 PnL (計測バイアス分離)
            pnl_at_exit_bps=pnl.pnl_at_exit_bps if filled else None,
            # 120# P2-1: 寄与分解基盤 — FFD/VG イベントフラグ
            ffd_boost_active=self._fast_fill_defense.is_boost_active(side),
            vg_triggered=self._maker_price.last_vg_triggered,
            # 158# P2-6: VG 詳細ログ (ヒンドサイト分析用)
            vg_velocity_bps=self._maker_price.last_vg_velocity_bps,
            vg_vpin=self._maker_price.last_vg_vpin,
            vg_boost_factor=self._maker_price.last_vg_boost_factor,
            # 165# AS-R1: velocity logging
            price_velocity_60s=_sg_velocity_60s,
            # 129# D.2: 残高制約による side 強制切替フラグ
            balance_forced_switch=balance_forced_switch or None,
            # 151# P3-03: confidence lot 可観測性 (§10 #7)
            confidence_lot_factor=_confidence_factor if self.config.enable_confidence_lot else None,
            order_lot_regime=_regime_lot,
            order_lot_effective=_order_lot,
            confidence_lot_mode=self.config.confidence_lot_mode if self.config.enable_confidence_lot else None,
            # 158# P1-5: A/B テスト variant 識別子
            ab_test_variant=self.config.ab_test_variant or None,
        )

        logger.info(
            f"Cycle {self._cycle_count} result: "
            f"filled={filled}, wait={queue_wait:.1f}s, "
            f"pnl={post_fill_pnl:.2f}bps" if post_fill_pnl is not None
            else f"Cycle {self._cycle_count} result: filled={filled}, wait={queue_wait:.1f}s"
        )

        # 113# resilience: API 成功を CircuitBreaker に記録
        await self._circuit_breaker.async_on_success()

        return record
