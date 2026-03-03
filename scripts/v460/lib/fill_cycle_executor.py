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
from typing import TYPE_CHECKING, Optional, cast

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.fill_config import (
    SkipGateResult as _SkipGateResult,
    FillMonitorResult as _FillMonitorResult,
    PnlMeasurement as _PnlMeasurement,
)
from scripts.v460.lib.ob_utils import best_bid_ask  # 200# 10-C: module-level import

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig
    from scripts.v460.lib.order_monitor import OrderLike
    from scripts.v460.lib.resilience import CircuitState
    from ztb.metrics.fill_quality import FillRecord

logger = logging.getLogger(__name__)


class FillCycleExecutorMixin:
    """run_single_cycle + OB/SkipGate/Fill/PnL ヘルパー (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: 1 取引サイクル実行, OB ラッパー, SkipGate, Fill 監視, PnL 計測
      NG: ループ制御, side kill, time filter, balance forced
    MAX LINES: 750 (超えたら _build_fill_record を別モジュールに分離せよ)
    ────────────────────────────────────────────────────
    188# _build_fill_record() 抽出済み
    """

    # 201# review: 動的属性のクラスレベル宣言 (mypy 検出 + IDE 補完)
    _postonly_crossing_streak: int = 0
    # 234# no_feasible_quote 連続カウンタ (制約集合崩壊検出用)
    # 236# per-side 化: buy/sell 交互実行で相互リセットされる問題を修正
    _consecutive_no_feasible: dict[str, int] | None = None
    # 236# hasattr 排除: __init__ 前でも安全なクラスレベルデフォルト
    _current_regime_value: object = None  # type: ignore[assignment]
    # 237# PhantomPositionGuard: クラスレベルデフォルト (hasattr 排除)
    _phantom_guard: object | None = None

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

    def _make_cycle_skip_record(
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
        """run_single_cycle 系 skip record の共通 wrapper."""
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

    def _maybe_register_phantom(
        self,
        monitor: _FillMonitorResult,
        side: str,
        order_lot: float,
        order_price: float,
    ) -> bool:
        """237# status_unknown 時に PhantomPositionGuard へ登録.

        Returns:
            True = pending_reconciliation (FillRecord に設定する)
        """
        if (
            monitor.filled
            or monitor.cancel_reason is None
            or not monitor.cancel_reason.startswith("status_unknown")
            or monitor.order_id_for_reconciliation is None
        ):
            return False
        if self._phantom_guard is not None:
            self._phantom_guard.register_unknown(
                order_id=monitor.order_id_for_reconciliation,
                side=side,
                quantity=order_lot,
                price=order_price,
            )
        return True

    def _resolve_fill_cancel_reason(
        self,
        *,
        filled: bool,
        queue_wait: float,
        cancel_reason_poll: str | None,
        effective_timeout: float | None,
    ) -> str | None:
        """約定結果に応じた cancel_reason を一元解決."""
        if cancel_reason_poll:
            return cancel_reason_poll
        if filled:
            return None
        timeout_limit = effective_timeout or self.config.order_timeout_sec
        return "timeout" if queue_wait >= timeout_limit else "unknown"

    def _compute_fill_spread_bps(
        self,
        *,
        spread_at_order: float | None,
        mid_at_fill: float | None,
    ) -> float | None:
        """FillRecord 用 spread_bps を安全に算出."""
        if spread_at_order is None or mid_at_fill is None or mid_at_fill <= 0:
            return None
        return spread_at_order / mid_at_fill * self._BPS_FACTOR

    def _build_fill_measurement_fields(
        self,
        *,
        fill_price: float | None,
        filled: bool,
        queue_wait: float,
        cancel_reason_poll: str | None,
        effective_timeout: float | None,
        pnl: _PnlMeasurement,
    ) -> dict[str, object]:
        """FillRecord の約定/計測系フィールドを構築."""
        return {
            "fill_price": fill_price,
            "filled": filled,
            "cancelled": not filled,
            "queue_wait_sec": queue_wait,
            "mid_at_fill": pnl.mid_at_fill,
            "mid_30s_after": pnl.mid_30s_after,
            "mid_60s_after": pnl.mid_60s_after,
            "mid_120s_after": pnl.mid_120s_after,
            "post_fill_30s_pnl": pnl.post_fill_pnl,
            "post_fill_60s_pnl": pnl.post_fill_60s_pnl,
            "post_fill_120s_pnl": pnl.post_fill_120s_pnl,
            "adverse_selected": pnl.adverse_selected,
            "adverse_selected_raw": pnl.adverse_selected_raw,
            "cancel_reason": self._resolve_fill_cancel_reason(
                filled=filled,
                queue_wait=queue_wait,
                cancel_reason_poll=cancel_reason_poll,
                effective_timeout=effective_timeout,
            ),
            "actual_measurement_sec": pnl.actual_measurement_sec if filled else None,
            "early_exit_triggered": pnl.early_exit_triggered if filled else None,
            "pnl_at_exit_bps": pnl.pnl_at_exit_bps if filled else None,
        }

    def _build_fill_market_fields(
        self,
        *,
        side: str,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        reprice_count: int,
        reprice_drift_bps: float | None,
        sg_skipped: bool,
        sg_score: float,
        sg_reason: str,
        sg_model_used: str,
        sg_as_prob: float | None,
        sg_threshold_used: float | None,
        sg_hour_offset: float | None,
        sg_velocity_bps: float | None,
        regime_str: str | None,
        regime_conf: float | None,
        regime_stab: int | None,
        regime_trend_pct: float | None,
        regime_vol_ratio: float | None,
        balance_forced_switch: bool,
        confidence_factor: float,
        regime_lot: float,
        order_lot: float,
        cancel_failed_likely_filled: bool,
        mid_at_fill: float | None,
    ) -> dict[str, object]:
        """FillRecord の市場観測/実行メタ系フィールドを構築."""
        return {
            "spread_at_order": spread_at_order,
            "spread_offset_ratio": effective_offset_ratio,
            "regime": regime_str,
            "regime_confidence": regime_conf,
            "regime_stability": regime_stab,
            "regime_trend_pct": regime_trend_pct,
            "regime_volatility_ratio": regime_vol_ratio,
            "orderbook_imbalance": self._maker_price._last_imbalance,
            "bid_depth_total": self._maker_price._last_bid_depth,
            "ask_depth_total": self._maker_price._last_ask_depth,
            "mid_price_trend_5s": self._maker_price._last_mid_trend_bps,
            "spread_bps": self._compute_fill_spread_bps(
                spread_at_order=spread_at_order,
                mid_at_fill=mid_at_fill,
            ),
            "effective_offset_used": effective_offset_ratio,
            "skip_gate_skipped": sg_skipped,
            "skip_gate_score": sg_score,
            "skip_gate_reason": sg_reason,
            "skip_gate_model_used": sg_model_used,
            "skip_gate_as_prob": sg_as_prob,
            "skip_gate_threshold_used": sg_threshold_used,
            "skip_gate_hour_offset": sg_hour_offset,
            "reprice_count": reprice_count,
            "reprice_drift_bps": reprice_drift_bps if reprice_count > 0 else None,
            "ffd_boost_active": self._fast_fill_defense.is_boost_active(side),
            "vg_triggered": self._maker_price.last_vg_triggered,
            "vg_velocity_bps": self._maker_price.last_vg_velocity_bps,
            "vg_vpin": self._maker_price.last_vg_vpin,
            "vg_boost_factor": self._maker_price.last_vg_boost_factor,
            "price_velocity_bps": sg_velocity_bps,
            "balance_forced_switch": balance_forced_switch or None,
            "confidence_lot_factor": (
                confidence_factor if self.config.enable_confidence_lot else None
            ),
            "order_lot_regime": regime_lot,
            "order_lot_effective": order_lot,
            "confidence_lot_mode": (
                self.config.confidence_lot_mode if self.config.enable_confidence_lot else None
            ),
            "ab_test_variant": self.config.ab_test_variant or None,
            "cancel_failed_likely_filled": cancel_failed_likely_filled or None,
        }

    def _build_fill_strategy_fields(
        self,
        *,
        post_fill_pnl: float | None,
        post_fill_120s_pnl: float | None,
        regime_str: str | None,
        regime_conf: float | None,
        macro_trend: str | None,
        macro_slope_5m: float | None,
        macro_slope_15m: float | None,
        macro_aligned: bool | None,
    ) -> dict[str, object]:
        """FillRecord の strategy/macro 系フィールドを構築."""
        ev_weighted = self._compute_ev_weighted(
            post_fill_pnl,
            post_fill_120s_pnl,
            w30=self._cycle_strategy.policy.ev_weighted_w30,
            w120=self._cycle_strategy.policy.ev_weighted_w120,
        ) if self._cycle_strategy is not None else self._compute_ev_weighted(
            post_fill_pnl,
            post_fill_120s_pnl,
        )
        return {
            "ev_weighted_pnl": ev_weighted,
            "gated_regime": (
                self._cycle_strategy.gated_regime(regime_str, regime_conf)
                if self._cycle_strategy is not None and regime_str is not None
                else None
            ),
            "effective_cycle_interval": (
                self._cycle_strategy.effective_interval(regime_str)
                if self._cycle_strategy is not None
                else None
            ),
            "macro_trend": macro_trend,
            "macro_slope_5m": macro_slope_5m,
            "macro_slope_15m": macro_slope_15m,
            "macro_aligned": macro_aligned,
        }

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
        one_sided_balance: bool = False,
    ) -> _SkipGateResult:
        """SkipGate ML 判定 — 121# SkipGateEvaluator に委譲.

        145# §9-#4: order_lot を渡してレジーム倍率適用後のロットで記録.
        190# B: one_sided_balance を渡して片側 balance 時の threshold 緩和.
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
            one_sided_balance=one_sided_balance,
        )

    async def _monitor_fill_polling(
        self,
        order: OrderLike,
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

        # 179# Chase: CycleStrategy から regime 別 chase パラメータを取得
        _chase_drift: float | None = None
        _chase_max_rp: int | None = None
        if self._cycle_strategy is not None:
            # 236# hasattr 排除: _current_regime_value は fill_record_helpers Mixin で定義済み
            _regime = self._current_regime_value()
            if self._cycle_strategy.is_chase_enabled(_regime, side):  # 187# B-1: side 方向制限
                _chase_drift = self._cycle_strategy.chase_drift_bps()
                _chase_max_rp = self._cycle_strategy.chase_max_reprice()

        return await self._order_monitor.monitor(
            adapter=self.adapter,
            order=order,
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
            chase_drift_bps_override=_chase_drift,          # 179# Chase
            chase_max_reprice_override=_chase_max_rp,       # 179# Chase
        )

    # ------------------------------------------------------------------
    # 181# EV_weighted: pnl30/pnl120 加重平均 (178# §1.3)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_ev_weighted(
        pnl30: float | None,
        pnl120: float | None,
        *,
        w30: float = 0.4,
        w120: float = 0.6,
    ) -> float | None:
        """30s/120s PnL の加重平均を計算.

        pnl120 が None (E3 サンプリング外) の場合は pnl30 単独値を返す。
        """
        if pnl30 is None:
            return None
        if pnl120 is None:
            return pnl30  # 120s 未計測時は 30s 単独
        return w30 * pnl30 + w120 * pnl120

    @staticmethod
    def _apply_offset_multiplier(
        *,
        side: str,
        order_price: float,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        offset_mult: float | None,
        aggressive_when_multiplier_gt_one: bool = False,
    ) -> tuple[float, float, float | None, float | None]:
        """offset 倍率を安全に適用し、更新後の価格・倍率を返す.

        `aggressive_when_multiplier_gt_one=False`:
          multiplier>1.0 で mid から遠ざける (195/196 の保守的発注)
        `aggressive_when_multiplier_gt_one=True`:
          multiplier>1.0 で mid に近づける (193 の EV 前向き調整)
        """
        if (
            offset_mult is None
            or offset_mult <= 0.0
            or spread_at_order is None
            or spread_at_order <= 0
            or order_price <= 0
        ):
            return order_price, effective_offset_ratio, None, None
        if offset_mult == 1.0:
            return order_price, effective_offset_ratio, None, None
        if not aggressive_when_multiplier_gt_one and offset_mult < 1.0:
            return order_price, effective_offset_ratio, None, None

        old_offset = spread_at_order * effective_offset_ratio
        new_offset = old_offset * offset_mult
        delta = new_offset - old_offset
        if aggressive_when_multiplier_gt_one:
            if side == "buy":
                order_price = round(order_price + delta)
            else:
                order_price = round(order_price - delta)
        else:
            if side == "buy":
                order_price = round(order_price - delta)
            else:
                order_price = round(order_price + delta)
        return order_price, effective_offset_ratio * offset_mult, offset_mult, delta

    # ------------------------------------------------------------------
    # 188# FillRecord 構築 (run_single_cycle からの抽出)
    # ------------------------------------------------------------------
    def _build_fill_record(
        self,
        *,
        cycle_id: str,
        t_submit: float,
        side: str,
        order_price: float,
        order_lot: float,
        fill_price: float | None,
        filled: bool,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        queue_wait: float,
        cancel_reason_poll: str | None,
        reprice_count: int,
        reprice_drift_bps: float | None,
        effective_timeout: float | None,
        cancel_failed_likely_filled: bool,
        pnl: _PnlMeasurement,
        sg_skipped: bool,
        sg_score: float,
        sg_reason: str,
        sg_model_used: str,
        sg_as_prob: float | None,
        sg_threshold_used: float | None,
        sg_hour_offset: float | None,
        sg_velocity_bps: float | None,
        regime_str: str | None,
        regime_conf: float | None,
        regime_stab: int | None,
        regime_trend_pct: float | None,
        regime_vol_ratio: float | None,
        balance_forced_switch: bool,
        confidence_factor: float,
        regime_lot: float,
        macro_trend: str | None = None,
        macro_slope_5m: float | None = None,
        macro_slope_15m: float | None = None,
        macro_aligned: bool | None = None,
    ) -> FillRecord:
        """188# FillRecord を組み立てる.

        run_single_cycle の末尾から抽出。self 経由のセンサー値 +
        サイクル変数を統合して 1 レコードを構築する。
        """
        from ztb.metrics.fill_quality import build_fill_record

        payload: dict[str, object] = {
            "cycle_id": cycle_id,
            "timestamp": t_submit,
            "side": side,
            "order_price": order_price,
            "order_quantity": order_lot,
            "run_id": self._run_id,
            "git_sha": self._git_sha,
        }
        payload.update(
            self._build_fill_measurement_fields(
                fill_price=fill_price,
                filled=filled,
                queue_wait=queue_wait,
                cancel_reason_poll=cancel_reason_poll,
                effective_timeout=effective_timeout,
                pnl=pnl,
            )
        )
        payload.update(
            self._build_fill_market_fields(
                side=side,
                spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio,
                reprice_count=reprice_count,
                reprice_drift_bps=reprice_drift_bps,
                sg_skipped=sg_skipped,
                sg_score=sg_score,
                sg_reason=sg_reason,
                sg_model_used=sg_model_used,
                sg_as_prob=sg_as_prob,
                sg_threshold_used=sg_threshold_used,
                sg_hour_offset=sg_hour_offset,
                sg_velocity_bps=sg_velocity_bps,
                regime_str=regime_str,
                regime_conf=regime_conf,
                regime_stab=regime_stab,
                regime_trend_pct=regime_trend_pct,
                regime_vol_ratio=regime_vol_ratio,
                balance_forced_switch=balance_forced_switch,
                confidence_factor=confidence_factor,
                regime_lot=regime_lot,
                order_lot=order_lot,
                cancel_failed_likely_filled=cancel_failed_likely_filled,
                mid_at_fill=pnl.mid_at_fill,
            )
        )
        payload.update(
            self._build_fill_strategy_fields(
                post_fill_pnl=pnl.post_fill_pnl,
                post_fill_120s_pnl=pnl.post_fill_120s_pnl,
                regime_str=regime_str,
                regime_conf=regime_conf,
                macro_trend=macro_trend,
                macro_slope_5m=macro_slope_5m,
                macro_slope_15m=macro_slope_15m,
                macro_aligned=macro_aligned,
            )
        )
        return build_fill_record(**payload)

    async def _measure_post_fill_pnl(
        self,
        filled: bool,
        fill_price: Optional[float],
        side: str,
    ) -> _PnlMeasurement:
        """約定後 PnL 計測 — 120# PnlMeasurer に委譲."""
        # 179# D: regime 別 post-fill wait
        _wait_override: float | None = None
        if self._cycle_strategy is not None:
            # 236# hasattr 排除: _current_regime_value は fill_record_helpers Mixin で定義済み
            _regime = self._current_regime_value()
            # 200# G: vol_ratio 動的 wait スケーリング
            _vol_ratio: float | None = None
            if (
                self._regime_detector is not None
            ):
                _vol_ratio = self._regime_detector.last_volatility_ratio
            _wait_override = self._cycle_strategy.effective_post_fill_wait(
                side, _regime, vol_ratio=_vol_ratio,
            )
        pnl = await self._pnl_measurer.measure(
            filled=filled,
            fill_price=fill_price,
            side=side,
            get_mid_price=self._get_mid_price,
            wait_sec_override=_wait_override,  # 179# D
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
        one_sided_balance: bool = False,
        trending_offset_mult: float | None = None,
        degraded_liquidation: bool = False,
    ) -> FillRecord:
        """1 サイクル: 発注 → 監視 → 結果記録.

        009# §4.2 の流れに準拠.
        041# 時間帯フィルター・残高チェック追加.
        055# Fix: side 決定前に最新 imbalance を取得.
        075# Fix: side_override で run_continuous() が決定した side を強制適用.
        129# D.2: balance_forced_switch フラグを FillRecord に記録.
        158# P1-1: balance_forced_rescue — offset 倍増で安全にポジション解消.
        190# B: one_sided_balance — 片側残高時の ev_weighted threshold 緩和.
        234# degraded_liquidation — Kill Gate blocked + balance_forced 時の縮退清算.
        """
        self._cycle_count += 1
        cycle_id = self._new_cycle_id()

        # 113# resilience: CircuitBreaker ガード — OPEN 中は API 呼出しを回避
        from scripts.v460.lib.resilience import CircuitState

        if self._circuit_breaker.state == CircuitState.OPEN:
            if not self._circuit_breaker.should_attempt_reset():
                logger.warning(
                    f"[circuit_breaker] OPEN — skipping cycle {self._cycle_count} "
                    f"(recovery in {self._circuit_breaker.config.recovery_timeout}s)"
                )
                return self._make_cycle_skip_record(
                    side=side_override or "buy",
                    cancel_reason=CR.CIRCUIT_BREAKER_OPEN,
                    cycle_id=cycle_id,
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
            # 234# no_feasible_quote 連続カウンタリセット (成功時)
            # 236# per-side 化
            if self._consecutive_no_feasible is None:
                self._consecutive_no_feasible = {}
            self._consecutive_no_feasible[side] = 0
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
                # 234# sell_guard + spread_too_narrow が同時発生→ 制約集合崩壊
                ob_cancel_reason = "sell_guard_reject"
                logger.warning(f"Maker price rejected: {e}")
            elif "spread too narrow" in err_msg:
                # 158# §20-D: spread_too_narrow を専用分類 — ERROR→INFO 降格
                ob_cancel_reason = "spread_too_narrow"
                logger.info(f"[158# §20-D] {e}")
            else:
                ob_cancel_reason = "orderbook_error"
                logger.error(f"Failed to compute maker price: {e}")

            # 234# no_feasible_quote 検出: spread 制約 (narrow/wide) で連続失敗
            # 236# per-side 化
            if ob_cancel_reason in ("spread_too_narrow", "sell_guard_reject"):
                if self._consecutive_no_feasible is None:
                    self._consecutive_no_feasible = {}
                _cnf = self._consecutive_no_feasible.get(side, 0) + 1
                self._consecutive_no_feasible[side] = _cnf
                if _cnf >= 3:
                    ob_cancel_reason = CR.NO_FEASIBLE_QUOTE
                    logger.warning(
                        f"[234#] NO_FEASIBLE_QUOTE: {_cnf} "
                        f"consecutive infeasible quotes ({side}) — constraint set collapse "
                        f"(min_spread={self.config.min_spread_jpy}, "
                        f"sell_max_spread={self.config.sell_max_spread_jpy})"
                    )
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
            return self._make_cycle_skip_record(
                side=side,
                cancel_reason=ob_cancel_reason,
                cycle_id=cycle_id,
                order_price=_fallback_price if not _fallback_stale else 0.0,
                spread_offset_ratio=self._maker_price.base_offset_ratio,
                error_message=(
                    f"{e} [fallback_age={_fallback_age:.1f}s stale={_fallback_stale}]"
                    if _fallback_age is not None else str(e)
                ),
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

        # 234# 縮退清算モード: Kill Gate blocked + balance_forced
        # min lot + wide offset で安全に在庫清算
        if degraded_liquidation and effective_offset_ratio > 0:
            _deg_offset_mult = self.config.degraded_liquidation_offset_mult
            _pre_deg_offset = effective_offset_ratio
            effective_offset_ratio = min(
                effective_offset_ratio * _deg_offset_mult,
                self.config.max_offset_ratio,
            )
            # 価格を degraded offset で再計算
            if spread_at_order is not None and spread_at_order > 0:
                mid_est = order_price + (spread_at_order * _pre_deg_offset / 2 if side == "buy"
                                         else -spread_at_order * _pre_deg_offset / 2)
                if side == "buy":
                    order_price = mid_est - spread_at_order * effective_offset_ratio / 2
                else:
                    order_price = mid_est + spread_at_order * effective_offset_ratio / 2
                order_price = round(order_price)
            else:
                # 235# C-3 guard: spread 不明時は offset 拡大のみ (価格再計算不可)
                logger.warning(
                    f"[235#] degraded_liquidation: spread unavailable — "
                    f"offset expanded but price NOT recalculated"
                )
            logger.warning(
                f"[234#] degraded_liquidation: offset "
                f"{_pre_deg_offset:.4f}→{effective_offset_ratio:.4f} "
                f"(×{_deg_offset_mult:.1f}), price={order_price:.0f}"
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
                    return self._make_cycle_skip_record(
                        side=side,
                        cancel_reason=CR.NARROW_SPREAD_PAUSE,
                        cycle_id=cycle_id,
                        order_price=order_price,
                        spread_at_order=spread_at_order,
                        spread_offset_ratio=effective_offset_ratio,
                    )
            else:
                self._narrow_spread_consecutive = 0

        # 151# §10 #4: regime_lot を1回だけ算出し、SkipGate/発注/記録へ共通引き回し
        _regime_lot = self._regime_adjusted_lot()

        # 234# 縮退清算モード: lot を大幅縮小
        if degraded_liquidation:
            _pre_deg_lot = _regime_lot
            _regime_lot = max(
                _regime_lot * self.config.degraded_liquidation_lot_mult,
                self.config.min_lot,
            )
            logger.warning(
                f"[234#] degraded_liquidation lot: "
                f"{_pre_deg_lot:.6f}→{_regime_lot:.6f} "
                f"(×{self.config.degraded_liquidation_lot_mult:.1f})"
            )

        sg = await self._evaluate_skip_gate(
            side, cycle_id, order_price, spread_at_order, effective_offset_ratio,
            order_lot=_regime_lot,
            one_sided_balance=one_sided_balance,
        )
        skip_gate_skipped = sg.skipped
        skip_gate_score = sg.score
        skip_gate_reason = sg.reason
        skip_gate_model_used = sg.model_used
        skip_gate_as_prob = sg.as_prob
        skip_gate_threshold_used = sg.threshold_used
        skip_gate_hour_offset = sg.hour_offset if sg.hour_offset != 0.0 else None
        _sg_velocity_bps = sg.price_velocity_bps  # 165# AS-R1
        if sg.early_return_record is not None:
            return sg.early_return_record

        # 193#: ev_weighted → offset 価格調整
        # SkipGate PASS 後に ev_score を使って order_price を post-hoc 調整
        # 200# M: DRY — compute_ev_offset_multiplier に共通化 + warning zone
        _ev_offset_applied = False
        if (
            sg.ev_score is not None
            and self.config.skip_gate_ev_as_offset_enabled
            and spread_at_order is not None
            and spread_at_order > 0
            and order_price > 0
        ):
            from scripts.v460.lib.fill_config import compute_ev_offset_multiplier
            _ev_s = sg.ev_score
            _ev_mult = compute_ev_offset_multiplier(
                ev_score=_ev_s,
                sensitivity=self.config.skip_gate_ev_offset_sensitivity,
                min_mult=self.config.skip_gate_ev_offset_min_mult,
                max_mult=self.config.skip_gate_ev_offset_max_mult,
                warning_threshold=self.config.skip_gate_ev_warning_threshold,
                warning_factor=self.config.skip_gate_ev_warning_offset_factor,
            )
            order_price, effective_offset_ratio, _applied_mult, _delta = self._apply_offset_multiplier(
                side=side,
                order_price=order_price,
                spread_at_order=spread_at_order,
                effective_offset_ratio=effective_offset_ratio,
                offset_mult=_ev_mult,
                aggressive_when_multiplier_gt_one=True,
            )
            if _applied_mult is not None and _delta is not None:
                _ev_offset_applied = True
                logger.info(
                    f"[193# ev_offset] {side}: ev_score={_ev_s:.3f} "
                    f"→ offset_mult={_applied_mult:.3f} "
                    f"(delta={_delta:+.0f}JPY, price={order_price:.0f})"
                )

        # 195#: velocity_skip ソフトモード — offset boost 適用
        # velocity が閾値を超えた場合、hard skip ではなく offset を拡大して保守的に発注
        _vel_offset_applied = False
        order_price, effective_offset_ratio, _vel_mult, _delta = self._apply_offset_multiplier(
            side=side,
            order_price=order_price,
            spread_at_order=spread_at_order,
            effective_offset_ratio=effective_offset_ratio,
            offset_mult=sg.velocity_offset_mult,
        )
        if _vel_mult is not None and _delta is not None:
            _vel_offset_applied = True
            logger.info(
                f"[195# vel_offset] {side}: velocity={sg.price_velocity_bps:.2f}bps "
                f"→ offset_mult={_vel_mult:.2f} "
                f"(delta={_delta:+.0f}JPY, price={order_price:.0f})"
            )

        # 196# trending_sell ソフトモード — offset boost 適用
        # trending regime での sell を skip せず、offset を拡大して保守的に発注
        order_price, effective_offset_ratio, _trend_mult, _delta = self._apply_offset_multiplier(
            side=side,
            order_price=order_price,
            spread_at_order=spread_at_order,
            effective_offset_ratio=effective_offset_ratio,
            offset_mult=trending_offset_mult if side == "sell" else None,
        )
        if _trend_mult is not None and _delta is not None:
            logger.info(
                f"[196# trend_offset] sell: trending regime "
                f"→ offset_mult={_trend_mult:.1f} "
                f"(delta={_delta:+.0f}JPY, price={order_price:.0f})"
            )

        # 202# C: VG sell-side 補完 — maker_price VG が未発火かつ velocity が高い sell で
        # 補足的 offset boost を適用。mid_trend_bps は point-to-point のため sell 側で
        # VG が盲点になるケースを velocity_bps で補完する。
        if (
            side == "sell"
            and not self._maker_price.last_vg_triggered
            and _sg_velocity_bps is not None
            and abs(_sg_velocity_bps) > self.config.volatility_guard_velocity_threshold_bps
            and not _vel_offset_applied  # 195# で既に補正済みなら二重適用しない
        ):
            _vg_supp_boost = self.config.volatility_guard_offset_boost_factor
            order_price, effective_offset_ratio, _vg_supp_mult, _vg_supp_delta = (
                self._apply_offset_multiplier(
                    side=side,
                    order_price=order_price,
                    spread_at_order=spread_at_order,
                    effective_offset_ratio=effective_offset_ratio,
                    offset_mult=_vg_supp_boost,
                )
            )
            if _vg_supp_mult is not None and _vg_supp_delta is not None:
                logger.info(
                    f"[202# C] VG sell supplement: velocity_bps="
                    f"{_sg_velocity_bps:.1f}bps → offset_mult={_vg_supp_mult:.2f} "
                    f"(delta={_vg_supp_delta:+.0f}JPY, price={order_price:.0f})"
                )

        # 215# P0-C: alert_mode offset 乗数 — 全サイド共通
        _alert_om = getattr(self, "_alert_offset_mult", 1.0)
        if _alert_om != 1.0:
            order_price, effective_offset_ratio, _a_mult, _a_delta = (
                self._apply_offset_multiplier(
                    side=side,
                    order_price=order_price,
                    spread_at_order=spread_at_order,
                    effective_offset_ratio=effective_offset_ratio,
                    offset_mult=_alert_om,
                )
            )
            if _a_mult is not None and _a_delta is not None:
                logger.warning(
                    f"[215# alert_mode] {side}: offset_mult={_a_mult:.2f} "
                    f"(delta={_a_delta:+.0f}JPY, price={order_price:.0f})"
                )

        # 2. 発注 (CM-2: リトライ付き)
        t_submit = time.time()
        order: object | None = None
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

        # 215# P0-C: alert_mode lot 乗数 — 縮小運転
        _alert_lm = getattr(self, "_alert_lot_mult", 1.0)
        if _alert_lm != 1.0:
            _pre_lot = _order_lot
            _order_lot = max(self.config.order_quantity, _order_lot * _alert_lm)
            logger.warning(
                f"[215# alert_mode] lot_mult={_alert_lm:.2f}: "
                f"{_pre_lot} → {_order_lot}"
            )

        # 224# B1: halt解除後ソフトリカバリ lot 縮小
        _recovery_lm = getattr(self, "_halt_recovery_lot_mult", 1.0)
        if _recovery_lm < 1.0:
            _pre_lot = _order_lot
            _order_lot = max(self.config.order_quantity, _order_lot * _recovery_lm)
            logger.info(
                f"[224# B1] Recovery lot_scale={_recovery_lm:.2f}: "
                f"{_pre_lot:.6f} → {_order_lot:.6f}"
            )

        for attempt in range(1 + self.config.max_order_retries):
            try:
                # 130# E1: postonly 二重確認 — 発注直前に mid price を再取得し
                # テイカー側になっていないか確認 (postonly_reject 低減)
                try:
                    _pre_ob = await self.adapter.get_orderbook(self.config.symbol, depth=1)
                    if _pre_ob and _pre_ob.bids and _pre_ob.asks:
                        _pre_best_bid, _pre_best_ask = best_bid_ask(_pre_ob)
                        # 200# B/I: crossing → skip (snap は offset pipeline を無効化する)
                        # Gemini 推奨: offset 全計算を無駄にする snap を止め、サイクルスキップで再計算を待つ
                        if side == "buy" and order_price >= _pre_best_ask:
                            logger.warning(
                                f"[postonly_guard] 200# buy price {order_price:.0f} >= best_ask "
                                f"{_pre_best_ask:.0f} → skip cycle (offset pipeline nullified)"
                            )
                            cancel_reason = CR.POSTONLY_CROSSING_SKIP
                            break
                        elif side == "sell" and order_price <= _pre_best_bid:
                            logger.warning(
                                f"[postonly_guard] 200# sell price {order_price:.0f} <= best_bid "
                                f"{_pre_best_bid:.0f} → skip cycle (offset pipeline nullified)"
                            )
                            cancel_reason = CR.POSTONLY_CROSSING_SKIP
                            break
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
            # 200# B/I: postonly crossing は意図的 skip — circuit_breaker に通知しない
            if cancel_reason != CR.POSTONLY_CROSSING_SKIP:
                logger.error(f"All order attempts failed: {last_error}")
                # 113# resilience: API 失敗を CircuitBreaker に記録
                await self._circuit_breaker.async_on_failure()
                self._postonly_crossing_streak = 0  # 201# 非 crossing で streak リセット
            else:
                # 201# review: crossing 連続発生を検出 → 高頻度時に warning
                self._postonly_crossing_streak = getattr(
                    self, "_postonly_crossing_streak", 0
                ) + 1
                if self._postonly_crossing_streak >= 3:
                    logger.warning(
                        f"[postonly_guard] 201# crossing streak={self._postonly_crossing_streak}"
                        " — offset pipeline may need recalibration"
                    )
                logger.info("[postonly_guard] 200# crossing → cycle skipped (no CB penalty)")
            return self._make_cycle_skip_record(
                timestamp=t_submit,
                side=side,
                cancel_reason=cancel_reason,
                cycle_id=cycle_id,
                order_quantity=_order_lot,
                order_price=order_price,
                spread_at_order=spread_at_order,
                spread_offset_ratio=effective_offset_ratio,  # 096# 計算済み実効値
                error_message=last_error,  # 031# エラー詳細を記録
            )
        if not isinstance(getattr(order, "order_id", None), str):
            raise TypeError(
                f"adapter.place_order returned non-OrderLike: {type(order).__name__}"
            )
        order = cast("OrderLike", order)

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
        cancel_failed_likely_filled = monitor.cancel_failed_likely_filled  # 166# C.7
        # 237# phantom guard 登録 (status_unknown 時)
        _pending_reconciliation = self._maybe_register_phantom(monitor, side, _order_lot, order_price)

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

        # 189# D: MacroRegime 更新 + compose_regimes
        _macro_trend: Optional[str] = None
        _macro_slope_5m: Optional[float] = None
        _macro_slope_15m: Optional[float] = None
        _macro_aligned: Optional[bool] = None
        if self._macro_regime_detector is not None:
            _macro_price = mid_at_fill if mid_at_fill is not None else None
            if _macro_price is None:
                _fb, _ = self._maker_price.get_fallback_price()
                _macro_price = _fb
            if _macro_price is not None:
                from scripts.v460.lib.macro_regime import compose_regimes
                macro_result = self._macro_regime_detector.update(t_submit, _macro_price)
                _macro_trend = macro_result.trend.value
                _macro_slope_5m = macro_result.slope_5m_bps_per_min
                _macro_slope_15m = macro_result.slope_15m_bps_per_min
                if regime_str is not None:
                    _, _macro_aligned = compose_regimes(
                        regime_str, regime_conf or 0.0, macro_result,
                    )
                    if not _macro_aligned:
                        _action = getattr(self.config, "macro_regime_conflict_action", "log")
                        if _action == "downgrade":
                            regime_str = "ranging"
                            logger.info(
                                "[macro_regime] micro/macro conflict → ranging downgrade "
                                "(micro=%s, macro=%s)", regime_str, _macro_trend,
                            )
                        else:
                            logger.debug(
                                "[macro_regime] micro/macro conflict detected "
                                "(micro=%s, macro=%s, aligned=False)",
                                regime_str, _macro_trend,
                            )

        record = self._build_fill_record(
            cycle_id=cycle_id,
            t_submit=t_submit,
            side=side,
            order_price=order_price,
            order_lot=_order_lot,
            fill_price=fill_price,
            filled=filled,
            spread_at_order=spread_at_order,
            effective_offset_ratio=effective_offset_ratio,
            queue_wait=queue_wait,
            cancel_reason_poll=cancel_reason_poll,
            reprice_count=reprice_count,
            reprice_drift_bps=reprice_drift_bps,
            effective_timeout=_effective_timeout,
            cancel_failed_likely_filled=cancel_failed_likely_filled,
            pnl=pnl,
            sg_skipped=skip_gate_skipped,
            sg_score=skip_gate_score,
            sg_reason=skip_gate_reason,
            sg_model_used=skip_gate_model_used,
            sg_as_prob=skip_gate_as_prob,
            sg_threshold_used=skip_gate_threshold_used,
            sg_hour_offset=skip_gate_hour_offset,
            sg_velocity_bps=_sg_velocity_bps,
            regime_str=regime_str,
            regime_conf=regime_conf,
            regime_stab=regime_stab,
            regime_trend_pct=regime_trend_pct,
            regime_vol_ratio=regime_vol_ratio,
            balance_forced_switch=balance_forced_switch,
            confidence_factor=_confidence_factor,
            regime_lot=_regime_lot,
            macro_trend=_macro_trend,
            macro_slope_5m=_macro_slope_5m,
            macro_slope_15m=_macro_slope_15m,
            macro_aligned=_macro_aligned,
        )

        logger.info(
            f"Cycle {self._cycle_count} result: "
            f"filled={filled}, wait={queue_wait:.1f}s, "
            f"pnl={post_fill_pnl:.2f}bps" if post_fill_pnl is not None
            else f"Cycle {self._cycle_count} result: filled={filled}, wait={queue_wait:.1f}s"
        )

        # 237# phantom guard: status_unknown 時の再照合待ちフラグ
        if _pending_reconciliation:
            record.pending_reconciliation = True

        # 113# resilience: API 成功を CircuitBreaker に記録
        await self._circuit_breaker.async_on_success()

        return record
