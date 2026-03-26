"""163# Mixin: FillCycleExecutorMixin -- run_single_cycle + 直接依存メソッド.

1 サイクル: 発注 → 約定監視 → PnL 計測 → FillRecord 構築。

WARNING -- AI Coding Agent / 人間開発者への注意:
    このファイルは Mixin クラスであり、単独でインスタンス化しないこと。
    FillTestRunner.__init__ で生成される属性に依存する。
    責務: 1 取引サイクルの実行 (OB取得, SkipGate, 発注, 約定監視, PnL計測)
    run_continuous のループ制御ロジックを追加しないこと。
    side kill / time filter / balance forced 判定は fill_loop_orchestrator に属する。

323# God Object 分割:
    FillRecordBuilderMixin → fill_record_builder.py (FillRecord 構築)
    PreOrderAdjustmentsMixin → pre_order_adjustments.py (offset/price 調整)
    OffsetPipelineMixin → offset_pipeline.py (offset 乗数チェーン + lot スケール)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.cross_venue_lead_lag import (
    CrossVenueEMAState,
    VenueMidSnapshot,
    build_cross_venue_event_details,
    compute_cross_venue_lead_lag_hint,
    compute_microprice,
    update_cross_venue_ema,
)
from scripts.v460.lib.fill_config import (
    SkipGateResult as _SkipGateResult,
    FillMonitorResult as _FillMonitorResult,
    PnlMeasurement as _PnlMeasurement,
)
from scripts.v460.lib.fill_record_builder import FillRecordBuilderMixin
from scripts.v460.lib.maker_price import InfeasibleQuoteError
from scripts.v460.lib.ob_utils import best_bid_ask  # 200# 10-C: module-level import
from scripts.v460.lib.offset_pipeline import OffsetPipelineMixin
from ztb.trading.execution.contracts import OrderLike as _ExecutionOrderLike

if TYPE_CHECKING:
    from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard
    from scripts.v460.lib.fill_config import FillTestConfig
    from scripts.v460.lib.order_monitor import OrderLike
    from scripts.v460.lib.phantom_position_guard import PhantomPositionGuard
    from ztb.trading.live.exchanges.base.broker_interfaces import IBroker
    from ztb.metrics.fill_quality import FillRecord

logger = logging.getLogger(__name__)


@dataclass
class _PreOrderPhaseResult:
    cycle_id: str
    side: str
    order_price: float
    spread_at_order: float | None
    effective_offset_ratio: float
    regime_lot: float
    skip_gate_skipped: bool | None
    skip_gate_score: float | None
    skip_gate_reason: str | None
    skip_gate_model_used: str | None
    skip_gate_as_prob: float | None
    skip_gate_threshold_used: float | None
    skip_gate_hour_offset: float | None
    sg_velocity_bps: float | None
    ev_offset_applied: bool
    ev_score_pretrade: float | None
    ev_offset_mult_applied: float | None
    macro_boost_applied: bool
    execution_pre_clamp_offset: float | None
    executor_offset_stages_json: str | None
    regime_at_order: str | None
    regime_obs_count: int | None
    mid_at_order: float | None
    # 642# 可観測性
    skip_gate_forced_pass: bool
    skip_gate_side_skip_rate: float | None
    execution_hard_skip_mult_used: float | None


@dataclass
class _SubmissionPhaseResult:
    order: "OrderLike"
    order_price: float
    order_lot: float
    confidence_factor: float
    queue_depth_ahead: float | None
    queue_fill_prob_est: float | None
    t_submit: float


@dataclass
class _FillPhaseResult:
    filled: bool
    fill_price: float | None
    queue_wait: float
    cancel_reason_poll: str | None
    reprice_count: int
    reprice_drift_bps: float | None
    effective_timeout: float | None
    cancel_failed_likely_filled: bool
    pending_reconciliation: bool
    order_price: float
    requote_attempts: int
    micro_partial_qty: float


class FillCycleExecutorMixin(FillRecordBuilderMixin, OffsetPipelineMixin):
    """run_single_cycle + OB/SkipGate/Fill/PnL ヘルパー (Mixin).

    ────────────────────────────────────────────────────
    責務境界 (Single Responsibility):
      OK: 1 取引サイクル実行, OB ラッパー, SkipGate, Fill 監視, PnL 計測
      NG: ループ制御, side kill, time filter, balance forced
    MAX LINES: 1300
    ────────────────────────────────────────────────────
    188# _build_fill_record() 抽出済み
    323# FillRecordBuilderMixin / PreOrderAdjustmentsMixin 分離済み
    460# OffsetPipelineMixin 分離済み
    """

    # 201# review: 動的属性のクラスレベル宣言 (mypy 検出 + IDE 補完)
    _postonly_crossing_streak: int = 0
    # 234# no_feasible_quote 連続カウンタ (制約集合崩壊検出用)
    # 236# per-side 化: buy/sell 交互実行で相互リセットされる問題を修正
    _consecutive_no_feasible: dict[str, int] = {}  # 304# None 判定排除
    # 266# _current_regime_value は fill_record_helpers Mixin が提供 (class-level 宣言不要)
    # 237# PhantomPositionGuard: クラスレベルデフォルト (hasattr 排除)
    # 238# C-1: object → PhantomPositionGuard 型安全化 (TYPE_CHECKING)
    _phantom_guard: PhantomPositionGuard | None = None
    # 253# getattr 排除: orchestrator Mixin が設定する属性のクラスレベルデフォルト
    _alert_offset_mult: float = 1.0
    _alert_lot_mult: float = 1.0
    _halt_recovery_lot_mult: float = 1.0
    _daily_drawdown_guard: DailyDrawdownGuard | None = None
    # 303# B: DD soft lot side 分離 — side 別 lot 倍率
    _dd_soft_lot_scale_buy: float = 1.0
    _dd_soft_lot_scale_sell: float = 1.0
    _cross_venue_reference_adapter: IBroker | None = None
    _cross_venue_prev_reference_snapshot: VenueMidSnapshot | None = None
    _cross_venue_ema_state: CrossVenueEMAState | None = None
    _narrow_spread_consecutive: int = 0
    # 449# getattr 排除: orchestrator が設定する run/sha 属性
    _run_id: str = ""
    _git_sha: str = ""
    # 467# config_hash: 設定識別子 (462# 残課題)
    _config_hash: str = ""
    # 467# status_unknown_fast 連続検知 (461# P0 残課題)
    _consecutive_status_unknown_fast: int = 0
    # 458# F-lite: 前サイクルの macro_trend を保持 → 今サイクルの offset に使用
    _last_macro_trend: str | None = None

    async def _compute_orderbook_imbalance(self, depth: int = 5) -> tuple[float, float, float]:
        """054# S1: 板不均衡を計算 — 120# MakerPriceCalculator に委譲."""
        r = await self._maker_price.compute_imbalance(self.adapter, self.config.symbol, depth=depth)
        return r.imbalance, r.bid_total, r.ask_total

    async def _get_mid_price(self) -> float:
        """板の best bid/ask から mid price を算出 — 120# MakerPriceCalculator に委譲."""
        return cast(
            float,
            await self._maker_price.get_mid_price(self.adapter, self.config.symbol),
        )

    async def _compute_maker_price(self, side: str) -> tuple[float, float, float]:
        """maker limit 価格を算出 — 120# MakerPriceCalculator に委譲."""
        r = await self._maker_price.compute(side, self.adapter, self.config.symbol)
        return r.price, r.spread, r.effective_offset_ratio

    async def _update_cross_venue_lead_lag_hint(self) -> None:
        """439# Cross-venue lead-lag hint を更新して maker_price に注入する.

        442# 拡張: L5 microprice + depth imbalance 信号。
        445# EMA 平滑化 + confidence scoring。
        """
        from scripts.v460.lib.event_logger import log_event
        from scripts.v460.lib.ob_utils import depth_volume

        self._maker_price.set_cross_venue_lead_lag_hint(None)
        if not self.config.cross_venue_lead_lag_enabled:
            return

        reference_adapter = self._cross_venue_reference_adapter
        if reference_adapter is None:
            return

        local_ob = self._maker_price._last_ob_snapshot
        if local_ob is None:
            return
        local_bid, local_ask = best_bid_ask(local_ob)
        if (
            local_bid is None
            or local_ask is None
            or local_bid <= 0.0
            or local_ask <= 0.0
        ):
            return

        # 442# local microprice (available from cached OB)
        # 512# DRY: compute_microprice ヘルパーで重複排除
        local_microprice: float | None = None
        local_bid_depth = 0.0
        local_ask_depth = 0.0
        if hasattr(local_ob, "bids") and hasattr(local_ob, "asks"):
            local_bid_depth = depth_volume(local_ob.bids, depth=5)
            local_ask_depth = depth_volume(local_ob.asks, depth=5)
            local_microprice = compute_microprice(local_ob.bids, local_ob.asks)

        local_snapshot = VenueMidSnapshot(
            exchange=str(getattr(local_ob, "exchange", "") or "local"),
            mid_price=(local_bid + local_ask) / 2.0,
            timestamp=float(getattr(local_ob, "timestamp", time.time())),
            microprice=local_microprice,
            bid_depth=local_bid_depth,
            ask_depth=local_ask_depth,
        )

        try:
            ob_depth = self.config.cross_venue_reference_ob_depth
            reference_ob = await reference_adapter.get_orderbook(
                self.config.symbol,
                depth=ob_depth,
            )
            ref_bid, ref_ask = best_bid_ask(reference_ob)
            if (
                ref_bid is None
                or ref_ask is None
                or ref_bid <= 0.0
                or ref_ask <= 0.0
            ):
                return

            # 442# reference microprice + depth
            # 512# DRY: compute_microprice ヘルパーで重複排除
            ref_microprice: float | None = None
            ref_bid_depth = 0.0
            ref_ask_depth = 0.0
            if hasattr(reference_ob, "bids") and hasattr(reference_ob, "asks"):
                ref_bid_depth = depth_volume(reference_ob.bids, depth=ob_depth)
                ref_ask_depth = depth_volume(reference_ob.asks, depth=ob_depth)
                if self.config.cross_venue_microprice_enabled:
                    ref_microprice = compute_microprice(
                        reference_ob.bids, reference_ob.asks,
                    )

            ref_mid = (ref_bid + ref_ask) / 2.0
            current_reference = VenueMidSnapshot(
                exchange=str(
                    getattr(reference_ob, "exchange", "")
                    or self.config.cross_venue_reference_exchange
                ),
                mid_price=ref_mid,
                timestamp=float(getattr(reference_ob, "timestamp", time.time())),
                microprice=ref_microprice,
                bid_depth=ref_bid_depth,
                ask_depth=ref_ask_depth,
            )

            # 445# EMA 更新: spread を平滑化して安定的な方向判定を行う
            loc_mid = local_snapshot.mid_price
            point_spread_bps = (
                (ref_mid - loc_mid) / loc_mid * 10_000.0 if loc_mid > 0 else 0.0
            )
            ema_state = self._cross_venue_ema_state
            ema_state = update_cross_venue_ema(
                ema_state,
                ref_mid=ref_mid,
                spread_bps=point_spread_bps,
                timestamp=current_reference.timestamp,
                alpha=self.config.cross_venue_ema_alpha,
                # 506# basis EMA: enabled 時のみ basis_alpha > 0
                basis_alpha=(
                    self.config.cross_venue_basis_ema_alpha
                    if self.config.cross_venue_basis_correction_enabled
                    else 0.0
                ),
            )
            self._cross_venue_ema_state = ema_state

            previous_reference = self._cross_venue_prev_reference_snapshot
            hint = compute_cross_venue_lead_lag_hint(
                local_snapshot=local_snapshot,
                reference_snapshot=current_reference,
                previous_reference_snapshot=previous_reference,
                max_age_sec=self.config.cross_venue_lead_lag_max_age_sec,
                spread_bps_threshold=self.config.cross_venue_lead_lag_spread_bps_threshold,
                velocity_bps_threshold=self.config.cross_venue_lead_lag_velocity_bps_threshold,
                # 445# confidence mode
                ema_spread_bps=ema_state.ema_spread_bps,
                min_confidence=self.config.cross_venue_min_confidence,
                confidence_reference_spread_bps=self.config.cross_venue_confidence_reference_spread_bps,
                # 449# DRY: 既計算の point_spread_bps を渡して重複排除
                precomputed_point_spread_bps=point_spread_bps,
                confidence_floor=self.config.cross_venue_confidence_floor,
                # 506# basis correction (de-meaning)
                basis_bps=(
                    ema_state.ema_basis_bps
                    if self.config.cross_venue_basis_correction_enabled
                    else 0.0
                ),
            )
            self._maker_price.set_cross_venue_lead_lag_hint(hint)
            self._cross_venue_prev_reference_snapshot = current_reference
            if hint is not None:
                logger.info(
                    "[cross_venue] hint direction=%s adverse_side=%s spread=%+.2fbps "
                    "velocity=%+.2fbps/s age=%.2fs conf=%.2f microprice_spread=%s "
                    "depth_imb=%s ema_spread=%+.2fbps basis=%+.2fbps adj=%s",
                    hint.direction,
                    hint.adverse_side,
                    hint.spread_bps,
                    hint.reference_velocity_bps,
                    hint.age_sec,
                    hint.confidence,
                    f"{hint.microprice_spread_bps:+.2f}bps" if hint.microprice_spread_bps is not None else "N/A",
                    f"{hint.depth_imbalance:+.3f}" if hint.depth_imbalance is not None else "N/A",
                    ema_state.ema_spread_bps,
                    hint.basis_bps,
                    f"{hint.adjusted_spread_bps:+.2f}bps" if hint.adjusted_spread_bps is not None else "N/A",
                )
                # 449# 安定性: クラスレベルデフォルト宣言により直接参照
                log_event(
                    "cross_venue_hint",
                    self.config.results_dir,
                    run_id=str(self._run_id),
                    git_sha=str(self._git_sha),
                    details=build_cross_venue_event_details(hint),
                )
            else:
                # 445# hint=None の理由を具体値付きで可視化
                # 449# DRY: _spread は L200 の point_spread_bps を再利用
                if previous_reference is None:
                    _reason = "first_call"
                    _vel_s = "N/A"
                else:
                    _dt = current_reference.timestamp - previous_reference.timestamp
                    _vel = (
                        (current_reference.mid_price - previous_reference.mid_price)
                        / previous_reference.mid_price * 10_000.0 / _dt
                        if _dt > 0 and previous_reference.mid_price > 0
                        else 0.0
                    )
                    _vel_s = f"{_vel:+.4f}bps/s"
                    _ema_spr = ema_state.ema_spread_bps
                    _spr_thr = self.config.cross_venue_lead_lag_spread_bps_threshold
                    _min_conf = self.config.cross_venue_min_confidence
                    if abs(_ema_spr) < _spr_thr:
                        _reason = f"ema_spread({_ema_spr:+.2f})<{_spr_thr}"
                    else:
                        _reason = f"low_confidence(<{_min_conf})"
                logger.info(
                    "[cross_venue] hint=None reason=%s spread=%+.2fbps "
                    "ema=%+.2fbps vel=%s",
                    _reason, point_spread_bps, ema_state.ema_spread_bps, _vel_s,
                )
        except (ConnectionError, TimeoutError, OSError, ValueError, AttributeError) as exc:
            logger.warning("cross-venue hint update error [%s]: %s", type(exc).__name__, exc)
            self._maker_price.set_cross_venue_lead_lag_hint(None)
        except Exception as exc:
            logger.error("cross-venue hint unexpected error [%s]: %s", type(exc).__name__, exc, exc_info=True)
            self._maker_price.set_cross_venue_lead_lag_hint(None)

    def _make_price_error_skip(
        self,
        *,
        side: str,
        cancel_reason: str,
        cycle_id: str,
        error: Exception,
    ) -> FillRecord:
        """239# maker price エラー時の fallback price + skip record 共通化.

        155# §9.5 #3: 前回 mid_price をフォールバック参照。
        156# §10 #5: 鮮度判定 — 閾値超は stale とみなす。
        """
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
            _fallback_stale = True
            logger.info(
                f"[155# ob_fallback] Using last mid_price={_fallback_price:.0f} "
                f"(no timestamp, treated as stale) as reference for skip record"
            )
        return self._make_cycle_skip_record(
            side=side,
            cancel_reason=cancel_reason,
            cycle_id=cycle_id,
            order_price=_fallback_price if not _fallback_stale else 0.0,
            spread_offset_ratio=self._maker_price.base_offset_ratio,
            error_message=(
                f"{error} [fallback_age={_fallback_age:.1f}s stale={_fallback_stale}]"
                if _fallback_age is not None else str(error)
            ),
        )

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

        238# C-2: BalanceChecker の最終クエリ残高を snapshot として渡す。
        これがないと Phase 2 (残高差分検出) が完全に無効。
        467#: status_unknown_fast 連続検知カウンタを更新。

        Returns:
            True = pending_reconciliation (FillRecord に設定する)
        """
        # 467#: status_unknown_fast 連続検知 (ファントム登録判定とは独立)
        if monitor.cancel_reason == "status_unknown_fast":
            self._consecutive_status_unknown_fast += 1
            if self._consecutive_status_unknown_fast >= 3:
                logger.warning(
                    f"[467# status_unknown_fast] {self._consecutive_status_unknown_fast} "
                    f"consecutive status_unknown_fast — possible API degradation or "
                    f"exchange connectivity issue"
                )
        else:
            self._consecutive_status_unknown_fast = 0

        if (
            monitor.filled
            or monitor.cancel_reason is None
            or not monitor.cancel_reason.startswith("status_unknown")
            or monitor.order_id_for_reconciliation is None
        ):
            return False

        if self._phantom_guard is not None:
            # 251# getattr → 型安全な property 直接参照 (238# C-2 完全化)
            _btc_snap = self._balance_checker.last_btc_free
            # 251# T-3: buy 側 JPY 残高照合用 snapshot 追加
            _jpy_snap = self._balance_checker.last_jpy_free
            self._phantom_guard.register_unknown(
                order_id=monitor.order_id_for_reconciliation,
                side=side,
                quantity=order_lot,
                price=order_price,
                balance_btc=_btc_snap,
                balance_jpy=_jpy_snap,
            )
        return True

    # ==================================================================
    # 113# R1: run_single_cycle から抽出したサブメソッド
    # ==================================================================

    async def _evaluate_skip_gate(
        self,
        side: str,
        cycle_id: str,
        order_price: float,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        *,
        order_lot: float | None = None,
        one_sided_balance: bool = False,
        prefetched_ob: object | None = None,
        prefetched_trades: object | None = None,
    ) -> _SkipGateResult:
        """SkipGate ML 判定 — 121# SkipGateEvaluator に委譲.

        145# §9-#4: order_lot を渡してレジーム倍率適用後のロットで記録.
        190# B: one_sided_balance を渡して片側 balance 時の threshold 緩和.
        355# L-1/L-2: prefetched_ob / prefetched_trades で API 呼出し削減.
        """
        regime_value = (
            self._regime_detector.current_regime.value
            if self._regime_detector is not None
            else None
        )
        _lot = order_lot if order_lot is not None else self._current_lot
        # 343# kill 解除直後の skip_gate 緩和 offset を計算
        _kill_rel_offset = 0.0
        _rc = self._kill_released_at_cycle_buy if side == "buy" else self._kill_released_at_cycle_sell
        if _rc is not None and self.config.skip_gate_kill_release_grace_cycles > 0:
            if 0 <= self._cycle_count - _rc < self.config.skip_gate_kill_release_grace_cycles:
                _kill_rel_offset = self.config.skip_gate_kill_release_offset
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
            kill_release_offset=_kill_rel_offset,
            prefetched_ob=prefetched_ob,
            prefetched_trades=prefetched_trades,
        )

    async def _monitor_fill_polling(
        self,
        order: OrderLike,
        order_price: float,
        side: str,
        t_submit: float,
        spread_at_order: float | None,
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
    # 188# FillRecord 構築 (run_single_cycle からの抽出)
    # 323# → fill_record_builder.py (FillRecordBuilderMixin)
    # ------------------------------------------------------------------

    async def _measure_post_fill_pnl(
        self,
        filled: bool,
        fill_price: float | None,
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

    def _log_cycle_result(
        self,
        *,
        filled: bool,
        queue_wait: float,
        post_fill_pnl: float | None,
        cancel_reason: str | None = None,
        sidecar_offset_bps: float = 0.0,
        sidecar_signal_status: str | None = None,
        order_id: str | None = None,
        regime: str = "n/a",
    ) -> None:
        # 487# P0: sidecar summary を cycle log に追記
        _sidecar_tag = ""
        if sidecar_signal_status and sidecar_signal_status != "missing":
            _sidecar_tag = f", sidecar={sidecar_signal_status}"
            if sidecar_offset_bps != 0.0:
                _sidecar_tag += f"({sidecar_offset_bps:+.3f}bps)"
        # 487# P2: cancel_reason を追記 (unfilled 時に原因可視化)
        _cancel_tag = f", reason={cancel_reason}" if cancel_reason and not filled else ""
        # 526# order_id 追記
        _id_tag = f", id={order_id}" if order_id else ""
        # 533# BTC 残高コンテキスト (在庫状態の可視化)
        _btc = self._balance_checker.last_btc_free
        _bal_tag = f", btc={_btc:.6f}" if _btc is not None else ""
        # 632# regime context
        _regime_tag = f", regime={regime}"
        if post_fill_pnl is not None:
            logger.info(
                f"Cycle {self._cycle_count} result: "
                f"filled={filled}, wait={queue_wait:.1f}s, pnl={post_fill_pnl:.2f}bps"
                f"{_id_tag}{_bal_tag}{_regime_tag}{_sidecar_tag}"
            )
            return
        logger.info(
            f"Cycle {self._cycle_count} result: "
            f"filled={filled}, wait={queue_wait:.1f}s"
            f"{_id_tag}{_cancel_tag}{_bal_tag}{_regime_tag}{_sidecar_tag}"
        )

    def _log_cycle_revenue_context(self, record: "FillRecord") -> None:
        """収益分析しやすい cycle 文脈を event log に追記する."""
        from scripts.v460.lib.event_logger import (
            build_cycle_revenue_event_details,
            log_event,
        )

        log_event(
            "cycle_revenue_context",
            self.config.results_dir,
            run_id=str(self._run_id),
            git_sha=str(self._git_sha),
            reason=record.cancel_reason if not record.filled else None,
            details=build_cycle_revenue_event_details(record),
        )

    async def _run_pre_order_phase(
        self,
        *,
        cycle_id: str,
        side_override: str | None,
        one_sided_balance: bool,
        trending_offset_mult: float | None,
        degraded_liquidation: bool,
        toxicity_offset_mult: float,
        sidecar_offset_bps: float,
    ) -> "FillRecord | _PreOrderPhaseResult":
        """Task C: pre-order phase (OB/SkipGate/offset) をまとめる."""
        try:
            imb, bid_d, ask_d = await self._compute_orderbook_imbalance(
                depth=self.config.imbalance_depth,
            )
            self._maker_price._last_imbalance = imb
            self._maker_price._last_bid_depth = bid_d
            self._maker_price._last_ask_depth = ask_d
            ob = self._maker_price._last_ob_snapshot
            if ob is not None:
                self._ob_recorder.record(ob.bids, ob.asks, ob.timestamp)
        except Exception as e:
            logger.warning(f"[ob_prefetch] Pre-fetch imbalance failed, using last: {e}")

        prefetched_trades: object | None = None
        try:
            recent = await self.adapter.get_recent_trades(
                self.config.symbol,
                limit=self.config.trades_recorder_fetch_limit,
            )
            self._trades_recorder.record_from_adapter(recent)
            prefetched_trades = recent
        except Exception as te:
            logger.debug(f"Trades fetch for recording skipped: {te}")

        side = side_override if side_override is not None else self._next_side()
        self._side_selector.update_after_decision(side)
        logger.info(f"=== Cycle {self._cycle_count} ({side}) ===")

        regime_at_order = self._current_regime_value()
        regime_obs_count: int | None = None
        if self._regime_detector is not None:
            regime_obs_count = self._regime_detector.observation_count

        await self._update_cross_venue_lead_lag_hint()
        self._maker_price.set_veto_btc_balance(self._balance_checker.last_btc_free)

        spread_at_order: float | None = None
        effective_offset_ratio: float = self.config.spread_offset_ratio
        mid_at_order: float | None = None
        try:
            order_price, spread_at_order, effective_offset_ratio = await self._compute_maker_price(side)
            mid_at_order = self._maker_price.get_fallback_price()[0]
            self._consecutive_no_feasible[side] = 0
        except InfeasibleQuoteError as e:
            ob_cancel_reason = e.reason
            if e.reason == "spread_too_narrow":
                logger.info(f"[158# §20-D] {e}")
            else:
                logger.warning(f"Maker price rejected: {e}")
            cnf = self._consecutive_no_feasible.get(side, 0) + 1
            self._consecutive_no_feasible[side] = cnf
            if cnf >= 3:
                ob_cancel_reason = CR.NO_FEASIBLE_QUOTE
                logger.warning(
                    f"[234#] NO_FEASIBLE_QUOTE: {cnf} "
                    f"consecutive infeasible quotes ({side}) -- "
                    f"last_reason={e.reason}, "
                    f"min_spread={self.config.min_spread_jpy}, "
                    f"sell_max_spread={self.config.sell_max_spread_jpy}"
                )
            return self._make_price_error_skip(
                side=side,
                cancel_reason=ob_cancel_reason,
                cycle_id=cycle_id,
                error=e,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "timeout" in err_msg or "timed out" in err_msg:
                ob_cancel_reason = CR.ORDERBOOK_TIMEOUT
            elif "rate" in err_msg or "limit" in err_msg or "too many" in err_msg:
                ob_cancel_reason = CR.ORDERBOOK_RATE_LIMIT
            elif "empty" in err_msg or "no bid" in err_msg or "no ask" in err_msg:
                ob_cancel_reason = CR.ORDERBOOK_EMPTY
            else:
                ob_cancel_reason = CR.ORDERBOOK_ERROR
            logger.error(f"Failed to compute maker price: {e}")
            return self._make_price_error_skip(
                side=side,
                cancel_reason=ob_cancel_reason,
                cycle_id=cycle_id,
                error=e,
            )

        if degraded_liquidation and effective_offset_ratio > 0:
            deg_offset_mult = self.config.degraded_liquidation_offset_mult
            pre_deg_offset = effective_offset_ratio
            effective_offset_ratio = min(
                effective_offset_ratio * deg_offset_mult,
                self.config.max_offset_ratio,
            )
            order_price = self._recalc_price_with_new_offset(
                side,
                order_price,
                spread_at_order,
                pre_deg_offset,
                effective_offset_ratio,
            )
            if spread_at_order is None or spread_at_order <= 0:
                logger.warning(
                    "[235#] degraded_liquidation: spread unavailable -- "
                    "offset expanded but price NOT recalculated"
                )
            logger.warning(
                f"[234#] degraded_liquidation: offset "
                f"{pre_deg_offset:.4f}->{effective_offset_ratio:.4f} "
                f"(x{deg_offset_mult:.1f}), price={order_price:.0f}"
            )

        if (
            self.config.narrow_spread_pause_enabled
            and spread_at_order is not None
            and order_price > 0
        ):
            mid_est = order_price
            spread_bps_val = (
                spread_at_order / mid_est * self._BPS_FACTOR if mid_est > 0 else 0.0
            )
            if spread_bps_val < self.config.narrow_spread_pause_bps:
                self._narrow_spread_consecutive += 1
                if self._narrow_spread_consecutive <= self.config.narrow_spread_pause_max_consecutive:
                    pause_sec = self.config.narrow_spread_pause_sec
                    logger.info(
                        f"[137# P1-08] Spread too narrow ({spread_bps_val:.1f}bps "
                        f"< {self.config.narrow_spread_pause_bps}bps). "
                        f"Pausing {pause_sec}s "
                        f"({self._narrow_spread_consecutive}/"
                        f"{self.config.narrow_spread_pause_max_consecutive})"
                    )
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

        regime_lot = self._regime_adjusted_lot()
        if degraded_liquidation:
            pre_deg_lot = regime_lot
            regime_lot = max(
                regime_lot * self.config.degraded_liquidation_lot_mult,
                self.config.min_order_btc,
            )
            logger.warning(
                f"[234#] degraded_liquidation lot: "
                f"{pre_deg_lot:.6f}->{regime_lot:.6f} "
                f"(x{self.config.degraded_liquidation_lot_mult:.1f})"
            )

        sg = await self._evaluate_skip_gate(
            side,
            cycle_id,
            order_price,
            spread_at_order,
            effective_offset_ratio,
            order_lot=regime_lot,
            one_sided_balance=one_sided_balance,
            prefetched_ob=self._maker_price._last_ob_snapshot,
            prefetched_trades=prefetched_trades,
        )
        if sg.early_return_record is not None:
            return sg.early_return_record

        offset_result = self._apply_offset_pipeline(
            side=side,
            order_price=order_price,
            spread_at_order=spread_at_order,
            effective_offset_ratio=effective_offset_ratio,
            sg_ev_score=sg.ev_score,
            sg_velocity_offset_mult=sg.velocity_offset_mult,
            sg_velocity_bps=sg.price_velocity_bps,
            trending_offset_mult=trending_offset_mult,
            toxicity_offset_mult=toxicity_offset_mult,
            sidecar_offset_bps=sidecar_offset_bps,
            cycle_id=cycle_id,
        )
        if offset_result.early_return_record is not None:
            return offset_result.early_return_record

        return _PreOrderPhaseResult(
            cycle_id=cycle_id,
            side=side,
            order_price=offset_result.order_price,
            spread_at_order=spread_at_order,
            effective_offset_ratio=offset_result.effective_offset_ratio,
            regime_lot=regime_lot,
            skip_gate_skipped=sg.skipped,
            skip_gate_score=sg.score,
            skip_gate_reason=sg.reason,
            skip_gate_model_used=sg.model_used,
            skip_gate_as_prob=sg.as_prob,
            skip_gate_threshold_used=sg.threshold_used,
            skip_gate_hour_offset=sg.hour_offset if sg.hour_offset != 0.0 else None,
            sg_velocity_bps=sg.price_velocity_bps,
            ev_offset_applied=offset_result.ev_offset_applied,
            ev_score_pretrade=offset_result.ev_score_pretrade,
            ev_offset_mult_applied=offset_result.ev_offset_mult_applied,
            macro_boost_applied=offset_result.macro_boost_applied,
            execution_pre_clamp_offset=offset_result.execution_pre_clamp_offset,
            executor_offset_stages_json=offset_result.executor_offset_stages_json,
            regime_at_order=regime_at_order,
            regime_obs_count=regime_obs_count,
            mid_at_order=mid_at_order,
            skip_gate_forced_pass=sg.forced_pass,
            skip_gate_side_skip_rate=sg.side_skip_rate,
            execution_hard_skip_mult_used=offset_result.execution_hard_skip_mult_used,
        )

    async def _submit_order_phase(
        self,
        *,
        pre_order: _PreOrderPhaseResult,
        one_sided_balance: bool,
    ) -> "FillRecord | _SubmissionPhaseResult":
        """Task C: submission phase (lot resolve / place order) をまとめる."""
        self._balance_checker.apply_lot_floor()
        order_lot, confidence_factor = self._effective_order_lot(
            pre_order.regime_lot,
            as_prob=pre_order.skip_gate_as_prob,
            dust_sweep_active=self._balance_checker.dust_sweep_active,
        )

        dust_active = self._balance_checker.dust_sweep_active
        min_lot = self.config.min_order_btc
        if not dust_active:
            alert_lm = self._alert_lot_mult
            if alert_lm != 1.0:
                order_lot = self._scale_lot(order_lot, alert_lm, min_lot, "215# alert_mode", warn=True)
            recovery_lm = self._halt_recovery_lot_mult
            if recovery_lm < 1.0:
                order_lot = self._scale_lot(order_lot, recovery_lm, min_lot, "224# B1 Recovery")
            dd_side_scale = (
                self._dd_soft_lot_scale_buy
                if pre_order.side == "buy"
                else self._dd_soft_lot_scale_sell
            )
            if dd_side_scale < 1.0:
                order_lot = self._scale_lot(order_lot, dd_side_scale, min_lot, f"303# B DD soft {pre_order.side}")
            dd_guard = self._daily_drawdown_guard
            if dd_guard is not None:
                cd_lm = dd_guard.get_cooldown_lot_scale()
                if cd_lm < 1.0:
                    order_lot = self._scale_lot(order_lot, cd_lm, min_lot, "246# cooldown_release")

        if self.config.max_lot > 0 and order_lot > self.config.max_lot:
            logger.warning(
                f"[373# F8] lot {order_lot:.6f} > max_lot {self.config.max_lot:.6f} -- clamped"
            )
            order_lot = self.config.max_lot

        queue_depth_ahead: float | None = None
        queue_fill_prob_est: float | None = None
        if self.config.queue_position_tracking_enabled and pre_order.order_price > 0:
            queue_depth_ahead = self._maker_price.estimate_queue_depth(
                pre_order.side,
                pre_order.order_price,
            )
            if queue_depth_ahead is not None and order_lot > 0:
                import math as _math

                queue_fill_prob_est = _math.exp(-queue_depth_ahead / max(order_lot, 1e-8))

        t_submit = time.time()
        order: _ExecutionOrderLike | None = None
        last_error: str | None = None
        cancel_reason: str = CR.UNKNOWN
        order_price = pre_order.order_price
        for attempt in range(1 + self.config.max_order_retries):
            try:
                try:
                    pre_ob = await self.adapter.get_orderbook(self.config.symbol, depth=1)
                    if pre_ob and pre_ob.bids and pre_ob.asks:
                        pre_best_bid, pre_best_ask = best_bid_ask(pre_ob)
                        if pre_order.side == "buy" and order_price >= pre_best_ask:
                            logger.warning(
                                f"[postonly_guard] 200# buy price {order_price:.0f} >= best_ask "
                                f"{pre_best_ask:.0f} -> skip cycle (offset pipeline nullified)"
                            )
                            cancel_reason = CR.POSTONLY_CROSSING_SKIP
                            break
                        if pre_order.side == "sell" and order_price <= pre_best_bid:
                            logger.warning(
                                f"[postonly_guard] 200# sell price {order_price:.0f} <= best_bid "
                                f"{pre_best_bid:.0f} -> skip cycle (offset pipeline nullified)"
                            )
                            cancel_reason = CR.POSTONLY_CROSSING_SKIP
                            break
                except Exception as pre_e:
                    logger.debug(f"[postonly_guard] Pre-check failed (non-fatal): {pre_e}")

                order = cast(
                    _ExecutionOrderLike,
                    await self.adapter.place_order(
                        symbol=self.config.symbol,
                        side=pre_order.side,
                        quantity=order_lot,
                        price=order_price,
                        order_type="limit",
                    ),
                )
                self._pending_order_id = order.order_id
                logger.info(
                    f"Placed {pre_order.side} limit @ {order_price:.0f} JPY, "
                    f"qty={order_lot}, id={order.order_id}"
                    + (f" (retry {attempt})" if attempt > 0 else "")
                )
                break
            except Exception as e:
                last_error = str(e)
                err_lower = last_error.lower()
                if "post_only" in err_lower or "taker" in err_lower:
                    cancel_reason = CR.POST_ONLY_REJECT
                elif (
                    "insufficient" in err_lower
                    or "balance" in err_lower
                    or any(p in last_error for p in self.config.insufficient_funds_patterns)
                ):
                    cancel_reason = CR.INSUFFICIENT_FUNDS
                elif "minimum" in err_lower or "size" in err_lower:
                    cancel_reason = CR.MINIMUM_SIZE
                else:
                    cancel_reason = CR.API_ERROR

                logger.warning(
                    f"Order attempt {attempt + 1} failed ({cancel_reason}): {e}"
                )
                if cancel_reason in {"insufficient_funds", "post_only_reject", "minimum_size"}:
                    logger.info(
                        f"[Bug10] Skipping retry -- {cancel_reason} is not retriable"
                    )
                    break

                if attempt < self.config.max_order_retries:
                    backoff = self.config.retry_delay_sec * (
                        self.config.retry_backoff_base ** attempt
                    )
                    if "rate" in err_lower or "limit" in err_lower or "too many" in err_lower:
                        backoff = max(backoff, self.config.rate_limit_min_backoff_sec)
                        logger.warning(
                            f"Rate-limit detected, extended backoff: {backoff:.1f}s (attempt {attempt + 1})"
                        )
                    else:
                        logger.info(f"Retry backoff: {backoff:.1f}s")
                    await asyncio.sleep(backoff)
                    try:
                        ob = await self.adapter.get_orderbook(self.config.symbol, depth=1)
                        if ob.bids and ob.asks:
                            order_price = ob.bids[0][0] if pre_order.side == "buy" else ob.asks[0][0]
                            logger.info(f"Retry with conservative price: {order_price:.0f}")
                    except Exception as e:
                        logger.debug(
                            "OB fetch failed during retry, using previous price: %s",
                            e,
                            exc_info=True,
                        )

        if order is None:
            if cancel_reason != CR.POSTONLY_CROSSING_SKIP:
                logger.error(
                    f"All order attempts failed (side={pre_order.side}, qty={order_lot:.8f}): {last_error}"
                )
                await self._circuit_breaker.async_on_failure()
                self._postonly_crossing_streak = 0
            else:
                self._postonly_crossing_streak += 1
                if self._postonly_crossing_streak >= 3:
                    logger.warning(
                        f"[postonly_guard] 201# crossing streak={self._postonly_crossing_streak}"
                        " -- offset pipeline may need recalibration"
                    )
                logger.info("[postonly_guard] 200# crossing -> cycle skipped (no CB penalty)")
            return self._make_cycle_skip_record(
                timestamp=t_submit,
                side=pre_order.side,
                cancel_reason=cancel_reason,
                cycle_id=pre_order.cycle_id,
                order_quantity=order_lot,
                order_price=order_price,
                spread_at_order=pre_order.spread_at_order,
                spread_offset_ratio=pre_order.effective_offset_ratio,
                error_message=last_error,
            )

        if not isinstance(getattr(order, "order_id", None), str):
            raise TypeError(
                f"adapter.place_order returned non-OrderLike: {type(order).__name__}"
            )

        return _SubmissionPhaseResult(
            order=cast("OrderLike", order),
            order_price=order_price,
            order_lot=order_lot,
            confidence_factor=confidence_factor,
            queue_depth_ahead=queue_depth_ahead,
            queue_fill_prob_est=queue_fill_prob_est,
            t_submit=t_submit,
        )

    async def _monitor_fill_phase(
        self,
        *,
        pre_order: _PreOrderPhaseResult,
        submission: _SubmissionPhaseResult,
    ) -> _FillPhaseResult:
        """Task C: fill monitoring phase (polling/timeout) をまとめる."""
        order = submission.order
        order_price = submission.order_price
        t_submit = submission.t_submit
        micro_enabled = self.config.micro_timeout_enabled
        requote_attempts = 0
        micro_partial_qty = 0.0
        remaining_lot = submission.order_lot
        first_t_submit = t_submit

        if micro_enabled:
            mt_wait = (
                self.config.micro_timeout_wait_sec_sell
                if pre_order.side == "sell" and self.config.micro_timeout_wait_sec_sell is not None
                else self.config.micro_timeout_wait_sec
            )
            lt = self._last_macro_trend
            if pre_order.side == "sell" and lt is not None:
                from scripts.v460.lib.macro_regime import MacroTrend

                if (
                    lt == MacroTrend.STRONG_UP.value
                    and self.config.macro_sell_timeout_strong_up is not None
                ):
                    mt_wait = self.config.macro_sell_timeout_strong_up
                    logger.info("[458# H] sell timeout shortened: macro=STRONG_UP -> %.1fs", mt_wait)
                elif (
                    lt == MacroTrend.WEAK_UP.value
                    and self.config.macro_sell_timeout_weak_up is not None
                ):
                    mt_wait = self.config.macro_sell_timeout_weak_up
                    logger.info("[458# H] sell timeout shortened: macro=WEAK_UP -> %.1fs", mt_wait)
            mt_max = self.config.micro_timeout_max_requote
            mt_cooloff = self.config.micro_timeout_requote_cooloff_sec
            original_timeout = self.config.order_timeout_sec
            original_timeout_sell = self.config.order_timeout_sec_sell
            mt_total_cap: float | None = (
                self.config.sell_age_cap_sec
                if pre_order.side == "sell"
                and self.config.sell_age_cap_sec is not None
                and self.config.sell_age_cap_sec > 0
                else None
            )

            for mt_attempt in range(mt_max):
                if self._kill_switch.is_killed():
                    break
                if mt_total_cap is not None:
                    mt_elapsed = time.time() - first_t_submit
                    if mt_elapsed >= mt_total_cap:
                        logger.info(
                            "[509#] micro_timeout sell_age_cap exceeded: "
                            "elapsed=%.1fs >= cap=%.0fs, stopping at attempt %d/%d",
                            mt_elapsed,
                            mt_total_cap,
                            mt_attempt + 1,
                            mt_max,
                        )
                        # 603# age_cap exceeded: 滞留注文をキャンセル
                        try:
                            await self.adapter.cancel_order(order.order_id)
                            logger.info(
                                "[603#] Cancelled order %s on age_cap exceeded",
                                order.order_id,
                            )
                        except Exception as e:
                            logger.warning(
                                "[603#] Cancel failed for order %s on age_cap "
                                "exceeded (may be filled/cancelled): %s",
                                order.order_id,
                                e,
                            )
                        break

                object.__setattr__(self.config, "order_timeout_sec", mt_wait)
                object.__setattr__(self.config, "order_timeout_sec_sell", None)
                try:
                    monitor = await self._monitor_fill_polling(
                        order,
                        order_price,
                        pre_order.side,
                        t_submit,
                        pre_order.spread_at_order,
                        pre_order.effective_offset_ratio,
                        order_lot=remaining_lot,
                    )
                finally:
                    object.__setattr__(self.config, "order_timeout_sec", original_timeout)
                    object.__setattr__(self.config, "order_timeout_sec_sell", original_timeout_sell)

                if monitor.filled or monitor.cancel_failed_likely_filled:
                    break

                requote_attempts = mt_attempt + 1
                if mt_attempt >= mt_max - 1:
                    logger.info(
                        "[452# micro_timeout] Max requote reached (%d), giving up",
                        mt_max,
                    )
                    break

                if mt_cooloff > 0:
                    await asyncio.sleep(mt_cooloff)

                try:
                    rq_mid = await self._get_mid_price()
                except Exception as e:
                    logger.warning("[452# micro_timeout] Mid price fetch failed: %s", e)
                    break

                rq_spread = (
                    pre_order.spread_at_order
                    if pre_order.spread_at_order and pre_order.spread_at_order > 0
                    else 0.0
                )
                if pre_order.side == "buy":
                    order_price = round(
                        rq_mid + rq_spread * (pre_order.effective_offset_ratio - 0.5)
                    )
                else:
                    order_price = round(
                        rq_mid + rq_spread * (0.5 - pre_order.effective_offset_ratio)
                    )

                logger.info(
                    "[452# micro_timeout] Re-quote %d/%d: new_price=%s (mid=%s, offset=%.4f)",
                    mt_attempt + 1,
                    mt_max,
                    order_price,
                    rq_mid,
                    pre_order.effective_offset_ratio,
                )
                t_submit = time.time()
                try:
                    order = await self.adapter.place_order(
                        symbol=self.config.symbol,
                        side=pre_order.side,
                        quantity=remaining_lot,
                        price=order_price,
                        order_type="limit",
                    )
                except Exception as e:
                    logger.warning("[452# micro_timeout] Re-quote order failed: %s", e)
                    break

                if order is None or not isinstance(getattr(order, "order_id", None), str):
                    logger.warning("[452# micro_timeout] Re-quote returned invalid order")
                    break
                order = cast("OrderLike", order)
        else:
            monitor = await self._monitor_fill_polling(
                order,
                order_price,
                pre_order.side,
                t_submit,
                pre_order.spread_at_order,
                pre_order.effective_offset_ratio,
                order_lot=submission.order_lot,
            )

        filled = monitor.filled
        fill_price = monitor.fill_price
        queue_wait = time.time() - first_t_submit if micro_enabled and filled else monitor.queue_wait
        if micro_enabled and not filled and monitor.cancel_reason == "timeout":
            cancel_reason_poll: str | None = "micro_timeout"
        else:
            cancel_reason_poll = monitor.cancel_reason
        final_order_price = monitor.final_order_price
        pending_reconciliation = self._maybe_register_phantom(
            monitor,
            pre_order.side,
            submission.order_lot,
            final_order_price,
        )

        return _FillPhaseResult(
            filled=filled,
            fill_price=fill_price,
            queue_wait=queue_wait,
            cancel_reason_poll=cancel_reason_poll,
            reprice_count=monitor.reprice_count,
            reprice_drift_bps=monitor.reprice_drift_bps,
            effective_timeout=monitor.effective_timeout,
            cancel_failed_likely_filled=monitor.cancel_failed_likely_filled,
            pending_reconciliation=pending_reconciliation,
            order_price=final_order_price,
            requote_attempts=requote_attempts,
            micro_partial_qty=micro_partial_qty,
        )

    async def _finalize_cycle(
        self,
        *,
        pre_order: _PreOrderPhaseResult,
        submission: _SubmissionPhaseResult,
        fill_phase: _FillPhaseResult,
        sidecar_offset_bps: float,
        sidecar_bias: float | None,
        sidecar_confidence: float | None,
        sidecar_model_version: str | None,
        sidecar_signal_status: str | None,
    ) -> "FillRecord":
        """Task C: post-fill phase (PnL/record/log) をまとめる."""
        pnl = await self._measure_post_fill_pnl(
            fill_phase.filled,
            fill_phase.fill_price,
            pre_order.side,
        )
        if fill_phase.filled:
            self._maker_price.update_inventory(pre_order.side)

        regime_str: str | None = None
        regime_conf: float | None = None
        regime_stab: int | None = None
        regime_trend_pct: float | None = None
        regime_vol_ratio: float | None = None
        if self._regime_detector is not None:
            regime_price = pnl.mid_at_fill
            if regime_price is None:
                regime_price, _ = self._maker_price.get_fallback_price()
            if regime_price is not None:
                regime_result = self._regime_detector.update(submission.t_submit, regime_price)
                regime_str = regime_result.regime.value
                regime_conf = regime_result.confidence
                regime_stab = regime_result.stability
                regime_trend_pct = regime_result.trend_pct
                regime_vol_ratio = regime_result.volatility_ratio

        macro_trend: str | None = None
        macro_slope_5m: float | None = None
        macro_slope_15m: float | None = None
        macro_aligned: bool | None = None
        if self._macro_regime_detector is not None:
            macro_price = pnl.mid_at_fill
            if macro_price is None:
                macro_price, _ = self._maker_price.get_fallback_price()
            if macro_price is not None:
                from scripts.v460.lib.macro_regime import compose_regimes

                macro_result = self._macro_regime_detector.update(submission.t_submit, macro_price)
                macro_trend = macro_result.trend.value
                macro_slope_5m = macro_result.slope_5m_bps_per_min
                macro_slope_15m = macro_result.slope_15m_bps_per_min
                if regime_str is not None:
                    _, macro_aligned = compose_regimes(
                        regime_str,
                        regime_conf or 0.0,
                        macro_result,
                    )
                    if not macro_aligned:
                        action = self.config.macro_regime_conflict_action
                        if action == "downgrade":
                            original_regime = regime_str
                            regime_str = "ranging"
                            logger.info(
                                "[macro_regime] micro/macro conflict -> ranging downgrade "
                                "(micro=%s, macro=%s)",
                                original_regime,
                                macro_trend,
                            )
                        else:
                            logger.debug(
                                "[macro_regime] micro/macro conflict detected "
                                "(micro=%s, macro=%s, aligned=False)",
                                regime_str,
                                macro_trend,
                            )
                self._last_macro_trend = macro_trend

        decision_path = self._derive_decision_path(
            ev_score_pretrade=pre_order.ev_score_pretrade,
            skip_gate_reason=pre_order.skip_gate_reason,
            ev_offset_applied=pre_order.ev_offset_applied,
        )

        record = self._build_fill_record(
            cycle_id=pre_order.cycle_id,
            t_submit=submission.t_submit,
            side=pre_order.side,
            order_price=fill_phase.order_price,
            order_lot=submission.order_lot,
            fill_price=fill_phase.fill_price,
            filled=fill_phase.filled,
            spread_at_order=pre_order.spread_at_order,
            effective_offset_ratio=pre_order.effective_offset_ratio,
            queue_wait=fill_phase.queue_wait,
            cancel_reason_poll=fill_phase.cancel_reason_poll,
            reprice_count=fill_phase.reprice_count,
            reprice_drift_bps=fill_phase.reprice_drift_bps,
            effective_timeout=fill_phase.effective_timeout,
            cancel_failed_likely_filled=fill_phase.cancel_failed_likely_filled,
            pnl=pnl,
            sg_skipped=pre_order.skip_gate_skipped,
            sg_score=pre_order.skip_gate_score,
            sg_reason=pre_order.skip_gate_reason,
            sg_model_used=pre_order.skip_gate_model_used,
            sg_as_prob=pre_order.skip_gate_as_prob,
            sg_threshold_used=pre_order.skip_gate_threshold_used,
            sg_hour_offset=pre_order.skip_gate_hour_offset,
            sg_velocity_bps=pre_order.sg_velocity_bps,
            regime_str=regime_str,
            regime_conf=regime_conf,
            regime_stab=regime_stab,
            regime_trend_pct=regime_trend_pct,
            regime_vol_ratio=regime_vol_ratio,
            confidence_factor=submission.confidence_factor,
            regime_lot=pre_order.regime_lot,
            macro_trend=macro_trend,
            macro_slope_5m=macro_slope_5m,
            macro_slope_15m=macro_slope_15m,
            macro_aligned=macro_aligned,
            macro_boost_applied=pre_order.macro_boost_applied or None,
            ev_score_pretrade=pre_order.ev_score_pretrade,
            ev_offset_mult_applied=pre_order.ev_offset_mult_applied,
            decision_path=decision_path,
            sidecar_offset_bps=sidecar_offset_bps if sidecar_offset_bps != 0.0 else None,
            sidecar_bias=sidecar_bias,
            sidecar_confidence=sidecar_confidence,
            sidecar_model_version=sidecar_model_version or None,
            sidecar_signal_status=sidecar_signal_status,
            queue_depth_ahead=submission.queue_depth_ahead,
            queue_fill_prob_est=submission.queue_fill_prob_est,
            regime_at_order=pre_order.regime_at_order,
            regime_observation_count=pre_order.regime_obs_count,
            mid_at_order=pre_order.mid_at_order,
            execution_pre_clamp_offset=pre_order.execution_pre_clamp_offset,
            executor_offset_stages=pre_order.executor_offset_stages_json,
            execution_additive_enabled=self.config.experimental_additive_pipeline,
            log_cycle_no=self._cycle_count,
            # 642# 可観測性
            sg_forced_pass=pre_order.skip_gate_forced_pass,
            sg_side_skip_rate=pre_order.skip_gate_side_skip_rate,
            execution_hard_skip_mult_used=pre_order.execution_hard_skip_mult_used,
            balance_jpy_at_order=self._balance_checker.last_jpy_free,
            balance_btc_at_order=self._balance_checker.last_btc_free,
        )

        self._log_cycle_result(
            filled=fill_phase.filled,
            queue_wait=fill_phase.queue_wait,
            post_fill_pnl=pnl.post_fill_pnl,
            cancel_reason=fill_phase.cancel_reason_poll,
            sidecar_offset_bps=sidecar_offset_bps,
            sidecar_signal_status=sidecar_signal_status,
            order_id=self._pending_order_id,
            regime=(
                self._regime_detector.current_regime.value
                if self._regime_detector else "n/a"
            ),
        )
        self._log_cycle_revenue_context(record)
        if fill_phase.pending_reconciliation:
            record.pending_reconciliation = True
        if self.config.micro_timeout_enabled:
            record.requote_attempts = fill_phase.requote_attempts
            if fill_phase.micro_partial_qty > 0:
                record.micro_timeout_partial_filled_qty = fill_phase.micro_partial_qty

        await self._circuit_breaker.async_on_success()
        return record

    async def run_single_cycle(
        self,
        side_override: str | None = None,
        one_sided_balance: bool = False,
        trending_offset_mult: float | None = None,
        degraded_liquidation: bool = False,
        toxicity_offset_mult: float = 1.0,
        sidecar_offset_bps: float = 0.0,
        sidecar_bias: float | None = None,
        # 487# P0: sidecar attribution 可観測性
        sidecar_confidence: float | None = None,
        sidecar_model_version: str | None = None,
        sidecar_signal_status: str | None = None,
    ) -> FillRecord:
        """1 サイクル: 発注 → 監視 → 結果記録."""
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
        pre_order = await self._run_pre_order_phase(
            cycle_id=cycle_id,
            side_override=side_override,
            one_sided_balance=one_sided_balance,
            trending_offset_mult=trending_offset_mult,
            degraded_liquidation=degraded_liquidation,
            toxicity_offset_mult=toxicity_offset_mult,
            sidecar_offset_bps=sidecar_offset_bps,
        )
        if not isinstance(pre_order, _PreOrderPhaseResult):
            return pre_order

        submission = await self._submit_order_phase(
            pre_order=pre_order,
            one_sided_balance=one_sided_balance,
        )
        if not isinstance(submission, _SubmissionPhaseResult):
            return submission

        fill_phase = await self._monitor_fill_phase(
            pre_order=pre_order,
            submission=submission,
        )
        return await self._finalize_cycle(
            pre_order=pre_order,
            submission=submission,
            fill_phase=fill_phase,
            sidecar_offset_bps=sidecar_offset_bps,
            sidecar_bias=sidecar_bias,
            sidecar_confidence=sidecar_confidence,
            sidecar_model_version=sidecar_model_version,
            sidecar_signal_status=sidecar_signal_status,
        )
