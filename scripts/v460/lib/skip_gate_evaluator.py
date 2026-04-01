"""121# SkipGate ML 評価モジュール.

FillTestRunner から SkipGate 初期化 + 評価ロジックを分離。
062# SkipGate ML フィルター / 088# 動的閾値較正 / 096# warm start を統合。
126# モデル hot-reload: retrain_scheduler によるモデル差替を自動検出・ロード。

461# リファクタ: モデル管理を skip_gate_model_loader.py,
ev_weighted 判定を skip_gate_ev_weighted.py に Mixin 抽出。

責務:
  - SkipGate の初期化 (Mixin チェーン統合)
  - 特徴量構築 (build_features_from_market_state)
  - evaluate() 呼び出し + 早期リターン FillRecord 生成
  - Protocol 定義 (SkipGateAdapter 等)

MAX LINES: 900
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.fill_config import FillTestConfig, SkipGateResult
from scripts.v460.lib.hour_rules import resolve_hour_float, utc_hour_from_timestamp
from scripts.v460.lib.skip_gate_ev_weighted import SkipGateEvWeightedMixin
from scripts.v460.lib.skip_gate_model_loader import SkipGateModelLoaderMixin
from ztb.ml.skip_gate_runtime import get_trade_field, normalize_recent_trades
from ztb.ml.skip_gate_fill_record import (
    SkipFillRecordContext as _SkipFillRecordContext,
    build_skip_fill_record_from_context,
)
from ztb.ml.skip_gate_result_fields import (
    SkipFillRecordExtraFields,
    build_skip_decision_result_fields,
    build_skip_fill_record_extra_fields,
)
from ztb.ml.skip_gate import SkipDecision, SkipGate, build_features_from_market_state
from ztb.ml.skip_gate_contracts import (
    SkipGateAdapter,
    SkipGateClassLike as _SkipGateClassLike,
    SkipGateConfigLike as _SkipGateConfigLike,
    SkipDecisionLike as _SkipDecisionLike,
    SkipGateLike as _SkipGateLike,
)
# ファイルの SHA256 ハッシュを算出 (126# hot-reload 用)
from ztb.trading.live.exchanges.base.broker_interfaces import OrderBookSnapshot
from ztb.utils.run_manifest import compute_file_hash

if TYPE_CHECKING:
    from ztb.metrics.fill_quality import FillRecord

logger = logging.getLogger(__name__)


class SkipGateEvaluator(SkipGateModelLoaderMixin, SkipGateEvWeightedMixin):
    """SkipGate ML フィルター (062# / 088# / 096# / 126# 統合).

    461# Mixin 構成:
      - SkipGateModelLoaderMixin: モデルロード / hot-reload / config override
      - SkipGateEvWeightedMixin: ev_weighted 統合判定 (188# / 190# / 193#)
    """

    # 126# hot-reload: モデルファイルのチェック間隔 (秒)
    # 158# YAML 外部化: config.hot_reload_check_interval_sec から読み込み (フォールバック用に残す)
    _HOT_RELOAD_CHECK_INTERVAL_SEC = 120.0  # デフォルト値 (config 未設定時のフォールバック)
    # 236# hasattr 排除: クラスレベルでデフォルト値を保証
    # None = __init__ 未実行 (mock/テスト用), 0.0 = 初期化済み
    _last_reload_check: float | None = None
    _SIDE_MODEL_SLOTS: tuple[tuple[str, str, str, str], ...] = (
        ("buy", "_gate_buy", "_gate_path_buy", "_model_file_hash_buy"),
        ("sell", "_gate_sell", "_gate_path_sell", "_model_file_hash_sell"),
    )
    # 188# C-1: ev_weighted 副 horizon モデルスロット
    _ALT_MODEL_SLOTS: tuple[tuple[str, str, str, str, str], ...] = (
        ("buy", "_gate_alt_buy", "_gate_path_alt_buy", "_model_file_hash_alt_buy", "skip_gate_model_path_buy_long"),
        ("sell", "_gate_alt_sell", "_gate_path_alt_sell", "_model_file_hash_alt_sell", "skip_gate_model_path_sell_short"),
    )

    def __init__(
        self,
        config: FillTestConfig,
        project_root: Path,
    ) -> None:
        self._config = config
        self._project_root = project_root
        self._skip_gate: _SkipGateLike | None = None  # SkipGate instance
        # 141# P1-01: side 別 SkipGate インスタンス (フォールバック用に unified も保持)
        self._gate_buy: _SkipGateLike | None = None
        self._gate_sell: _SkipGateLike | None = None
        self._gate_path_buy: Path | None = None
        self._gate_path_sell: Path | None = None
        self._model_file_hash_buy: str = ""
        self._model_file_hash_sell: str = ""
        # 188# C-1: ev_weighted 副 horizon モデルスロット
        self._gate_alt_buy: _SkipGateLike | None = None
        self._gate_alt_sell: _SkipGateLike | None = None
        self._gate_path_alt_buy: Path | None = None
        self._gate_path_alt_sell: Path | None = None
        self._model_file_hash_alt_buy: str = ""
        self._model_file_hash_alt_sell: str = ""
        # 190# A: ev_weighted 連続 skip 安全弁カウンタ
        self._ev_consecutive_skip_count: int = 0
        # 596# Primary model 連続 skip 安全弁カウンタ (evaluator-level)
        self._primary_consecutive_skip_count: int = 0
        # 657# A-5: toxic_sell_veto 連続 veto カウンタ (時間減衰用)
        self._toxic_veto_consecutive_count: int = 0
        # 156# D-1: OB fetch 失敗カウンタ
        self._ob_fetch_fail_count: int = 0
        self._ob_fetch_total_count: int = 0
        # 126# hot-reload 状態
        self._gate_path: Path | None = None
        self._model_file_hash: str = ""
        self._last_reload_check: float = 0.0

        if not config.skip_gate_enabled:
            return

        try:
            skip_gate_cls = cast(_SkipGateClassLike, SkipGate)

            gate_path = self._resolve_model_path(config.skip_gate_model_path)
            if not gate_path.exists():
                logger.warning(
                    f"[skip_gate] Model not found: {gate_path}. "
                    f"SkipGate disabled (unified). Trying side models..."
                )
                # 143# A.1 #2: unified 不在でも side モデルのロードを試行
                self._load_side_models(skip_gate_cls)
                # 188# C-1: ev_weighted 副 horizon モデルロード
                self._load_alt_models(skip_gate_cls)
                self._last_reload_check = time.monotonic()
                return

            self._gate_path = gate_path
            skip_gate = self._load_gate_from_path(
                skip_gate_cls,
                gate_path,
                apply_warm_start=True,
            )
            self._skip_gate = skip_gate
            self._model_file_hash = self._read_model_hash(gate_path)
            self._last_reload_check = time.monotonic()

            # 141# P1-01: side 別モデルロード
            self._load_side_models(skip_gate_cls)
            # 188# C-1: ev_weighted 副 horizon モデルロード
            self._load_alt_models(skip_gate_cls)
        except Exception as e:
            logger.error(f"[skip_gate] Failed to load: {e}. SkipGate disabled.")
            self._skip_gate = None

    # --- 461# Mixin 移管済: _resolve_model_path, _read_model_hash → skip_gate_model_loader.py ---

    @staticmethod
    def _compute_velocity_offset_multiplier(
        *,
        observed_velocity_bps: float,
        threshold_bps: float,
        base_multiplier: float,
        max_multiplier: float,
        proportional: bool,
    ) -> tuple[float, bool]:
        """velocity soft mode の offset 乗数を安全に解決する.

        200# L: velocity_math.compute_velocity_offset_multiplier への委譲 (SSOT).
        後方互換のため static method wrapper を維持。
        """
        from scripts.v460.lib.velocity_math import compute_velocity_offset_multiplier
        return compute_velocity_offset_multiplier(
            observed_velocity_bps=observed_velocity_bps,
            threshold_bps=threshold_bps,
            base_multiplier=base_multiplier,
            max_multiplier=max_multiplier,
            proportional=proportional,
        )

    # --- 461# Mixin 移管済: _load_gate_from_path → skip_gate_model_loader.py ---

    @staticmethod
    def _get_trade_field(
        trade: object,
        *,
        key: str,
        fallback_key: str | None = None,
        default: object = None,
    ) -> object:
        """dict / object 両対応で trade field を取得する."""
        return get_trade_field(
            trade,
            key=key,
            fallback_key=fallback_key,
            default=default,
        )

    @classmethod
    def _normalize_recent_trades(
        cls,
        trades: object,
        *,
        fallback_timestamp: float,
    ) -> list[dict[str, object]] | None:
        """adapter の recent_trades を skip_gate 用の共通形式へ正規化."""
        del cls
        return normalize_recent_trades(
            trades,
            fallback_timestamp=fallback_timestamp,
        )

    @staticmethod
    def _make_skip_fill_record(
        *,
        context: _SkipFillRecordContext,
        extra_fields: SkipFillRecordExtraFields,
    ) -> "FillRecord":
        """skip 系の early return 用 FillRecord を共通生成."""
        return build_skip_fill_record_from_context(
            context=context,
            extra_fields=extra_fields,
        )

    @staticmethod
    def _build_skip_fill_record_context(
        *,
        cycle_id: str,
        timestamp: float,
        side: str,
        order_price: float,
        order_quantity: float,
        cancel_reason: str,
        spread_at_order: float | None,
        spread_offset_ratio: float,
        run_id: str,
        git_sha: str | None,
        regime_value: str | None,
        decision_trace_id: str | None,
    ) -> _SkipFillRecordContext:
        """Build the v460-specific core context for early skip FillRecord output."""
        return _SkipFillRecordContext(
            cycle_id=cycle_id,
            timestamp=timestamp,
            side=side,
            order_price=order_price,
            order_quantity=order_quantity,
            cancel_reason=cancel_reason,
            spread_at_order=spread_at_order,
            spread_offset_ratio=spread_offset_ratio,
            run_id=run_id,
            git_sha=git_sha,
            regime_value=regime_value,
            decision_trace_id=decision_trace_id,
        )

    @staticmethod
    def _assign_result_fields(
        result: SkipGateResult,
        *,
        skipped: bool,
        score: float,
        reason: str,
        model_used: str,
        as_prob: float | None = None,
        threshold_used: float | None = None,
        hour_offset: float = 0.0,
        price_velocity_bps: float | None = None,
        forced_pass: bool = False,
        side_skip_rate: float | None = None,
    ) -> None:
        """SkipGateResult の共通フィールドを一括設定する."""
        result.skipped = skipped
        result.score = score
        result.reason = reason
        result.model_used = model_used
        result.as_prob = as_prob
        result.threshold_used = threshold_used
        result.hour_offset = hour_offset
        result.price_velocity_bps = price_velocity_bps
        result.forced_pass = forced_pass
        result.side_skip_rate = side_skip_rate

    def _apply_decision_to_result(
        self,
        result: SkipGateResult,
        *,
        decision: _SkipDecisionLike,
        side: str,
        price_velocity_bps: float | None,
        hour_offset: float,
    ) -> None:
        """最終 decision を SkipGateResult へ反映する."""
        is_side_model = (
            (side == "buy" and self._gate_buy is not None)
            or (side == "sell" and self._gate_sell is not None)
        )
        fields = build_skip_decision_result_fields(
            decision,
            side=side,
            has_side_specific_model=is_side_model,
            hour_offset=hour_offset,
            price_velocity_bps=price_velocity_bps,
        )
        self._assign_result_fields(
            result,
            skipped=fields.skipped,
            score=fields.score,
            reason=fields.reason,
            model_used=fields.model_used,
            as_prob=fields.as_prob,
            threshold_used=fields.threshold_used,
            hour_offset=fields.hour_offset,
            price_velocity_bps=fields.price_velocity_bps,
            forced_pass=fields.forced_pass,
            side_skip_rate=fields.side_skip_rate,
        )

    def _set_early_skip_result(
        self,
        result: SkipGateResult,
        *,
        context: _SkipFillRecordContext,
        score: float,
        reason: str,
        model_used: str,
        last_imbalance: float | None,
        last_bid_depth: float | None,
        last_ask_depth: float | None,
        as_prob: float | None = None,
        threshold_used: float | None = None,
        hour_offset: float | None = None,
        price_velocity_bps: float | None = None,
        trend_5s_guard_triggered: bool | None = None,
        trend_5s_guard_action: str | None = None,
        trend_5s_at_order: float | None = None,
        ev_score_pretrade: float | None = None,
        decision_path: str | None = None,
    ) -> None:
        """Populate SkipGateResult and early-return FillRecord together.

        This keeps the v460-specific FillRecord assembly local while reducing
        duplication around rule-based and final skip returns.
        """
        extra_fields = build_skip_fill_record_extra_fields(
            score=score,
            reason=reason,
            model_used=model_used,
            orderbook_imbalance=last_imbalance,
            bid_depth_total=last_bid_depth,
            ask_depth_total=last_ask_depth,
            as_prob=as_prob,
            threshold_used=threshold_used,
            hour_offset=hour_offset,
            price_velocity_bps=price_velocity_bps,
            trend_5s_guard_triggered=trend_5s_guard_triggered,
            trend_5s_guard_action=trend_5s_guard_action,
            trend_5s_at_order=trend_5s_at_order,
            ev_score_pretrade=ev_score_pretrade,
            decision_path=decision_path,
        )
        self._assign_result_fields(
            result,
            skipped=True,
            score=score,
            reason=reason,
            model_used=model_used,
            as_prob=as_prob,
            threshold_used=threshold_used,
            hour_offset=hour_offset or 0.0,
            price_velocity_bps=price_velocity_bps,
        )
        result.early_return_record = self._make_skip_fill_record(
            context=context,
            extra_fields=extra_fields,
        )

    def _apply_trend_5s_sell_guard(
        self,
        *,
        side: str,
        trend_5s_bps: float | None,
    ) -> tuple[str, float | None]:
        """684# Phase M1: independent short-horizon sell guard."""
        cfg = self._config.trend_5s_sell_guard
        if side != "sell" or not cfg.enabled or trend_5s_bps is None:
            return "none", None
        if trend_5s_bps > cfg.hard_veto_threshold_bps:
            return "veto", None
        if trend_5s_bps > cfg.threshold_bps:
            return "boost", cfg.offset_boost_factor
        return "none", None

    # --- 461# Mixin 移管済 ---
    # _apply_config_overrides, _apply_warm_start, _inject_calibrator → skip_gate_model_loader.py
    # _load_side_models, _load_alt_models → skip_gate_model_loader.py
    # _check_and_reload_model, _check_and_reload_side_models → skip_gate_model_loader.py
    # _try_ev_weighted_decision, _ev_weighted_as_offset → skip_gate_ev_weighted.py

    @property
    def skip_gate(self) -> _SkipGateLike | None:
        """内部 SkipGate インスタンスへのアクセス (OrderMonitor 等で使用)."""
        return self._skip_gate

    @property
    def ob_fetch_stats(self) -> tuple[int, int]:
        """156# §18: OB fetch 統計 (fail_count, total_count)."""
        return self._ob_fetch_fail_count, self._ob_fetch_total_count

    def _select_gate_for_side(self, side: str) -> _SkipGateLike | None:
        """141# P1-01: side に適合する SkipGate を返す.

        side 別モデルが存在する場合はそちらを優先し、
        なければ統一モデルにフォールバック。
        """
        # 255# getattr → 直接参照 (__init__ で _gate_buy/_gate_sell 宣言済み)
        if side == "buy" and self._gate_buy is not None:
            return self._gate_buy
        if side == "sell" and self._gate_sell is not None:
            return self._gate_sell
        return self._skip_gate

    async def evaluate(
        self,
        side: str,
        cycle_id: str,
        order_price: float,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        adapter: SkipGateAdapter,
        symbol: str,
        current_lot: float,
        run_id: str,
        git_sha: str | None,
        regime_value: str | None,
        last_imbalance: float | None,
        last_bid_depth: float | None,
        last_ask_depth: float | None,
        imbalance_enabled: bool,
        maker_price_vpin_setter: object | None = None,
        mid_trend_bps: float | None = None,
        *,
        decision_trace_id: str | None = None,
        one_sided_balance: bool = False,
        kill_release_offset: float = 0.0,
        prefetched_ob: OrderBookSnapshot | None = None,
        prefetched_trades: object | None = None,
    ) -> SkipGateResult:
        """062# SkipGate ML 判定.

        Args:
            maker_price_vpin_setter: callable(vpin) — MakerPriceCalculator._last_vpin を設定
            prefetched_ob: 355# L-1 OB prefetch 共有 — サイクル先頭の
                imbalance pre-fetch で取得済みの depth=5 OB を再利用。
                None の場合は従来通り adapter から fetch。
            prefetched_trades: 355# L-2 Trades prefetch 共有 — サイクル先頭の
                trades recorder 向け fetch (limit=100) の結果を再利用。
                None の場合は従来通り adapter から fetch。
        """
        result = SkipGateResult()
        # 143# A.1 #2: unified 不在でも side モデルが使える場合は続行
        # 255# getattr → 直接参照 (__init__ で宣言済み)
        if self._skip_gate is None and self._gate_buy is None and self._gate_sell is None:
            return result

        # 126# hot-reload: モデルファイル変更を検出してリロード
        self._check_and_reload_model()

        # 124# Rule: unknown regime での sell スキップ
        # WF結果: S20%_30=+0.198, S20%_120=+0.140 (両 horizon 改善)
        if (
            self._config.skip_sell_unknown_regime
            and side == "sell"
            and (regime_value is None or regime_value == "unknown")
        ):
            event_ts = time.time()
            early_context = self._build_skip_fill_record_context(
                cycle_id=cycle_id,
                timestamp=event_ts,
                side=side,
                order_price=order_price,
                order_quantity=current_lot,
                cancel_reason=CR.SKIP_GATE_RULE_UNKNOWN_SELL,
                spread_at_order=spread_at_order,
                spread_offset_ratio=effective_offset_ratio,
                run_id=run_id,
                git_sha=git_sha,
                regime_value=regime_value,
                decision_trace_id=decision_trace_id,
            )
            logger.info(
                "[dt=%s] [skip_gate] SKIP: sell in unknown regime "
                "(124# rule_skip_unknown_sell)",
                decision_trace_id or "n/a",
            )
            self._set_early_skip_result(
                result,
                context=early_context,
                score=0.0,
                reason="rule_skip_unknown_sell",
                model_used="rule",
                last_imbalance=last_imbalance,
                last_bid_depth=last_bid_depth,
                last_ask_depth=last_ask_depth,
            )
            return result

        # side 固有モデルのみ存在し、要求 side にはモデルがないケースを明示処理
        active_gate = self._select_gate_for_side(side)
        if active_gate is None:
            self._assign_result_fields(
                result,
                skipped=False,
                score=0.0,
                reason=f"no_model_for_side:{side}",
                model_used="none",
            )
            logger.info(
                f"[skip_gate] No model available for side={side}. "
                f"unified={'yes' if self._skip_gate is not None else 'no'}"
            )
            return result

        try:
            sg_regime = regime_value or "unknown"
            market_ts = time.time()

            # 直近約定データ取得
            # 355# L-2: prefetched_trades があればAPI呼出しをスキップ
            recent_trades_data: list[dict[str, object]] | None = None
            try:
                if prefetched_trades is not None:
                    trades = prefetched_trades
                else:
                    # 265# Protocol 型安全化: getattr → 直接呼び出し
                    trades = await adapter.get_recent_trades(
                        symbol,
                        limit=self._config.skip_gate_recent_trades_limit,
                    )
                recent_trades_data = self._normalize_recent_trades(
                    trades,
                    fallback_timestamp=market_ts,
                )
            except Exception as exc:
                logger.debug("Trades formatting failed: %s", exc)

            # 072# OB トグル
            ob_bid: float | None = None
            ob_ask: float | None = None
            ob_bid_vol: float | None = None
            ob_ask_vol: float | None = None
            # 141# P1-01: side 別モデルの use_ob_features を参照
            sg_use_ob = active_gate.config.use_ob_features
            if sg_use_ob:
                self._ob_fetch_total_count += 1
                try:
                    from scripts.v460.lib.ob_utils import extract_price, depth_volume
                    # 355# L-1: prefetched_ob があればAPI呼出しをスキップ
                    if prefetched_ob is not None:
                        ob = prefetched_ob
                    else:
                        # 265# Protocol 型安全化: getattr → 直接呼び出し
                        ob = await adapter.get_orderbook(
                            symbol,
                            depth=self._config.skip_gate_ob_depth,
                        )
                    if ob and hasattr(ob, "bids") and hasattr(ob, "asks"):
                        # 266# type:ignore 排除: OrderBookSnapshot Protocol 型安全化
                        bids = ob.bids
                        asks = ob.asks
                        # 145# §9-#3: tuple/object 両対応 (ob_utils 使用)
                        ob_bid = extract_price(bids[0])
                        ob_ask = extract_price(asks[0])
                        ob_bid_vol = depth_volume(bids, self._config.skip_gate_ob_depth)
                        ob_ask_vol = depth_volume(asks, self._config.skip_gate_ob_depth)
                except Exception as e:
                    self._ob_fetch_fail_count += 1
                    # 156# D-1: 初回 + 10回毎に warning、それ以外は debug
                    if self._ob_fetch_fail_count == 1 or self._ob_fetch_fail_count % 10 == 0:
                        logger.warning(
                            "[skip_gate] OB fetch failed (%d times): %s",
                            self._ob_fetch_fail_count,
                            e,
                        )
                    else:
                        logger.debug("[skip_gate] OB fetch failed: %s", e)

            gate_features = build_features_from_market_state(
                side=side,
                spread_jpy=spread_at_order or 0.0,
                offset_ratio=effective_offset_ratio,
                regime=sg_regime,
                recent_trades=recent_trades_data,
                market_timestamp=market_ts,
                best_bid=ob_bid,
                best_ask=ob_ask,
                bid_vol_5=ob_bid_vol,
                ask_vol_5=ob_ask_vol,
                use_ob_features=sg_use_ob,
            )

            # 107# Volatility Guard: VPIN キャッシュ
            # 366# M5: Volume-Sync VPIN (Easley 2012) — vpin_vol_sync_enabled 時は
            # 出来高バケット VPIN で置換。低出来高時のノイズ耐性が向上。
            vpin_value = gate_features.get("vpin_60s")
            if (
                self._config.vpin_vol_sync_enabled
                and recent_trades_data
                and len(recent_trades_data) >= 5
            ):
                try:
                    vpin_value = self._compute_vol_sync_vpin(recent_trades_data)
                    gate_features["vpin_60s"] = vpin_value
                except Exception as e:
                    logger.debug("[skip_gate] vol_sync VPIN failed, using fallback: %s", e)
            if maker_price_vpin_setter is not None and callable(maker_price_vpin_setter):
                maker_price_vpin_setter(vpin_value)

            # 165# AS-R1: velocity-based pre-ML sell/buy skip rule
            # 195# ソフト化: velocity_skip_as_offset_enabled 時は hard skip せず offset boost
            _pv60 = gate_features.get("price_velocity_bps", 0.0)
            _velocity_sell_triggered = (
                self._config.sell_velocity_skip_enabled
                and side == "sell"
                and _pv60 > self._config.sell_velocity_skip_threshold_bps
            )
            _velocity_buy_triggered = (
                self._config.buy_velocity_skip_enabled
                and side == "buy"
                and _pv60 < self._config.buy_velocity_skip_threshold_bps
            )

            if _velocity_sell_triggered or _velocity_buy_triggered:
                _vel_label = "sell" if _velocity_sell_triggered else "buy"
                _vel_th = (
                    self._config.sell_velocity_skip_threshold_bps
                    if _velocity_sell_triggered
                    else self._config.buy_velocity_skip_threshold_bps
                )

                if self._config.velocity_skip_as_offset_enabled:
                    # 195# ソフトモード: skip せず offset boost 倍率を記録
                    # 196# 比例モード: 閾値超過量に比例した段階的 boost
                    _boost, _is_proportional = self._compute_velocity_offset_multiplier(
                        observed_velocity_bps=_pv60,
                        threshold_bps=_vel_th,
                        base_multiplier=self._config.velocity_offset_boost_factor,
                        max_multiplier=self._config.velocity_offset_max_mult,
                        proportional=self._config.velocity_offset_proportional,
                    )
                    result.velocity_offset_mult = _boost
                    result.price_velocity_bps = _pv60
                    logger.info(
                        f"[skip_gate] 195# velocity→offset: {_vel_label} "
                        f"velocity={_pv60:.2f}bps (th={_vel_th}) "
                        f"→ offset_mult={_boost:.2f}"
                        f"{' (proportional)' if _is_proportional else ''}"
                    )
                    # hard skip しない → ML 判定に進む
                else:
                    # 旧モード: hard skip
                    _reason = f"rule_velocity_{_vel_label}_skip"
                    _cancel = (
                        CR.SKIP_GATE_RULE_VELOCITY_SELL
                        if _velocity_sell_triggered
                        else CR.SKIP_GATE_RULE_VELOCITY_BUY
                    )
                    logger.info(
                        "[dt=%s] [skip_gate] SKIP: %s velocity %.2fbps %s %.2fbps "
                        "(165# AS-R1 %s)",
                        decision_trace_id or "n/a",
                        _vel_label,
                        _pv60,
                        ">" if _velocity_sell_triggered else "<",
                        _vel_th,
                        _reason,
                    )
                    early_context = self._build_skip_fill_record_context(
                        cycle_id=cycle_id,
                        timestamp=market_ts,
                        side=side,
                        order_price=order_price,
                        order_quantity=current_lot,
                        cancel_reason=_cancel,
                        spread_at_order=spread_at_order,
                        spread_offset_ratio=effective_offset_ratio,
                        run_id=run_id,
                        git_sha=git_sha,
                        regime_value=regime_value,
                        decision_trace_id=decision_trace_id,
                    )
                    self._set_early_skip_result(
                        result,
                        context=early_context,
                        score=_pv60,
                        reason=_reason,
                        model_used="rule",
                        last_imbalance=last_imbalance,
                        last_bid_depth=last_bid_depth,
                        last_ask_depth=last_ask_depth,
                        price_velocity_bps=_pv60,
                    )
                    return result

            # 654# P0-2: Toxic Low-Spread Sell Veto (651#/652# compound guard)
            # 657# A-4: ソフト化 — 全条件充足時もoffset boostモード選択可
            # 657# A-5: 時間減衰 — 連続 veto で α^n 減衰、sticky 防止
            # Glosten-Milgrom: 狭spreading + 高VPIN + OBI偏り + 上方加速 = informed flow
            if (
                self._config.toxic_sell_veto_enabled
                and side == "sell"
                and order_price > 0
                and spread_at_order is not None
            ):
                _spread_bps = (spread_at_order / order_price) * 10000.0
                _obi = last_imbalance if last_imbalance is not None else 0.0
                _vpin = vpin_value if vpin_value is not None else 0.0
                _vel = _pv60 if isinstance(_pv60, (int, float)) else 0.0
                _vel_th = self._config.toxic_sell_veto_velocity_threshold

                # 各条件を個別評価
                _cond_spread = _spread_bps < self._config.toxic_sell_veto_spread_bps
                _cond_obi = _obi > self._config.toxic_sell_veto_obi_threshold
                _cond_vpin = _vpin > self._config.toxic_sell_veto_vpin_threshold
                _cond_vel = (_vel_th <= 0.0 or _vel > _vel_th)

                if _cond_spread and _cond_obi and _cond_vpin and _cond_vel:
                    # 全条件充足
                    self._toxic_veto_consecutive_count += 1

                    # 657# A-5: 時間減衰 — 連続 veto が続くほどboostのみに軟化
                    _decay = self._config.toxic_sell_veto_decay_alpha ** (
                        self._toxic_veto_consecutive_count - 1
                    )

                    # 分岐: as_offset=true→常にソフト / false→decayが
                    # 50%未満でソフト化へフォールバック / それ以外→hard skip
                    _soft_mode = (
                        self._config.toxic_sell_veto_as_offset_enabled
                        or _decay < 0.5
                    )
                    if _soft_mode:
                        # 657# A-4: ソフトモード — offset boost で保守的に発注
                        _boost = self._config.toxic_sell_veto_offset_boost_factor
                        # A-5 減衰適用: boost_effective = 1.0 + (boost - 1.0) * decay
                        _boost_eff = 1.0 + (_boost - 1.0) * _decay
                        result.toxic_veto_offset_mult = _boost_eff
                        logger.info(
                            "[skip_gate] 657# toxic_veto→offset: "
                            "spread=%.2fbps OBI=%.3f VPIN=%.3f vel=%.2fbps "
                            "→ offset_mult=%.2f (decay=%.2f, consec=%d)",
                            _spread_bps, _obi, _vpin, _vel,
                            _boost_eff, _decay,
                            self._toxic_veto_consecutive_count,
                        )
                        # hard skip しない → ML 判定に進む
                    else:
                        # 旧モード: hard veto
                        logger.info(
                            "[dt=%s] [skip_gate] 654# VETO: toxic low-spread sell "
                            "spread=%.2fbps OBI=%.3f VPIN=%.3f vel=%.2fbps "
                            "(consec=%d, decay=%.2f)",
                            decision_trace_id or "n/a",
                            _spread_bps, _obi, _vpin, _vel,
                            self._toxic_veto_consecutive_count, _decay,
                        )
                        early_context = self._build_skip_fill_record_context(
                            cycle_id=cycle_id,
                            timestamp=market_ts,
                            side=side,
                            order_price=order_price,
                            order_quantity=current_lot,
                            cancel_reason=CR.TOXIC_LOW_SPREAD_SELL_VETO,
                            spread_at_order=spread_at_order,
                            spread_offset_ratio=effective_offset_ratio,
                            run_id=run_id,
                            git_sha=git_sha,
                            regime_value=regime_value,
                            decision_trace_id=decision_trace_id,
                        )
                        self._set_early_skip_result(
                            result,
                            context=early_context,
                            score=_spread_bps,
                            reason="rule_toxic_low_spread_sell_veto",
                            model_used="rule",
                            last_imbalance=last_imbalance,
                            last_bid_depth=last_bid_depth,
                            last_ask_depth=last_ask_depth,
                            price_velocity_bps=(
                                _pv60 if isinstance(_pv60, (int, float)) else None
                            ),
                        )
                        return result
                else:
                    # 条件不充足 → 連続カウンタリセット
                    self._toxic_veto_consecutive_count = 0

            # 684# Phase M1: trend_5s sell guard (independent from velocity/toxic veto).
            _trend_guard_action, _trend_guard_mult = self._apply_trend_5s_sell_guard(
                side=side,
                trend_5s_bps=mid_trend_bps,
            )
            result.trend_5s_at_order = mid_trend_bps
            result.trend_5s_guard_action = _trend_guard_action
            result.trend_5s_guard_triggered = _trend_guard_action in {"boost", "veto"}
            result.trend_5s_guard_offset_mult = _trend_guard_mult

            if _trend_guard_action == "veto":
                logger.info(
                    "[dt=%s] [skip_gate] 684# trend_5s veto: sell trend_5s=%.2fbps > %.2fbps",
                    decision_trace_id or "n/a",
                    mid_trend_bps,
                    self._config.trend_5s_sell_guard_hard_veto_threshold_bps,
                )
                early_context = self._build_skip_fill_record_context(
                    cycle_id=cycle_id,
                    timestamp=market_ts,
                    side=side,
                    order_price=order_price,
                    order_quantity=current_lot,
                    cancel_reason=CR.TREND_5S_SELL_GUARD_VETO,
                    spread_at_order=spread_at_order,
                    spread_offset_ratio=effective_offset_ratio,
                    run_id=run_id,
                    git_sha=git_sha,
                    regime_value=regime_value,
                    decision_trace_id=decision_trace_id,
                )
                self._set_early_skip_result(
                    result,
                    context=early_context,
                    score=float(mid_trend_bps or 0.0),
                    reason="rule_trend_5s_sell_guard_veto",
                    model_used="rule",
                    last_imbalance=last_imbalance,
                    last_bid_depth=last_bid_depth,
                    last_ask_depth=last_ask_depth,
                    price_velocity_bps=_pv60 if isinstance(_pv60, (int, float)) else None,
                    trend_5s_guard_triggered=True,
                    trend_5s_guard_action="veto",
                    trend_5s_at_order=mid_trend_bps,
                )
                return result

            # 158# P1-6: 時間帯別 skip_gate 閾値調整
            _utc_hour = utc_hour_from_timestamp(market_ts)
            _hour_offset = resolve_hour_float(
                self._config.skip_gate_hour_offsets,
                _utc_hour,
            )

            # 183# narrow spread adverse guard: spread < threshold → 閾値厳格化
            _spread_offset = 0.0
            _ns_thr = self._config.skip_gate_narrow_spread_threshold_jpy
            if _ns_thr > 0 and spread_at_order is not None and spread_at_order < _ns_thr:
                _spread_offset = self._config.skip_gate_narrow_spread_offset
                if _spread_offset != 0.0:
                    logger.debug(
                        "[skip_gate] 183# narrow spread guard: spread=%.0f < %.0f → offset +%.2f",
                        spread_at_order, _ns_thr, _spread_offset,
                    )

            _total_offset = _hour_offset + _spread_offset + kill_release_offset
            if kill_release_offset != 0.0:
                logger.info(
                    "[skip_gate] 343# kill release grace: offset %+.2f applied",
                    kill_release_offset,
                )

            # 634# P0: ranging + sell の skip 確率を大幅に上げる (閾値を厳格化)
            if side == "sell" and sg_regime == "ranging":
                _sell_ranging_penalty = self._config.skip_gate_sell_ranging_offset
                _total_offset += _sell_ranging_penalty
                logger.debug(f"[634# SkipGate] applied sell/ranging penalty offset: +{_sell_ranging_penalty}")

            # 186# strictness clamp: 過剰な厳格化/緩和を防止 (187# YAML外部化)
            _OFFSET_FLOOR = self._config.skip_gate_offset_floor
            _OFFSET_CEIL = self._config.skip_gate_offset_ceil
            if _total_offset < _OFFSET_FLOOR or _total_offset > _OFFSET_CEIL:
                logger.debug(
                    "[skip_gate] 186# clamp: raw_offset=%.2f → clamped to [%.1f, %.1f]",
                    _total_offset, _OFFSET_FLOOR, _OFFSET_CEIL,
                )
                _total_offset = max(_OFFSET_FLOOR, min(_OFFSET_CEIL, _total_offset))

            # 141# P1-01: side 別モデルにディスパッチ (フォールバック: unified)
            decision = active_gate.evaluate(
                gate_features,
                side=side,
                regime=sg_regime,
                threshold_offset=_total_offset,
            )

            # 188# C-1: ev_weighted — 副 horizon モデルがあれば統合判定
            _ev_combined = self._try_ev_weighted_decision(
                side, gate_features, sg_regime, _total_offset, decision,
                one_sided_balance=one_sided_balance,
            )
            if _ev_combined is not None:
                # 193#: ev_as_offset モードでは primary decision を保持し、
                # ev_score のみ抽出。旧モードでは完全置換。
                if self._config.skip_gate_ev_as_offset_enabled:
                    result.ev_score = _ev_combined.predicted_pnl_bps
                    # emergency skip のみ primary を上書き
                    if _ev_combined.should_skip:
                        decision = _ev_combined
                else:
                    decision = _ev_combined

            # 596# Primary model 連続 skip 安全弁 (evaluator-level).
            # 190# A は ev_as_offset_enabled=True で無効化されるため、
            # mode に依存しない evaluator-level の安全弁を追加。
            _primary_max = self._config.skip_gate_primary_max_consecutive_skip
            if decision.should_skip and _primary_max > 0:
                self._primary_consecutive_skip_count += 1
                if self._primary_consecutive_skip_count >= _primary_max:
                    logger.warning(
                        "[skip_gate] 596# primary consecutive skip safety valve: "
                        "%d consecutive skips >= limit %d — forcing PASS "
                        "(score=%.3f, side=%s)",
                        self._primary_consecutive_skip_count, _primary_max,
                        decision.predicted_pnl_bps, side,
                    )
                    from ztb.ml.skip_gate import SkipDecision as _SD
                    decision = _SD(
                        should_skip=False,
                        predicted_pnl_bps=decision.predicted_pnl_bps,
                        threshold_bps=decision.threshold_bps,
                        features_used=decision.features_used,
                        reason="primary_safety_valve_pass",
                        model_used=decision.model_used,
                        as_probability=decision.as_probability,
                        threshold_used=decision.threshold_used,
                    )
                    self._primary_consecutive_skip_count = 0
            elif not decision.should_skip:
                self._primary_consecutive_skip_count = 0

            self._apply_decision_to_result(
                result,
                decision=decision,
                side=side,
                price_velocity_bps=gate_features.get("price_velocity_bps"),
                hour_offset=_total_offset,
            )

            if decision.should_skip:
                if self._config.skip_gate_bypass_mode:
                    result.bypassed = True
                    result.skipped = False
                    logger.info(
                        "[dt=%s] [skip_gate] 686# BYPASS: %s skip suppressed "
                        "(score=%.3f, reason=%s, model=%s, features=%d)",
                        decision_trace_id or "n/a",
                        side,
                        result.score,
                        result.reason,
                        result.model_used,
                        decision.features_used,
                    )
                    return result
                logger.info(
                    "[dt=%s] [skip_gate] SKIP: %s order skipped "
                    "(score=%.3f, reason=%s, model=%s, features=%d)",
                    decision_trace_id or "n/a",
                    side,
                    result.score,
                    result.reason,
                    result.model_used,
                    decision.features_used,
                )
                early_context = self._build_skip_fill_record_context(
                    cycle_id=cycle_id,
                    timestamp=market_ts,
                    side=side,
                    order_price=order_price,
                    order_quantity=current_lot,
                    cancel_reason=CR.SKIP_GATE,
                    spread_at_order=spread_at_order,
                    spread_offset_ratio=effective_offset_ratio,
                    run_id=run_id,
                    git_sha=git_sha,
                    regime_value=regime_value,
                    decision_trace_id=decision_trace_id,
                )
                self._set_early_skip_result(
                    result,
                    context=early_context,
                    score=result.score,
                    reason=result.reason,
                    model_used=result.model_used,
                    last_imbalance=last_imbalance,
                    last_bid_depth=last_bid_depth,
                    last_ask_depth=last_ask_depth,
                    as_prob=result.as_prob,
                    threshold_used=result.threshold_used,
                    hour_offset=_total_offset if _total_offset != 0.0 else None,
                    price_velocity_bps=result.price_velocity_bps,
                    ev_score_pretrade=result.ev_score,
                    decision_path=(
                        "ev_emergency_skip"
                        if result.ev_score is not None and "emergency_skip" in result.reason
                        else "ev_normal_skip" if result.ev_score is not None
                        else None
                    ),
                )
            else:
                logger.debug(
                    "[dt=%s] [skip_gate] PASS: %s order allowed "
                    "(score=%.3f, reason=%s, model=%s)",
                    decision_trace_id or "n/a",
                    side,
                    result.score,
                    result.reason,
                    result.model_used,
                )
        except Exception as e:
            logger.warning(f"[skip_gate] Evaluation failed (non-fatal): {e}")
            result.reason = f"error:{e}"

        return result

    # ------------------------------------------------------------------
    # 366# M5: Volume-Sync VPIN ヘルパー
    # ------------------------------------------------------------------
    def _compute_vol_sync_vpin(self, trades: list[dict]) -> float:
        """recent_trades から Volume-Sync VPIN を計算.

        Easley-López de Prado (2012) 出来高バケット VPIN。
        """
        from scripts.v460.lib.vpin_volume_sync import (
            NEUTRAL_VPIN,
            compute_vpin_volume_sync,
        )

        if not trades:
            return NEUTRAL_VPIN

        # 累積配列を構築 (長さ N+1, [0]=0.0)
        n = len(trades)
        cum_total = np.zeros(n + 1, dtype=np.float64)
        cum_buy = np.zeros(n + 1, dtype=np.float64)
        for i, t in enumerate(trades):
            amount = max(float(t.get("amount", 0.0)), 0.0)  # 負値ガード
            cum_total[i + 1] = cum_total[i] + amount
            is_buy = t.get("side", "").lower() in ("buy", "bid")
            cum_buy[i + 1] = cum_buy[i] + (amount if is_buy else 0.0)

        return compute_vpin_volume_sync(
            cumulative_total_volume=cum_total,
            cumulative_buy_volume=cum_buy,
            end_index=n,
            bucket_size=self._config.vpin_vol_sync_bucket_btc,
            n_buckets=self._config.vpin_vol_sync_n_buckets,
        )
