"""121# SkipGate ML 評価モジュール.

FillTestRunner から SkipGate 初期化 + 評価ロジックを分離。
062# SkipGate ML フィルター / 088# 動的閾値較正 / 096# warm start を統合。
126# モデル hot-reload: retrain_scheduler によるモデル差替を自動検出・ロード。

責務:
  - SkipGate モデルの読み込み・設定オーバーライド
  - 特徴量構築 (build_features_from_market_state)
  - evaluate() 呼び出し + 早期リターン FillRecord 生成
  - 126# モデルファイルの変更検出 + アトミック hot-reload
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from scripts.v460.lib.fill_config import FillTestConfig, SkipGateResult

if TYPE_CHECKING:
    from ztb.metrics.fill_quality import FillRecord

logger = logging.getLogger(__name__)


class SkipGateEvaluator:
    """SkipGate ML フィルター (062# / 088# / 096# / 126# 統合)."""

    # 126# hot-reload: モデルファイルのチェック間隔 (秒)
    _HOT_RELOAD_CHECK_INTERVAL_SEC = 120.0  # 2分 (cycle_interval と同程度)

    def __init__(
        self,
        config: FillTestConfig,
        project_root: Path,
    ) -> None:
        self._config = config
        self._project_root = project_root
        self._skip_gate: object | None = None  # SkipGate instance
        # 126# hot-reload 状態
        self._gate_path: Path | None = None
        self._model_file_hash: str = ""
        self._last_reload_check: float = 0.0

        if not config.skip_gate_enabled:
            return

        try:
            from scripts.v460.ml.skip_gate import SkipGate

            gate_path = Path(config.skip_gate_model_path)
            if not gate_path.is_absolute():
                gate_path = project_root / gate_path
            if not gate_path.exists():
                logger.warning(
                    f"[skip_gate] Model not found: {gate_path}. "
                    f"SkipGate disabled."
                )
                return

            self._gate_path = gate_path
            skip_gate = SkipGate.load(gate_path)
            self._apply_config_overrides(skip_gate)
            self._apply_warm_start(skip_gate)
            # 139# §8-#1: ScoreCalibrator 注入
            self._inject_calibrator(skip_gate)
            self._skip_gate = skip_gate
            self._model_file_hash = self._compute_file_hash(gate_path)
            self._last_reload_check = time.monotonic()
        except Exception as e:
            logger.error(f"[skip_gate] Failed to load: {e}. SkipGate disabled.")
            self._skip_gate = None

    def _apply_config_overrides(self, skip_gate: object) -> None:
        """YAML 設定でモデル内 config をオーバーライド."""
        config = self._config
        sg = skip_gate  # type: ignore[assignment]
        sg.config.mode = config.skip_gate_mode
        sg.config.as_threshold = config.skip_gate_as_threshold
        sg.config.threshold_bps = config.skip_gate_pnl_threshold
        sg.config.max_skip_rate = config.skip_gate_max_skip_rate
        # 118# A3: side 別有効/無効
        sg.config.buy_enabled = config.skip_gate_buy_enabled
        sg.config.sell_enabled = config.skip_gate_sell_enabled
        # 068# §3.3: side 別閾値
        sg.config.as_threshold_buy = config.skip_gate_as_threshold_buy
        sg.config.as_threshold_sell = config.skip_gate_as_threshold_sell
        # 072# OB トグル
        sg.config.use_ob_features = config.skip_gate_use_ob_features
        # 088# 動的閾値較正
        sg.config.adaptive_threshold = config.skip_gate_adaptive_threshold
        sg.config.target_skip_rate_buy = config.skip_gate_target_skip_rate_buy
        sg.config.target_skip_rate_sell = config.skip_gate_target_skip_rate_sell
        sg.config.adaptive_window = config.skip_gate_adaptive_window
        sg.config.adaptive_min_samples = config.skip_gate_adaptive_min_samples
        sg.config.adaptive_step = config.skip_gate_adaptive_step
        sg.config.adaptive_floor = config.skip_gate_adaptive_floor
        sg.config.adaptive_ceiling = config.skip_gate_adaptive_ceiling
        logger.info(
            f"[skip_gate] Loaded: mode={config.skip_gate_mode}, "
            f"as_threshold={config.skip_gate_as_threshold}, "
            f"use_ob_features={config.skip_gate_use_ob_features}, "
            f"features={len(sg.feature_cols)}, "  # type: ignore[attr-defined]
            f"path={self._gate_path}"
        )

    def _apply_warm_start(self, skip_gate: object) -> None:
        """096# warm_start: 直近 P(AS) 履歴を復元."""
        if not self._config.skip_gate_adaptive_threshold:
            return
        try:
            from scripts.v460.ml.skip_gate import warm_start_skip_gate_thresholds
            warm_start_skip_gate_thresholds(
                skip_gate,
                self._config.results_dir,
                window=self._config.skip_gate_adaptive_window,
            )
        except Exception as ws_err:
            logger.warning(f"[skip_gate] Warm start failed (non-fatal): {ws_err}")

    def _inject_calibrator(self, skip_gate: object) -> None:
        """139# §8-#1: ScoreCalibrator を SkipGate に注入.

        config.skip_gate_score_calibration=True かつ calibrator_path が有効な場合、
        pkl からロードして SkipGate._score_calibrator に設定する。
        無効時は明示的に None を設定し、ログで可視化。
        """
        config = self._config
        if not config.skip_gate_score_calibration:
            skip_gate._score_calibrator = None  # type: ignore[attr-defined]
            logger.debug("[skip_gate] Score calibration disabled")
            return

        cal_path = config.skip_gate_calibrator_path
        if not cal_path:
            skip_gate._score_calibrator = None  # type: ignore[attr-defined]
            logger.info("[skip_gate] Score calibration enabled but no calibrator_path set")
            return

        try:
            from ztb.ml.score_calibrator import ScoreCalibrator, ScoreCalibratorConfig

            path = Path(cal_path)
            if not path.is_absolute():
                path = self._project_root / path
            cal = ScoreCalibrator.load(path)
            skip_gate._score_calibrator = cal  # type: ignore[attr-defined]
            status = f"fitted={cal.is_fitted}, n={cal.sample_count}"
            logger.info(f"[skip_gate] 139# ScoreCalibrator injected: {status}")
        except Exception as e:
            skip_gate._score_calibrator = None  # type: ignore[attr-defined]
            logger.warning(f"[skip_gate] ScoreCalibrator load failed: {e}")

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """ファイルの SHA256 ハッシュを算出 (126# hot-reload 用)."""
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            return ""

    def _check_and_reload_model(self) -> None:
        """126# モデルファイル変更を検出してリロード.

        retrain_scheduler がアトミックに pkl を差し替えた場合、
        次の evaluate() 呼び出し時にこのメソッドで検出・リロードする。
        """
        # __init__ がモック/スキップされた場合は何もしない
        if not hasattr(self, "_last_reload_check"):
            return
        now = time.monotonic()
        if now - self._last_reload_check < self._HOT_RELOAD_CHECK_INTERVAL_SEC:
            return
        self._last_reload_check = now

        if self._gate_path is None or not self._gate_path.exists():
            return

        new_hash = self._compute_file_hash(self._gate_path)
        if new_hash == self._model_file_hash or not new_hash:
            return

        # モデルファイルが変更された — リロード
        logger.info(
            f"[skip_gate] 126# Model file changed detected "
            f"(hash {self._model_file_hash[:8]}→{new_hash[:8]}). Reloading..."
        )
        try:
            from scripts.v460.ml.skip_gate import SkipGate
            new_gate = SkipGate.load(self._gate_path)
            self._apply_config_overrides(new_gate)
            self._apply_warm_start(new_gate)
            # 139# §8-#1: hot-reload 後も calibrator 再注入
            self._inject_calibrator(new_gate)
            self._skip_gate = new_gate
            self._model_file_hash = new_hash
            n_samples = new_gate.metadata.get("n_samples", "?")
            version = new_gate.metadata.get("version", "?")
            n_features = len(new_gate.feature_cols) if hasattr(new_gate, "feature_cols") else "?"
            logger.info(
                f"[skip_gate] 126# Hot-reload success: "
                f"version={version}, n_samples={n_samples}, "
                f"n_features={n_features}, "
                f"mode={new_gate.config.mode}, "  # type: ignore[union-attr]
                f"use_ob={new_gate.config.use_ob_features}"  # type: ignore[union-attr]
            )
        except Exception as e:
            logger.error(
                f"[skip_gate] 126# Hot-reload FAILED: {e}. "
                f"Keeping previous model."
            )

    @property
    def skip_gate(self) -> object | None:
        """内部 SkipGate インスタンスへのアクセス (OrderMonitor 等で使用)."""
        return self._skip_gate

    async def evaluate(
        self,
        side: str,
        cycle_id: str,
        order_price: float,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        adapter: object,
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
    ) -> SkipGateResult:
        """062# SkipGate ML 判定.

        Args:
            maker_price_vpin_setter: callable(vpin) — MakerPriceCalculator._last_vpin を設定
        """
        result = SkipGateResult()
        if self._skip_gate is None:
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
            from ztb.metrics.fill_quality import FillRecord

            logger.info(
                f"[skip_gate] SKIP: sell in unknown regime "
                f"(124# rule_skip_unknown_sell)"
            )
            result.skipped = True
            result.score = 0.0
            result.reason = "rule_skip_unknown_sell"
            result.model_used = "rule"
            result.early_return_record = FillRecord(
                cycle_id=cycle_id,
                timestamp=time.time(),
                side=side,
                order_price=order_price,
                order_quantity=current_lot,
                cancelled=True,
                cancel_reason="skip_gate_rule_unknown_sell",
                spread_at_order=spread_at_order,
                spread_offset_ratio=effective_offset_ratio,
                skip_gate_skipped=True,
                skip_gate_score=0.0,
                skip_gate_reason="rule_skip_unknown_sell",
                skip_gate_model_used="rule",
                orderbook_imbalance=last_imbalance,
                bid_depth_total=last_bid_depth,
                ask_depth_total=last_ask_depth,
                run_id=run_id,
                git_sha=git_sha,
            )
            return result

        try:
            from scripts.v460.ml.skip_gate import build_features_from_market_state

            sg_regime = regime_value or "unknown"

            # 直近約定データ取得
            recent_trades_data: list[dict] | None = None
            try:
                trades = await adapter.get_recent_trades(  # type: ignore[union-attr]
                    symbol, limit=self._config.skip_gate_recent_trades_limit,
                )
                if trades:
                    recent_trades_data = [
                        {
                            "ts": getattr(t, "timestamp", time.time()),
                            "price": getattr(t, "price", 0.0),
                            "amount": getattr(t, "amount", getattr(t, "quantity", 0.0)),
                            "side": getattr(t, "side", "buy"),
                        }
                        for t in trades
                    ]
            except Exception:
                pass

            # 072# OB トグル
            ob_bid: float | None = None
            ob_ask: float | None = None
            ob_bid_vol: float | None = None
            ob_ask_vol: float | None = None
            sg_use_ob = self._skip_gate.config.use_ob_features  # type: ignore[union-attr]
            if sg_use_ob:
                try:
                    ob = await adapter.get_orderbook(symbol, depth=self._config.skip_gate_ob_depth)  # type: ignore[union-attr]
                    if ob and ob.bids and ob.asks:
                        ob_bid = ob.bids[0].price
                        ob_ask = ob.asks[0].price
                        ob_bid_vol = sum(lv.quantity for lv in ob.bids[:self._config.skip_gate_ob_depth])
                        ob_ask_vol = sum(lv.quantity for lv in ob.asks[:self._config.skip_gate_ob_depth])
                except Exception as e:
                    logger.debug(f"[skip_gate] OB fetch failed: {e}")

            gate_features = build_features_from_market_state(
                side=side,
                spread_jpy=spread_at_order or 0.0,
                offset_ratio=effective_offset_ratio,
                regime=sg_regime,
                recent_trades=recent_trades_data,
                market_timestamp=time.time(),
                best_bid=ob_bid,
                best_ask=ob_ask,
                bid_vol_5=ob_bid_vol,
                ask_vol_5=ob_ask_vol,
                use_ob_features=sg_use_ob,
            )

            # 107# Volatility Guard: VPIN キャッシュ
            if maker_price_vpin_setter is not None and callable(maker_price_vpin_setter):
                maker_price_vpin_setter(gate_features.get("vpin_60s"))

            decision = self._skip_gate.evaluate(gate_features, side=side)  # type: ignore[union-attr]
            result.skipped = decision.should_skip
            result.score = decision.predicted_pnl_bps
            result.reason = decision.reason
            result.model_used = decision.model_used
            result.as_prob = decision.as_probability
            result.threshold_used = decision.threshold_used

            if decision.should_skip:
                from ztb.metrics.fill_quality import FillRecord

                logger.info(
                    f"[skip_gate] SKIP: {side} order skipped "
                    f"(score={result.score:.3f}, reason={result.reason}, "
                    f"model={result.model_used}, features={decision.features_used})"
                )
                result.early_return_record = FillRecord(
                    cycle_id=cycle_id,
                    timestamp=time.time(),
                    side=side,
                    order_price=order_price,
                    order_quantity=current_lot,
                    cancelled=True,
                    cancel_reason="skip_gate",
                    spread_at_order=spread_at_order,
                    spread_offset_ratio=effective_offset_ratio,
                    skip_gate_skipped=True,
                    skip_gate_score=result.score,
                    skip_gate_reason=result.reason,
                    skip_gate_model_used=result.model_used,
                    skip_gate_as_prob=result.as_prob,
                    skip_gate_threshold_used=result.threshold_used,
                    # 122# R5: OB 記録を imbalance_enabled と独立させ常時記録
                    orderbook_imbalance=last_imbalance,
                    bid_depth_total=last_bid_depth,
                    ask_depth_total=last_ask_depth,
                    run_id=run_id,
                    git_sha=git_sha,
                )
            else:
                logger.debug(
                    f"[skip_gate] PASS: {side} order allowed "
                    f"(score={result.score:.3f}, reason={result.reason}, "
                    f"model={result.model_used})"
                )
        except Exception as e:
            logger.warning(f"[skip_gate] Evaluation failed (non-fatal): {e}")
            result.reason = f"error:{e}"

        return result
