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
        # 141# P1-01: side 別 SkipGate インスタンス (フォールバック用に unified も保持)
        self._gate_buy: object | None = None
        self._gate_sell: object | None = None
        self._gate_path_buy: Path | None = None
        self._gate_path_sell: Path | None = None
        self._model_file_hash_buy: str = ""
        self._model_file_hash_sell: str = ""
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

            # 141# P1-01: side 別モデルロード
            self._load_side_models(SkipGate)
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
        # 141# P1-04: regime 別閾値
        sg.config.regime_thresholds = config.skip_gate_regime_thresholds
        # 142# M-3: regime_thresholds キーバリデーション
        _valid_regimes = {"trending", "ranging", "high_vol", "unknown"}
        for key in config.skip_gate_regime_thresholds:
            if key not in _valid_regimes:
                logger.warning(
                    f"[skip_gate] 142# unknown regime key in regime_thresholds: "
                    f"'{key}'. Valid: {sorted(_valid_regimes)}"
                )
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

    def _load_side_models(self, skip_gate_cls: type) -> None:
        """141# P1-01: side 別モデルをロード.

        model_path_buy/sell が設定されていてファイルが存在する場合にロード。
        存在しない場合は unified モデルにフォールバック（_gate_buy/_gate_sell は None のまま）。
        """
        config = self._config
        for side, attr_gate, attr_path, attr_hash in (
            ("buy", "_gate_buy", "_gate_path_buy", "_model_file_hash_buy"),
            ("sell", "_gate_sell", "_gate_path_sell", "_model_file_hash_sell"),
        ):
            model_path_str = getattr(config, f"skip_gate_model_path_{side}", None)
            if not model_path_str:
                continue
            gate_path = Path(model_path_str)
            if not gate_path.is_absolute():
                gate_path = self._project_root / gate_path
            if not gate_path.exists():
                logger.info(
                    f"[skip_gate] 141# {side} model not found: {gate_path}. "
                    f"Will use unified model."
                )
                continue
            try:
                side_gate = skip_gate_cls.load(gate_path)
                self._apply_config_overrides(side_gate)
                self._inject_calibrator(side_gate)
                setattr(self, attr_gate, side_gate)
                setattr(self, attr_path, gate_path)
                setattr(self, attr_hash, self._compute_file_hash(gate_path))
                n_features = len(side_gate.feature_cols) if hasattr(side_gate, "feature_cols") else "?"
                target = side_gate.metadata.get("target", "?") if hasattr(side_gate, "metadata") else "?"
                logger.info(
                    f"[skip_gate] 141# {side} model loaded: {gate_path}, "
                    f"features={n_features}, target={target}"
                )
            except Exception as e:
                logger.warning(
                    f"[skip_gate] 141# {side} model load failed: {e}. "
                    f"Will use unified model."
                )

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

        # 141# P1-01: side 別モデルの hot-reload チェック
        self._check_and_reload_side_models()

    def _check_and_reload_side_models(self) -> None:
        """141# side 別モデルの変更検出 + リロード."""
        from scripts.v460.ml.skip_gate import SkipGate

        for side, attr_gate, attr_path, attr_hash in (
            ("buy", "_gate_buy", "_gate_path_buy", "_model_file_hash_buy"),
            ("sell", "_gate_sell", "_gate_path_sell", "_model_file_hash_sell"),
        ):
            gate_path: Path | None = getattr(self, attr_path, None)
            if gate_path is None:
                # パス未設定の場合: config から新規ロード試行
                model_path_str = getattr(self._config, f"skip_gate_model_path_{side}", None)
                if not model_path_str:
                    continue
                gate_path = Path(model_path_str)
                if not gate_path.is_absolute():
                    gate_path = self._project_root / gate_path
                if not gate_path.exists():
                    continue
                # 新規モデルファイル出現 → ロード
                try:
                    new_gate = SkipGate.load(gate_path)
                    self._apply_config_overrides(new_gate)
                    self._inject_calibrator(new_gate)
                    setattr(self, attr_gate, new_gate)
                    setattr(self, attr_path, gate_path)
                    setattr(self, attr_hash, self._compute_file_hash(gate_path))
                    logger.info(f"[skip_gate] 141# {side} model first load via hot-reload: {gate_path}")
                except Exception as e:
                    logger.warning(f"[skip_gate] 141# {side} model first load failed: {e}")
                continue

            if not gate_path.exists():
                continue
            old_hash = getattr(self, attr_hash, "")
            new_hash = self._compute_file_hash(gate_path)
            if new_hash == old_hash or not new_hash:
                continue
            try:
                new_gate = SkipGate.load(gate_path)
                self._apply_config_overrides(new_gate)
                self._inject_calibrator(new_gate)
                setattr(self, attr_gate, new_gate)
                setattr(self, attr_hash, new_hash)
                logger.info(
                    f"[skip_gate] 141# {side} model hot-reloaded: "
                    f"{old_hash[:8]}→{new_hash[:8]}"
                )
            except Exception as e:
                logger.warning(
                    f"[skip_gate] 141# {side} model hot-reload failed: {e}. "
                    f"Keeping previous."
                )

    @property
    def skip_gate(self) -> object | None:
        """内部 SkipGate インスタンスへのアクセス (OrderMonitor 等で使用)."""
        return self._skip_gate

    def _select_gate_for_side(self, side: str) -> object:
        """141# P1-01: side に適合する SkipGate を返す.

        side 別モデルが存在する場合はそちらを優先し、
        なければ統一モデルにフォールバック。
        """
        if side == "buy" and getattr(self, "_gate_buy", None) is not None:
            return self._gate_buy
        if side == "sell" and getattr(self, "_gate_sell", None) is not None:
            return self._gate_sell
        return self._skip_gate  # type: ignore[return-value]

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
            # 141# P1-01: side 別モデルの use_ob_features を参照
            active_gate_for_ob = self._select_gate_for_side(side)
            sg_use_ob = active_gate_for_ob.config.use_ob_features  # type: ignore[union-attr]
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

            # 141# P1-01: side 別モデルにディスパッチ (フォールバック: unified)
            active_gate = self._select_gate_for_side(side)
            decision = active_gate.evaluate(gate_features, side=side, regime=sg_regime)  # type: ignore[union-attr]
            result.skipped = decision.should_skip
            result.score = decision.predicted_pnl_bps
            result.reason = decision.reason
            # 141#: model_used にどのモデルが使われたかを示す
            is_side_model = (
                (side == "buy" and self._gate_buy is not None)
                or (side == "sell" and self._gate_sell is not None)
            )
            model_tag = f"side_{side}" if is_side_model else "unified"
            result.model_used = f"{decision.model_used}:{model_tag}"
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
