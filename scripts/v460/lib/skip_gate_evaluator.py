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

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from scripts.v460.lib.fill_config import FillTestConfig, SkipGateResult
# ファイルの SHA256 ハッシュを算出 (126# hot-reload 用)
from ztb.utils.run_manifest import compute_file_hash

if TYPE_CHECKING:
    from ztb.metrics.fill_quality import FillRecord

logger = logging.getLogger(__name__)


class _SkipGateConfigLike(Protocol):
    mode: str
    as_threshold: float
    threshold_bps: float
    max_skip_rate: float
    buy_enabled: bool
    sell_enabled: bool
    as_threshold_buy: float | None
    as_threshold_sell: float | None
    use_ob_features: bool
    adaptive_threshold: bool
    target_skip_rate_buy: float
    target_skip_rate_sell: float
    adaptive_window: int
    adaptive_min_samples: int
    adaptive_step: float
    adaptive_floor: float
    adaptive_ceiling: float
    regime_thresholds: dict[str, float]


class _SkipDecisionLike(Protocol):
    should_skip: bool
    predicted_pnl_bps: float
    threshold_bps: float
    reason: str
    model_used: str
    as_probability: float | None
    threshold_used: float | None
    features_used: int


class _SkipGateLike(Protocol):
    config: _SkipGateConfigLike
    metadata: dict[str, object]
    feature_cols: list[str]

    def evaluate(
        self,
        features: dict[str, object],
        *,
        side: str | None = None,
        regime: str | None = None,
        threshold_offset: float = ...,
    ) -> _SkipDecisionLike:
        ...


class _SkipGateClassLike(Protocol):
    @staticmethod
    def load(path: Path) -> _SkipGateLike:
        ...


class SkipGateEvaluator:
    """SkipGate ML フィルター (062# / 088# / 096# / 126# 統合)."""

    # 126# hot-reload: モデルファイルのチェック間隔 (秒)
    # 158# YAML 外部化: config.hot_reload_check_interval_sec から読み込み (フォールバック用に残す)
    _HOT_RELOAD_CHECK_INTERVAL_SEC = 120.0  # デフォルト値 (config 未設定時のフォールバック)
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
            from scripts.v460.ml.skip_gate import SkipGate
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
            self._model_file_hash = compute_file_hash(gate_path)
            self._last_reload_check = time.monotonic()

            # 141# P1-01: side 別モデルロード
            self._load_side_models(skip_gate_cls)
            # 188# C-1: ev_weighted 副 horizon モデルロード
            self._load_alt_models(skip_gate_cls)
        except Exception as e:
            logger.error(f"[skip_gate] Failed to load: {e}. SkipGate disabled.")
            self._skip_gate = None

    def _resolve_model_path(self, model_path: str) -> Path:
        path = Path(model_path)
        if not path.is_absolute():
            path = self._project_root / path
        return path

    def _load_gate_from_path(
        self,
        skip_gate_cls: _SkipGateClassLike,
        gate_path: Path,
        *,
        apply_warm_start: bool = False,
    ) -> _SkipGateLike:
        gate = skip_gate_cls.load(gate_path)
        self._apply_config_overrides(gate)
        if apply_warm_start:
            self._apply_warm_start(gate)
        self._inject_calibrator(gate)
        return gate

    def _apply_config_overrides(self, skip_gate: _SkipGateLike) -> None:
        """YAML 設定でモデル内 config をオーバーライド."""
        config = self._config
        sg = skip_gate
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
        _valid_regimes = {"trending", "trending_up", "trending_down", "ranging", "high_vol", "unknown"}
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
            f"features={len(sg.feature_cols)}, "
            f"path={self._gate_path}"
        )

    def _apply_warm_start(self, skip_gate: _SkipGateLike) -> None:
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

    def _inject_calibrator(self, skip_gate: _SkipGateLike) -> None:
        """139# §8-#1: ScoreCalibrator を SkipGate に注入.

        config.skip_gate_score_calibration=True かつ calibrator_path が有効な場合、
        pkl からロードして SkipGate._score_calibrator に設定する。
        無効時は明示的に None を設定し、ログで可視化。
        """
        config = self._config
        if not config.skip_gate_score_calibration:
            setattr(skip_gate, "_score_calibrator", None)
            logger.debug("[skip_gate] Score calibration disabled")
            return

        cal_path = config.skip_gate_calibrator_path
        if not cal_path:
            setattr(skip_gate, "_score_calibrator", None)
            logger.info("[skip_gate] Score calibration enabled but no calibrator_path set")
            return

        try:
            from ztb.ml.score_calibrator import ScoreCalibrator

            path = self._resolve_model_path(cal_path)
            cal = ScoreCalibrator.load(path)
            setattr(skip_gate, "_score_calibrator", cal)
            status = f"fitted={cal.is_fitted}, n={cal.sample_count}"
            logger.info(f"[skip_gate] 139# ScoreCalibrator injected: {status}")
        except Exception as e:
            setattr(skip_gate, "_score_calibrator", None)
            logger.warning(f"[skip_gate] ScoreCalibrator load failed: {e}")

    def _load_side_models(self, skip_gate_cls: _SkipGateClassLike) -> None:
        """141# P1-01: side 別モデルをロード.

        model_path_buy/sell が設定されていてファイルが存在する場合にロード。
        存在しない場合は unified モデルにフォールバック（_gate_buy/_gate_sell は None のまま）。
        """
        config = self._config
        for side, attr_gate, attr_path, attr_hash in self._SIDE_MODEL_SLOTS:
            model_path_str = getattr(config, f"skip_gate_model_path_{side}", None)
            if not model_path_str:
                continue
            gate_path = self._resolve_model_path(model_path_str)
            if not gate_path.exists():
                logger.info(
                    f"[skip_gate] 141# {side} model not found: {gate_path}. "
                    f"Will use unified model."
                )
                continue
            try:
                side_gate = self._load_gate_from_path(skip_gate_cls, gate_path)
                setattr(self, attr_gate, side_gate)
                setattr(self, attr_path, gate_path)
                setattr(self, attr_hash, compute_file_hash(gate_path))
                n_features = len(side_gate.feature_cols)
                target = side_gate.metadata.get("target", "?")
                logger.info(
                    f"[skip_gate] 141# {side} model loaded: {gate_path}, "
                    f"features={n_features}, target={target}"
                )
            except Exception as e:
                logger.warning(
                    f"[skip_gate] 141# {side} model load failed: {e}. "
                    f"Will use unified model."
                )

    def _load_alt_models(self, skip_gate_cls: _SkipGateClassLike) -> None:
        """188# C-1: ev_weighted 用の副 horizon モデルをロード.

        buy_long (pnl120 buy), sell_short (pnl30 sell) が設定・存在する場合にロード。
        ev_weighted_enabled=False でもパスを差していればロード (後の YAML hot-reload で有効化可能)。
        """
        config = self._config
        for side, attr_gate, attr_path, attr_hash, config_key in self._ALT_MODEL_SLOTS:
            model_path_str = getattr(config, config_key, None)
            if not model_path_str:
                continue
            gate_path = self._resolve_model_path(model_path_str)
            if not gate_path.exists():
                logger.info(
                    f"[skip_gate] 188# alt {side} model not found: {gate_path}. "
                    f"ev_weighted will fall back to single-model evaluation."
                )
                continue
            try:
                alt_gate = self._load_gate_from_path(skip_gate_cls, gate_path)
                setattr(self, attr_gate, alt_gate)
                setattr(self, attr_path, gate_path)
                setattr(self, attr_hash, compute_file_hash(gate_path))
                n_features = len(alt_gate.feature_cols)
                target = alt_gate.metadata.get("target", "?")
                logger.info(
                    f"[skip_gate] 188# alt {side} model loaded: {gate_path}, "
                    f"features={n_features}, target={target}"
                )
            except Exception as e:
                logger.warning(
                    f"[skip_gate] 188# alt {side} model load failed: {e}. "
                    f"ev_weighted will fall back to single-model evaluation."
                )

    def _try_ev_weighted_decision(
        self,
        side: str,
        features: dict[str, object] | dict[str, float],
        regime: str,
        threshold_offset: float,
        primary_decision: _SkipDecisionLike,
        *,
        one_sided_balance: bool = False,
    ) -> _SkipDecisionLike | None:
        """188# C-1: ev_weighted 統合判定.

        副 horizon モデルが存在し ev_weighted_enabled=True の場合、
        primary (短期) と alt (長期) の predicted_pnl を ev_weighted で統合。
        AS mode では ev_weighted は適用しない (確率ベースのため加重平均が不適切)。

        193#: ev_as_offset_enabled=True 時は offset 修飾子モードに切り替え。
        - ゲート判定は行わず ev_score を計算して返す (should_skip=False 固定)
        - ev_score < emergency_threshold の場合のみ hard skip
        - 安全弁・片側緩和は不要 (offset が連続的に調整するため)

        190# A: 連続 skip 安全弁 + B: 片側 balance 時の threshold 緩和。
        (193# ev_as_offset_enabled=True 時は無視)

        Returns:
            統合判定の SkipDecision, または None (ev_weighted 不適用時).
        """
        if not self._config.skip_gate_ev_weighted_enabled:
            return None

        # AS mode では ev_weighted 不適用
        alt_gate: _SkipGateLike | None
        if side == "buy":
            alt_gate = self._gate_alt_buy
        elif side == "sell":
            alt_gate = self._gate_alt_sell
        else:
            return None

        if alt_gate is None:
            return None

        # PnL mode のみ ev_weighted (AS mode は確率空間の加重平均が不適切)
        if alt_gate.config.mode != "pnl":
            logger.debug(
                "[skip_gate] 188# ev_weighted skipped: alt model mode=%s (pnl required)",
                alt_gate.config.mode,
            )
            return None

        try:
            alt_decision = alt_gate.evaluate(
                features,
                side=side,
                regime=regime,
                threshold_offset=threshold_offset,
            )
        except Exception as e:
            logger.warning("[skip_gate] 188# alt model evaluate failed: %s", e)
            return None

        # primary と alt の pred_pnl を ev_weighted 合成
        w30 = self._config.skip_gate_ev_w30
        w120 = self._config.skip_gate_ev_w120
        primary_pnl = primary_decision.predicted_pnl_bps
        alt_pnl = alt_decision.predicted_pnl_bps

        # buy: primary=pnl30 (短期), alt=pnl120 (長期)
        # sell: primary=pnl120 (長期), alt=pnl30 (短期)
        if side == "buy":
            ev_score = w30 * primary_pnl + w120 * alt_pnl
        else:
            ev_score = w30 * alt_pnl + w120 * primary_pnl

        # 193#: offset 修飾子モード
        if self._config.skip_gate_ev_as_offset_enabled:
            return self._ev_weighted_as_offset(
                side, ev_score, primary_decision,
            )

        # --- 旧モード: ハードゲート (ev_as_offset_enabled=False) ---
        # ev_weighted threshold — primary の threshold_used を基準に判定
        threshold_used = primary_decision.threshold_used
        if threshold_used is None:
            threshold_used = 0.0

        # 190# B: 片側 balance 時の threshold 緩和
        # BTC=0 で buy しか選択肢がない状態で threshold が厳しすぎるとデッドロック
        _threshold_relaxation = self._config.skip_gate_ev_one_sided_threshold_shift
        if one_sided_balance and _threshold_relaxation != 0.0:
            _original_threshold = threshold_used
            threshold_used += _threshold_relaxation  # 負方向シフト = 緩和
            logger.debug(
                "[skip_gate] 190# B: one_sided_balance threshold relaxation: "
                "%.3f → %.3f (shift=%.3f)",
                _original_threshold, threshold_used, _threshold_relaxation,
            )

        should_skip = ev_score < threshold_used

        # 190# A: 連続 skip 安全弁
        _max_consecutive = self._config.skip_gate_ev_max_consecutive_skip
        if should_skip and _max_consecutive > 0:
            self._ev_consecutive_skip_count += 1
            if self._ev_consecutive_skip_count >= _max_consecutive:
                logger.warning(
                    "[skip_gate] 190# A: ev_weighted consecutive skip safety valve: "
                    "%d consecutive skips >= limit %d — forcing PASS "
                    "(score=%.3f, threshold=%.3f)",
                    self._ev_consecutive_skip_count, _max_consecutive,
                    ev_score, threshold_used,
                )
                self._ev_consecutive_skip_count = 0
                should_skip = False
        elif not should_skip:
            # PASS 時はカウンタリセット
            self._ev_consecutive_skip_count = 0

        logger.debug(
            "[skip_gate] 188# ev_weighted: side=%s pnl_primary=%.3f pnl_alt=%.3f "
            "ev=%.3f threshold=%.3f skip=%s consec=%d",
            side, primary_pnl, alt_pnl, ev_score, threshold_used, should_skip,
            self._ev_consecutive_skip_count,
        )

        from scripts.v460.ml.skip_gate import SkipDecision
        _reason = (
            "ev_weighted_skip" if should_skip
            else "ev_weighted_pass_safety" if self._ev_consecutive_skip_count == 0 and ev_score < threshold_used
            else "ev_weighted_pass"
        )
        return SkipDecision(
            should_skip=should_skip,
            predicted_pnl_bps=ev_score,
            threshold_bps=primary_decision.threshold_bps,
            features_used=primary_decision.features_used if hasattr(primary_decision, "features_used") else 0,
            reason=_reason,
            model_used="ev_weighted",
            as_probability=primary_decision.as_probability,
            threshold_used=threshold_used,
        )

    def _ev_weighted_as_offset(
        self,
        side: str,
        ev_score: float,
        primary_decision: _SkipDecisionLike,
    ) -> _SkipDecisionLike:
        """193#: ev_weighted を offset 修飾子として機能させる.

        - ev_score を predicted_pnl_bps に格納して返却
        - should_skip は emergency threshold 未満の場合のみ True
        - 安全弁・片側緩和は不要 (offset が連続的に調整)
        - ev_score は SkipGateResult.ev_score 経由で executor に渡り、
          order_price を post-hoc 調整する

        Returns:
            SkipDecision (should_skip=False except emergency)
        """
        from scripts.v460.ml.skip_gate import SkipDecision

        _emergency = self._config.skip_gate_ev_emergency_skip_threshold
        if ev_score < _emergency:
            logger.warning(
                "[skip_gate] 193# ev_weighted EMERGENCY SKIP: "
                "ev_score=%.3f < emergency_threshold=%.3f",
                ev_score, _emergency,
            )
            # 安全弁カウンタリセット (旧モード互換)
            self._ev_consecutive_skip_count = 0
            return SkipDecision(
                should_skip=True,
                predicted_pnl_bps=ev_score,
                threshold_bps=primary_decision.threshold_bps,
                features_used=primary_decision.features_used if hasattr(primary_decision, "features_used") else 0,
                reason="ev_weighted_emergency_skip",
                model_used="ev_weighted",
                as_probability=primary_decision.as_probability,
                threshold_used=_emergency,
            )

        # offset 修飾子モード: 安全弁カウンタリセット
        self._ev_consecutive_skip_count = 0

        # offset 乗数を計算 (ログ用)
        _sens = self._config.skip_gate_ev_offset_sensitivity
        _min_m = self._config.skip_gate_ev_offset_min_mult
        _max_m = self._config.skip_gate_ev_offset_max_mult
        _raw_mult = 1.0 + _sens * ev_score
        _clamped_mult = max(_min_m, min(_max_m, _raw_mult))

        logger.info(
            "[skip_gate] 193# ev_weighted→offset: side=%s ev_score=%.3f "
            "→ offset_mult=%.3f (raw=%.3f, sens=%.3f, clamp=[%.2f,%.2f])",
            side, ev_score, _clamped_mult, _raw_mult, _sens, _min_m, _max_m,
        )

        return SkipDecision(
            should_skip=False,
            predicted_pnl_bps=ev_score,
            threshold_bps=primary_decision.threshold_bps,
            features_used=primary_decision.features_used if hasattr(primary_decision, "features_used") else 0,
            reason="ev_weighted_offset",
            model_used="ev_weighted",
            as_probability=primary_decision.as_probability,
            threshold_used=0.0,
        )


    def _check_and_reload_model(self) -> None:
        """126# モデルファイル変更を検出してリロード.

        retrain_scheduler がアトミックに pkl を差し替えた場合、
        次の evaluate() 呼び出し時にこのメソッドで検出・リロードする。
        """
        # __init__ がモック/スキップされた場合は何もしない
        if not hasattr(self, "_last_reload_check"):
            return
        now = time.monotonic()
        # 158# YAML 外部化: config から hot_reload 間隔を取得
        _raw = getattr(self._config, "hot_reload_check_interval_sec", None)
        interval = _raw if isinstance(_raw, (int, float)) else self._HOT_RELOAD_CHECK_INTERVAL_SEC
        if now - self._last_reload_check < interval:
            return
        self._last_reload_check = now

        # 143# A.1 #1: side 別モデルは unified 更新とは独立にチェック
        self._check_and_reload_side_models()

        if self._gate_path is None or not self._gate_path.exists():
            return

        new_hash = compute_file_hash(self._gate_path)
        if new_hash == self._model_file_hash or not new_hash:
            return

        # モデルファイルが変更された — リロード
        logger.info(
            f"[skip_gate] 126# Model file changed detected "
            f"(hash {self._model_file_hash[:8]}→{new_hash[:8]}). Reloading..."
        )
        try:
            from scripts.v460.ml.skip_gate import SkipGate
            skip_gate_cls = cast(_SkipGateClassLike, SkipGate)
            new_gate = self._load_gate_from_path(
                skip_gate_cls,
                self._gate_path,
                apply_warm_start=True,
            )
            self._skip_gate = new_gate
            # 175# _inject_calibrator は _load_gate_from_path 内で既に実行済み
            self._model_file_hash = new_hash
            n_samples = new_gate.metadata.get("n_samples", "?")
            version = new_gate.metadata.get("version", "?")
            n_features = len(new_gate.feature_cols)
            logger.info(
                f"[skip_gate] 126# Hot-reload success: "
                f"version={version}, n_samples={n_samples}, "
                f"n_features={n_features}, "
                f"mode={new_gate.config.mode}, "
                f"use_ob={new_gate.config.use_ob_features}"
            )
        except Exception as e:
            logger.error(
                f"[skip_gate] 126# Hot-reload FAILED: {e}. "
                f"Keeping previous model."
            )

        # 143# A.1 #1: (moved to top of _check_and_reload_model)

    def _check_and_reload_side_models(self) -> None:
        """141# side 別モデルの変更検出 + リロード."""
        try:
            from scripts.v460.ml.skip_gate import SkipGate
        except Exception as e:
            logger.debug(f"[skip_gate] side hot-reload skipped (import failed): {e}")
            return
        skip_gate_cls = cast(_SkipGateClassLike, SkipGate)

        for side, attr_gate, attr_path, attr_hash in self._SIDE_MODEL_SLOTS:
            gate_path: Path | None = getattr(self, attr_path, None)
            if gate_path is None:
                # パス未設定の場合: config から新規ロード試行
                model_path_str = getattr(self._config, f"skip_gate_model_path_{side}", None)
                if not model_path_str:
                    continue
                gate_path = self._resolve_model_path(model_path_str)
                if not gate_path.exists():
                    continue
                # 新規モデルファイル出現 → ロード
                try:
                    new_gate = self._load_gate_from_path(skip_gate_cls, gate_path)
                    setattr(self, attr_gate, new_gate)
                    setattr(self, attr_path, gate_path)
                    setattr(self, attr_hash, compute_file_hash(gate_path))
                    logger.info(f"[skip_gate] 141# {side} model first load via hot-reload: {gate_path}")
                except Exception as e:
                    logger.warning(f"[skip_gate] 141# {side} model first load failed: {e}")
                continue

            if not gate_path.exists():
                continue
            old_hash = getattr(self, attr_hash, "")
            new_hash = compute_file_hash(gate_path)
            if new_hash == old_hash or not new_hash:
                continue
            try:
                new_gate = self._load_gate_from_path(skip_gate_cls, gate_path)
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
        if side == "buy" and getattr(self, "_gate_buy", None) is not None:
            return self._gate_buy
        if side == "sell" and getattr(self, "_gate_sell", None) is not None:
            return self._gate_sell
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
        *,
        one_sided_balance: bool = False,
    ) -> SkipGateResult:
        """062# SkipGate ML 判定.

        Args:
            maker_price_vpin_setter: callable(vpin) — MakerPriceCalculator._last_vpin を設定
        """
        result = SkipGateResult()
        # 143# A.1 #2: unified 不在でも side モデルが使える場合は続行
        if self._skip_gate is None and getattr(self, "_gate_buy", None) is None and getattr(self, "_gate_sell", None) is None:
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
                regime=regime_value,  # 160#
            )
            return result

        # side 固有モデルのみ存在し、要求 side にはモデルがないケースを明示処理
        active_gate = self._select_gate_for_side(side)
        if active_gate is None:
            result.skipped = False
            result.score = 0.0
            result.reason = f"no_model_for_side:{side}"
            result.model_used = "none"
            logger.info(
                f"[skip_gate] No model available for side={side}. "
                f"unified={'yes' if self._skip_gate is not None else 'no'}"
            )
            return result

        try:
            from scripts.v460.ml.skip_gate import build_features_from_market_state

            sg_regime = regime_value or "unknown"

            # 直近約定データ取得
            recent_trades_data: list[dict[str, object]] | None = None
            try:
                get_recent_trades = getattr(adapter, "get_recent_trades", None)
                if callable(get_recent_trades):
                    trades = await get_recent_trades(
                        symbol,
                        limit=self._config.skip_gate_recent_trades_limit,
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
                    get_orderbook = getattr(adapter, "get_orderbook", None)
                    if callable(get_orderbook):
                        ob = await get_orderbook(
                            symbol,
                            depth=self._config.skip_gate_ob_depth,
                        )
                        if ob and getattr(ob, "bids", None) and getattr(ob, "asks", None):
                            bids = getattr(ob, "bids")
                            asks = getattr(ob, "asks")
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

            # 165# AS-R1: velocity-based pre-ML sell/buy skip rule
            # 195# ソフト化: velocity_skip_as_offset_enabled 時は hard skip せず offset boost
            _pv60 = gate_features.get("price_velocity_60s", 0.0)
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
                    if self._config.velocity_offset_proportional:
                        _excess_ratio = abs(_pv60) / abs(_vel_th)  # >= 1.0
                        _base = self._config.velocity_offset_boost_factor
                        _boost = 1.0 + (_base - 1.0) * _excess_ratio
                        _boost = min(_boost, self._config.velocity_offset_max_mult)
                    else:
                        _boost = self._config.velocity_offset_boost_factor
                    result.velocity_offset_mult = _boost
                    result.price_velocity_60s = _pv60
                    logger.info(
                        f"[skip_gate] 195# velocity→offset: {_vel_label} "
                        f"velocity={_pv60:.2f}bps (th={_vel_th}) "
                        f"→ offset_mult={_boost:.2f}"
                        f"{' (proportional)' if self._config.velocity_offset_proportional else ''}"
                    )
                    # hard skip しない → ML 判定に進む
                else:
                    # 旧モード: hard skip
                    from ztb.metrics.fill_quality import FillRecord

                    _reason = f"rule_velocity_{_vel_label}_skip"
                    _cancel = f"skip_gate_rule_velocity_{_vel_label}"
                    logger.info(
                        f"[skip_gate] SKIP: {_vel_label} velocity {_pv60:.2f}bps "
                        f"{'>' if _velocity_sell_triggered else '<'} {_vel_th}bps "
                        f"(165# AS-R1 {_reason})"
                    )
                    result.skipped = True
                    result.score = _pv60
                    result.reason = _reason
                    result.model_used = "rule"
                    result.early_return_record = FillRecord(
                        cycle_id=cycle_id,
                        timestamp=time.time(),
                        side=side,
                        order_price=order_price,
                        order_quantity=current_lot,
                        cancelled=True,
                        cancel_reason=_cancel,
                        spread_at_order=spread_at_order,
                        spread_offset_ratio=effective_offset_ratio,
                        skip_gate_skipped=True,
                        skip_gate_score=_pv60,
                        skip_gate_reason=_reason,
                        skip_gate_model_used="rule",
                        orderbook_imbalance=last_imbalance,
                        bid_depth_total=last_bid_depth,
                        ask_depth_total=last_ask_depth,
                        run_id=run_id,
                        git_sha=git_sha,
                        regime=regime_value,
                        price_velocity_60s=_pv60,
                    )
                    return result

            # 158# P1-6: 時間帯別 skip_gate 閾値調整
            from datetime import datetime, timezone as _tz
            _utc_hour = datetime.now(_tz.utc).hour
            _hour_offset = self._config.skip_gate_hour_offsets.get(_utc_hour, 0.0)

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

            _total_offset = _hour_offset + _spread_offset

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

            result.skipped = decision.should_skip
            result.score = decision.predicted_pnl_bps
            result.reason = decision.reason
            result.price_velocity_60s = gate_features.get("price_velocity_60s")
            # 141#/188#: model_used にどのモデルが使われたかを示す
            if decision.model_used == "ev_weighted":
                model_tag = f"ev_weighted_{side}"
            else:
                is_side_model = (
                    (side == "buy" and self._gate_buy is not None)
                    or (side == "sell" and self._gate_sell is not None)
                )
                model_tag = f"side_{side}" if is_side_model else "unified"
            result.model_used = f"{decision.model_used}:{model_tag}"
            result.as_prob = decision.as_probability
            result.threshold_used = decision.threshold_used
            result.hour_offset = _total_offset

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
                    skip_gate_hour_offset=_total_offset if _total_offset != 0.0 else None,
                    # 122# R5: OB 記録を imbalance_enabled と独立させ常時記録
                    orderbook_imbalance=last_imbalance,
                    bid_depth_total=last_bid_depth,
                    ask_depth_total=last_ask_depth,
                    run_id=run_id,
                    git_sha=git_sha,
                    regime=regime_value,  # 160#
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
