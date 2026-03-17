"""461# SkipGate モデル管理 Mixin.

skip_gate_evaluator.py からモデルの読み込み・設定オーバーライド・
hot-reload 関連メソッドを抽出。

責務:
  - モデルファイルパス解決 / SHA256 ハッシュ読み取り
  - unified / side 別 / alt (ev_weighted) モデルのロード
  - config オーバーライド / warm-start / calibrator 注入
  - 126# hot-reload: ファイル変更検出 + アトミックリロード
  - 141# side 別 hot-reload

MAX LINES: 400
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ztb.utils.run_manifest import compute_file_hash

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig
    from scripts.v460.lib.skip_gate_evaluator import (
        _SkipGateClassLike,
        _SkipGateLike,
    )

logger = logging.getLogger(__name__)


class SkipGateModelLoaderMixin:
    """モデル管理系メソッドを提供する Mixin.

    SkipGateEvaluator が継承して使用する。
    self._config: FillTestConfig, self._project_root: Path を前提とする。
    """

    # --- path / hash ---

    def _resolve_model_path(self, model_path: str) -> Path:
        path = Path(model_path)
        if not path.is_absolute():
            path = self._project_root / path  # type: ignore[attr-defined]
        return path

    @staticmethod
    def _read_model_hash(path: Path) -> str:
        """Prefer a fresh sidecar hash before falling back to a full file scan."""
        hash_path = path.with_suffix(path.suffix + ".sha256")
        try:
            model_stat = path.stat()
            if hash_path.exists():
                hash_stat = hash_path.stat()
                if hash_stat.st_mtime_ns >= model_stat.st_mtime_ns:
                    digest = hash_path.read_text(encoding="utf-8").strip().lower()
                    if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                        return digest
        except OSError:
            pass
        return compute_file_hash(path)

    # --- gate loading ---

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

    # --- config overrides ---

    def _apply_config_overrides(self, skip_gate: _SkipGateLike) -> None:
        """YAML 設定でモデル内 config をオーバーライド."""
        config: FillTestConfig = self._config  # type: ignore[attr-defined]
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
            f"path={self._gate_path}"  # type: ignore[attr-defined]
        )

    def _apply_warm_start(self, skip_gate: _SkipGateLike) -> None:
        """096# warm_start: 直近 P(AS) 履歴を復元."""
        config: FillTestConfig = self._config  # type: ignore[attr-defined]
        if not config.skip_gate_adaptive_threshold:
            return
        try:
            from scripts.v460.ml.skip_gate import warm_start_skip_gate_thresholds
            warm_start_skip_gate_thresholds(
                skip_gate,
                config.results_dir,
                window=config.skip_gate_adaptive_window,
            )
        except Exception as ws_err:
            logger.warning(f"[skip_gate] Warm start failed (non-fatal): {ws_err}")

    def _inject_calibrator(self, skip_gate: _SkipGateLike) -> None:
        """139# §8-#1: ScoreCalibrator を SkipGate に注入."""
        config: FillTestConfig = self._config  # type: ignore[attr-defined]
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

    # --- side / alt model loading ---

    def _load_side_models(self, skip_gate_cls: _SkipGateClassLike) -> None:
        """141# P1-01: side 別モデルをロード."""
        config: FillTestConfig = self._config  # type: ignore[attr-defined]
        for side, attr_gate, attr_path, attr_hash in self._SIDE_MODEL_SLOTS:  # type: ignore[attr-defined]
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
                setattr(self, attr_hash, self._read_model_hash(gate_path))
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
        """188# C-1: ev_weighted 用の副 horizon モデルをロード."""
        config: FillTestConfig = self._config  # type: ignore[attr-defined]
        for side, attr_gate, attr_path, attr_hash, config_key in self._ALT_MODEL_SLOTS:  # type: ignore[attr-defined]
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
                setattr(self, attr_hash, self._read_model_hash(gate_path))
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

    # --- hot-reload ---

    def _check_and_reload_model(self) -> None:
        """126# モデルファイル変更を検出してリロード."""
        # 236# hasattr 排除: クラスレベルで None (= __init__ 未実行) をガード
        if self._last_reload_check is None:  # type: ignore[attr-defined]
            return
        now = time.monotonic()
        # 158# YAML 外部化: config から hot_reload 間隔を取得
        config: FillTestConfig = self._config  # type: ignore[attr-defined]
        interval = config.hot_reload_check_interval_sec
        if now - self._last_reload_check < interval:  # type: ignore[attr-defined]
            return
        self._last_reload_check = now  # type: ignore[attr-defined]

        # 143# A.1 #1: side 別モデルは unified 更新とは独立にチェック
        self._check_and_reload_side_models()

        if self._gate_path is None or not self._gate_path.exists():  # type: ignore[attr-defined]
            return

        new_hash = self._read_model_hash(self._gate_path)  # type: ignore[attr-defined]
        if new_hash == self._model_file_hash or not new_hash:  # type: ignore[attr-defined]
            return

        # モデルファイルが変更された — リロード
        logger.info(
            f"[skip_gate] 126# Model file changed detected "
            f"(hash {self._model_file_hash[:8]}→{new_hash[:8]}). Reloading..."  # type: ignore[attr-defined]
        )
        try:
            from scripts.v460.ml.skip_gate import SkipGate
            skip_gate_cls = cast("_SkipGateClassLike", SkipGate)
            new_gate = self._load_gate_from_path(
                skip_gate_cls,
                self._gate_path,  # type: ignore[attr-defined]
                apply_warm_start=True,
            )
            self._skip_gate = new_gate  # type: ignore[attr-defined]
            # 175# _inject_calibrator は _load_gate_from_path 内で既に実行済み
            self._model_file_hash = new_hash  # type: ignore[attr-defined]
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

    def _check_and_reload_side_models(self) -> None:
        """141# side 別モデルの変更検出 + リロード."""
        try:
            from scripts.v460.ml.skip_gate import SkipGate
        except Exception as e:
            logger.debug(f"[skip_gate] side hot-reload skipped (import failed): {e}")
            return
        skip_gate_cls = cast("_SkipGateClassLike", SkipGate)

        for side, attr_gate, attr_path, attr_hash in self._SIDE_MODEL_SLOTS:  # type: ignore[attr-defined]
            gate_path: Path | None = getattr(self, attr_path, None)
            if gate_path is None:
                # パス未設定の場合: config から新規ロード試行
                config: FillTestConfig = self._config  # type: ignore[attr-defined]
                model_path_str = getattr(config, f"skip_gate_model_path_{side}", None)
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
                    setattr(self, attr_hash, self._read_model_hash(gate_path))
                    logger.info(f"[skip_gate] 141# {side} model first load via hot-reload: {gate_path}")
                except Exception as e:
                    logger.warning(f"[skip_gate] 141# {side} model first load failed: {e}")
                continue

            if not gate_path.exists():
                continue
            old_hash = getattr(self, attr_hash, "")
            new_hash = self._read_model_hash(gate_path)
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
