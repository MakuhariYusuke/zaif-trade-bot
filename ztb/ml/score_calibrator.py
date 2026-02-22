"""138# P1-03: SkipGate score calibration — isotonic regression による予測値校正.

raw model score (predicted_pnl_bps or P(AS)) を FillRecord の実績 PnL に基づき
isotonic regression で校正する。

- fit() で (raw_score, actual_pnl) ペアから校正関数を学習
- calibrate() で raw → calibrated 写像を適用
- save()/load() で永続化 (pickle)
- 最小サンプル数チェック、フォールバック (未学習時はパススルー)

用途:
  FillRecord.skip_gate_score (raw) → calibrated score
  → threshold 判定の精度向上 (skip すべき注文をより正確に識別)

参考: sklearn.isotonic.IsotonicRegression, sklearn.calibration.CalibratedClassifierCV
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MIN_SAMPLES_DEFAULT = 30  # 校正に必要な最小サンプル数


@dataclass
class CalibrationStats:
    """校正統計量."""

    n_samples: int = 0
    r2_score: float = 0.0
    mae_before: float = 0.0  # 校正前の平均絶対誤差
    mae_after: float = 0.0   # 校正後の平均絶対誤差
    improvement_pct: float = 0.0  # MAE 改善率 (%)


@dataclass
class ScoreCalibratorConfig:
    """ScoreCalibrator 設定."""

    enabled: bool = False
    min_samples: int = _MIN_SAMPLES_DEFAULT
    # 校正対象: "pnl" (predicted_pnl_bps → actual_pnl_bps) or "as" (P(AS) → actual_as_rate)
    mode: str = "pnl"
    # 増分更新: True で fit() に新データを追加、False で全置換
    incremental: bool = True
    # 自動再学習間隔 (新規レコード数)
    refit_interval: int = 100
    # 永続化パス (None の場合はインメモリのみ)
    persist_path: Optional[str] = None


class ScoreCalibrator:
    """Isotonic regression によるスコア校正器.

    Usage:
        cal = ScoreCalibrator(ScoreCalibratorConfig(enabled=True))
        cal.fit(raw_scores=[0.5, 0.3, ...], actual_values=[-1.2, 0.8, ...])
        calibrated = cal.calibrate(0.4)  # → 校正済みスコア
    """

    __slots__ = (
        "_config",
        "_model",
        "_is_fitted",
        "_raw_scores",
        "_actual_values",
        "_stats",
        "_records_since_fit",
    )

    def __init__(self, config: ScoreCalibratorConfig | None = None) -> None:
        self._config = config or ScoreCalibratorConfig()
        self._model: object | None = None
        self._is_fitted: bool = False
        # 増分データ蓄積用
        self._raw_scores: list[float] = []
        self._actual_values: list[float] = []
        self._stats: CalibrationStats = CalibrationStats()
        self._records_since_fit: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def config(self) -> ScoreCalibratorConfig:
        return self._config

    @property
    def stats(self) -> CalibrationStats:
        return self._stats

    @property
    def sample_count(self) -> int:
        return len(self._raw_scores)

    def add_observation(self, raw_score: float, actual_value: float) -> bool:
        """新規観測値を蓄積し、必要に応じて自動 refit.

        Args:
            raw_score: SkipGate の raw 予測値.
            actual_value: 実績値 (PnL bps or AS flag).

        Returns:
            True if refit が実行された.
        """
        if not self._config.enabled:
            return False
        if not np.isfinite(raw_score) or not np.isfinite(actual_value):
            return False

        self._raw_scores.append(raw_score)
        self._actual_values.append(actual_value)
        self._records_since_fit += 1

        # 自動 refit
        if (
            self._records_since_fit >= self._config.refit_interval
            and len(self._raw_scores) >= self._config.min_samples
        ):
            self.fit()
            return True
        return False

    def fit(
        self,
        raw_scores: list[float] | None = None,
        actual_values: list[float] | None = None,
    ) -> CalibrationStats:
        """Isotonic regression で校正関数を学習.

        Args:
            raw_scores: raw 予測値のリスト (None の場合は蓄積データを使用).
            actual_values: 実績値のリスト.

        Returns:
            CalibrationStats.
        """
        if raw_scores is not None and actual_values is not None:
            if self._config.incremental:
                self._raw_scores.extend(raw_scores)
                self._actual_values.extend(actual_values)
            else:
                self._raw_scores = list(raw_scores)
                self._actual_values = list(actual_values)

        n = len(self._raw_scores)
        if n < self._config.min_samples:
            logger.debug(
                f"[score_cal] Insufficient data: {n}/{self._config.min_samples}"
            )
            self._stats = CalibrationStats(n_samples=n)
            return self._stats

        X = np.array(self._raw_scores)
        y = np.array(self._actual_values)

        try:
            from sklearn.isotonic import IsotonicRegression

            ir = IsotonicRegression(
                increasing=True,  # 高スコア → 高 PnL (or 低 AS)
                out_of_bounds="clip",
            )
            ir.fit(X, y)
            self._model = ir
            self._is_fitted = True
            self._records_since_fit = 0

            # 統計量算出
            y_pred_before = X  # raw score = 校正前予測
            y_pred_after = ir.predict(X)
            mae_before = float(np.mean(np.abs(y - y_pred_before)))
            mae_after = float(np.mean(np.abs(y - y_pred_after)))
            ss_res = float(np.sum((y - y_pred_after) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            improvement = (
                (mae_before - mae_after) / mae_before * 100
                if mae_before > 0 else 0.0
            )

            self._stats = CalibrationStats(
                n_samples=n,
                r2_score=round(r2, 4),
                mae_before=round(mae_before, 4),
                mae_after=round(mae_after, 4),
                improvement_pct=round(improvement, 1),
            )

            logger.info(
                f"[score_cal] Fitted: n={n}, R²={r2:.3f}, "
                f"MAE {mae_before:.3f}→{mae_after:.3f} "
                f"({improvement:+.1f}%)"
            )

            # 自動永続化
            if self._config.persist_path:
                self.save(Path(self._config.persist_path))

        except ImportError:
            logger.warning(
                "[score_cal] sklearn not available, calibration disabled"
            )
            self._stats = CalibrationStats(n_samples=n)
        except Exception as e:
            logger.error(f"[score_cal] fit failed: {e}")
            self._stats = CalibrationStats(n_samples=n)

        return self._stats

    def calibrate(self, raw_score: float) -> float:
        """raw score → 校正済みスコアに変換.

        未学習時は raw score をそのまま返す (パススルー).

        Args:
            raw_score: SkipGate の raw 予測値.

        Returns:
            校正済みスコア.
        """
        if not self._config.enabled:
            return raw_score
        if not self._is_fitted or self._model is None:
            return raw_score
        if not np.isfinite(raw_score):
            return raw_score

        try:
            calibrated = float(self._model.predict([raw_score])[0])  # type: ignore[union-attr]
            return calibrated
        except Exception:
            return raw_score

    def save(self, path: Path) -> bool:
        """校正器の状態を pickle で永続化.

        Args:
            path: 保存先パス.

        Returns:
            成功時 True.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "model": self._model,
                "is_fitted": self._is_fitted,
                "raw_scores": self._raw_scores,
                "actual_values": self._actual_values,
                "stats": self._stats,
                "config": self._config,
            }
            with open(path, "wb") as f:
                pickle.dump(state, f)
            logger.debug(f"[score_cal] Saved to {path}")
            return True
        except Exception as e:
            logger.error(f"[score_cal] Save failed: {e}")
            return False

    @classmethod
    def load(cls, path: Path) -> "ScoreCalibrator":
        """pickle から校正器を復元.

        Args:
            path: pkl ファイルパス.

        Returns:
            復元された ScoreCalibrator (ファイルなし / エラー時はデフォルト).
        """
        if not path.exists():
            logger.debug(f"[score_cal] No calibrator at {path}, returning default")
            return cls()
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)  # noqa: S301
            cal = cls(state.get("config"))
            cal._model = state.get("model")
            cal._is_fitted = state.get("is_fitted", False)
            cal._raw_scores = state.get("raw_scores", [])
            cal._actual_values = state.get("actual_values", [])
            cal._stats = state.get("stats", CalibrationStats())
            logger.info(
                f"[score_cal] Loaded from {path}: "
                f"n={cal.sample_count}, fitted={cal.is_fitted}"
            )
            return cal
        except Exception as e:
            logger.error(f"[score_cal] Load failed: {e}")
            return cls()

    @staticmethod
    def from_fill_records(
        records: list[object],
        config: ScoreCalibratorConfig | None = None,
    ) -> "ScoreCalibrator":
        """FillRecord リストから校正器を学習.

        filled かつ skip_gate_score と post_fill_30s_pnl が有効なレコードを使用。

        Args:
            records: FillRecord のリスト.
            config: 設定.

        Returns:
            学習済み ScoreCalibrator.
        """
        cal = ScoreCalibrator(config or ScoreCalibratorConfig(enabled=True))
        scores: list[float] = []
        actuals: list[float] = []

        for r in records:
            score = getattr(r, "skip_gate_score", None)
            pnl = getattr(r, "post_fill_30s_pnl", None)
            filled = getattr(r, "filled", False)
            if filled and score is not None and pnl is not None:
                if np.isfinite(score) and np.isfinite(pnl):
                    scores.append(float(score))
                    actuals.append(float(pnl))

        if scores:
            cal.fit(raw_scores=scores, actual_values=actuals)
        return cal
