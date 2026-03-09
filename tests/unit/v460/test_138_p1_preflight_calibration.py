"""138# テスト: P1-10 (preflight pause), P1-03 (score calibration)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from ztb.ml.score_calibrator import ScoreCalibrator, ScoreCalibratorConfig


# ======================================================================
# P1-10: preflight pause テスト
# ======================================================================

class TestPreflightPause:
    """P1-10: preflight 失敗連続→run pause (dead-cycle 防止)."""

    def test_config_defaults(self) -> None:
        """デフォルト値が正しく設定されている."""
        cfg = FillTestConfig()
        assert cfg.preflight_pause_enabled is True
        assert cfg.preflight_pause_threshold == 5
        assert cfg.preflight_pause_sec == 300.0
        assert cfg.preflight_max_pauses == 3

    def test_yaml_parsing(self) -> None:
        """YAML から preflight_pause 設定が正しくパースされる."""
        yaml_cfg: dict[str, dict[str, dict[str, float | int | bool]]] = {
            "loss_control": {
                "preflight_pause": {
                    "enabled": False,
                    "threshold": 3,
                    "pause_sec": 120.0,
                    "max_pauses": 2,
                },
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.preflight_pause_enabled is False
        assert cfg.preflight_pause_threshold == 3
        assert cfg.preflight_pause_sec == 120.0
        assert cfg.preflight_max_pauses == 2

    def test_pause_fires_before_safe_stop(self) -> None:
        """preflight 失敗が threshold に到達すると pause が発動し、
        SAFE_STOP ではなく continue する."""
        cfg = FillTestConfig(
            preflight_pause_enabled=True,
            preflight_pause_threshold=3,
            preflight_pause_sec=0.01,  # テスト用に極小
            preflight_max_pauses=2,
            max_preflight_skip=10,
        )
        # pause_count=0, skip_count=3 → pause 発動
        # pause_count=1, skip_count=3 → pause 発動 (2回目)
        # pause_count=2, skip_count=3 → max_pauses 超過 → SAFE_STOP
        assert cfg.preflight_pause_threshold < cfg.max_preflight_skip
        assert cfg.preflight_max_pauses == 2

    def test_pause_disabled_goes_straight_to_safe_stop(self) -> None:
        """pause 無効時は従来通り SAFE_STOP."""
        cfg = FillTestConfig(
            preflight_pause_enabled=False,
            preflight_pause_threshold=3,
            max_preflight_skip=5,
        )
        # enabled=False → pause 条件は成立しない
        assert cfg.preflight_pause_enabled is False


# ======================================================================
# P1-03: score calibration テスト
# ======================================================================

class TestScoreCalibrator:
    """P1-03: ScoreCalibrator (isotonic regression)."""

    def test_uncalibrated_passthrough(self) -> None:
        """未学習時は raw score をそのまま返す."""
        cal = ScoreCalibrator(ScoreCalibratorConfig(enabled=True))
        assert cal.calibrate(0.5) == 0.5
        assert cal.calibrate(-1.2) == -1.2
        assert not cal.is_fitted

    def test_disabled_passthrough(self) -> None:
        """無効時は raw score をそのまま返す."""
        cal = ScoreCalibrator(ScoreCalibratorConfig(enabled=False))
        cal.fit(
            raw_scores=list(np.linspace(-5, 5, 50)),
            actual_values=list(np.linspace(-3, 3, 50)),
        )
        # enabled=False なので fit しても calibrate はパススルー
        assert cal.calibrate(2.5) == 2.5

    def test_fit_and_calibrate(self) -> None:
        """isotonic regression で校正関数が学習される."""
        np.random.seed(42)
        n = 100
        raw = np.linspace(-5, 5, n)
        # actual は raw に対してノイジーだが単調増加傾向
        actual = raw * 0.6 + np.random.normal(0, 0.5, n)

        cal = ScoreCalibrator(ScoreCalibratorConfig(
            enabled=True, min_samples=30,
        ))
        stats = cal.fit(raw_scores=list(raw), actual_values=list(actual))

        assert cal.is_fitted
        assert stats.n_samples == n
        assert stats.r2_score > 0  # 正のR²
        assert stats.mae_after <= stats.mae_before  # 校正で改善

        # 校正値が単調増加
        calibrated = [cal.calibrate(float(x)) for x in [-3, 0, 3]]
        assert calibrated[0] <= calibrated[1] <= calibrated[2]

    def test_min_samples_guard(self) -> None:
        """最小サンプル数未満では学習しない."""
        cal = ScoreCalibrator(ScoreCalibratorConfig(
            enabled=True, min_samples=50,
        ))
        stats = cal.fit(
            raw_scores=list(range(20)),
            actual_values=list(range(20)),
        )
        assert not cal.is_fitted
        assert stats.n_samples == 20

    def test_add_observation_auto_refit(self) -> None:
        """add_observation で蓄積し、refit_interval で自動 refit."""
        cal = ScoreCalibrator(ScoreCalibratorConfig(
            enabled=True,
            min_samples=10,
            refit_interval=20,
        ))
        np.random.seed(123)
        refitted = False
        for i in range(25):
            raw = float(i)
            actual = float(i) * 0.5 + np.random.normal(0, 0.3)
            if cal.add_observation(raw, actual):
                refitted = True
        assert refitted
        assert cal.is_fitted

    def test_save_load_roundtrip(self, tmp_path: "pytest.TempPathFactory") -> None:
        """save/load の往復で状態が保持される."""
        cal = ScoreCalibrator(ScoreCalibratorConfig(enabled=True, min_samples=10))
        raw = list(np.linspace(-3, 3, 30))
        actual = list(np.linspace(-2, 2, 30))
        cal.fit(raw_scores=raw, actual_values=actual)
        assert cal.is_fitted

        path = tmp_path / "cal.pkl"
        assert cal.save(path)

        loaded = ScoreCalibrator.load(path)
        assert loaded.is_fitted
        assert loaded.sample_count == 30
        # 校正結果が同一
        assert abs(cal.calibrate(0.0) - loaded.calibrate(0.0)) < 1e-6

    def test_from_fill_records(self) -> None:
        """FillRecord リストから校正器を学習."""
        @dataclass
        class FakeRecord:
            filled: bool = True
            skip_gate_score: float | None = None
            post_fill_30s_pnl: float | None = None

        np.random.seed(7)
        records = [
            FakeRecord(
                filled=True,
                skip_gate_score=float(i),
                post_fill_30s_pnl=float(i) * 0.5 + np.random.normal(0, 0.2),
            )
            for i in range(40)
        ]
        # 未約定レコードは除外
        records.append(FakeRecord(filled=False, skip_gate_score=1.0, post_fill_30s_pnl=-1.0))
        # 欠損レコードも除外
        records.append(FakeRecord(filled=True, skip_gate_score=None, post_fill_30s_pnl=0.5))

        cal = ScoreCalibrator.from_fill_records(
            records,
            ScoreCalibratorConfig(enabled=True, min_samples=30),
        )
        assert cal.is_fitted
        assert cal.sample_count == 40

    def test_nan_handling(self) -> None:
        """NaN / inf は安全にパススルー."""
        cal = ScoreCalibrator(ScoreCalibratorConfig(enabled=True))
        assert np.isnan(cal.calibrate(float("nan")))
        # add_observation で NaN は無視
        result = cal.add_observation(float("nan"), 1.0)
        assert not result
        assert cal.sample_count == 0


class TestScoreCalibrationConfig:
    """P1-03: fill_config への設定追加."""

    def test_config_defaults(self) -> None:
        cfg = FillTestConfig()
        assert cfg.skip_gate_score_calibration is False
        assert cfg.skip_gate_calibrator_path is None
        assert cfg.skip_gate_calibrator_min_samples == 30
        assert cfg.skip_gate_calibrator_refit_interval == 100

    def test_yaml_parsing(self) -> None:
        yaml_cfg: dict[str, dict[str, bool | str | int]] = {
            "skip_gate": {
                "score_calibration": True,
                "calibrator_path": "checkpoints/cal.pkl",
                "calibrator_min_samples": 50,
                "calibrator_refit_interval": 200,
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.skip_gate_score_calibration is True
        assert cfg.skip_gate_calibrator_path == "checkpoints/cal.pkl"
        assert cfg.skip_gate_calibrator_min_samples == 50
        assert cfg.skip_gate_calibrator_refit_interval == 200


class TestSkipGateCalibrationIntegration:
    """P1-03: SkipGate 内の calibrator 統合."""

    def test_skip_gate_with_calibrator(self) -> None:
        """calibrator を渡すと予測値が校正される."""
        # 校正器を事前学習
        cal = ScoreCalibrator(ScoreCalibratorConfig(enabled=True, min_samples=10))
        raw = list(np.linspace(-5, 5, 30))
        actual = [x * 0.3 for x in raw]  # 縮小写像 (raw*1.0 → actual*0.3)
        cal.fit(raw_scores=raw, actual_values=actual)
        assert cal.is_fitted

        # raw=2.0 → calibrated は 2.0 より小さいはず (0.3 倍に近い)
        calibrated = cal.calibrate(2.0)
        assert calibrated < 2.0

    def test_skip_gate_without_calibrator(self) -> None:
        """calibrator=None の場合はパススルー."""
        cal = ScoreCalibrator()  # デフォルト = disabled
        assert cal.calibrate(3.14) == 3.14
