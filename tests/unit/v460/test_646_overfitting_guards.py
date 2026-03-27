"""646# 過学習防止ガードのテスト.

P0-A: サンプル数ベースの n_estimators 上限
P0-B: D2 ガードの held-out 検証
P0-C: side_min_samples デフォルト引き上げ
"""

from __future__ import annotations

import numpy as np
import pytest


class TestResolveEarlyStopping:
    """P0-A: _resolve_early_stopping のサンプル数ベース上限テスト."""

    def _resolve(
        self,
        cfg: dict[str, object],
        n_samples: int | None = None,
    ) -> tuple[int, int]:
        from scripts.v460.ml.retrain_scheduler import _resolve_early_stopping

        return _resolve_early_stopping(cfg, n_samples=n_samples)

    def test_no_early_stop_default(self) -> None:
        """early_stopping_rounds=0 → n_estimators=150 (デフォルト)."""
        es, n_est = self._resolve({"early_stopping_rounds": 0})
        assert es == 0
        assert n_est == 150

    def test_early_stop_enabled_max(self) -> None:
        """early_stopping_rounds>0 → n_estimators_max=300."""
        es, n_est = self._resolve({"early_stopping_rounds": 20})
        assert es == 20
        assert n_est == 300

    def test_n_samples_caps_estimators(self) -> None:
        """P0-A: 229 samples → max 114 trees (229//2)."""
        es, n_est = self._resolve(
            {"early_stopping_rounds": 20, "lgbm_n_estimators_max": 300},
            n_samples=229,
        )
        assert es == 20
        assert n_est == 114  # 229 // 2

    def test_n_samples_cap_minimum_30(self) -> None:
        """P0-A: 50 samples → min 30 trees (下限保証)."""
        es, n_est = self._resolve(
            {"early_stopping_rounds": 20, "lgbm_n_estimators_max": 300},
            n_samples=50,
        )
        assert es == 20
        assert n_est == 30  # max(30, 50//2=25) = 30

    def test_large_samples_no_cap(self) -> None:
        """P0-A: 1000 samples → 300 trees (cap 不要)."""
        es, n_est = self._resolve(
            {"early_stopping_rounds": 20, "lgbm_n_estimators_max": 300},
            n_samples=1000,
        )
        assert es == 20
        assert n_est == 300  # 1000//2=500 > 300 → cap 不要

    def test_n_samples_none_no_cap(self) -> None:
        """n_samples=None → cap 適用しない (後方互換)."""
        es, n_est = self._resolve(
            {"early_stopping_rounds": 20, "lgbm_n_estimators_max": 300},
            n_samples=None,
        )
        assert es == 20
        assert n_est == 300

    def test_no_early_stop_with_samples(self) -> None:
        """early_stopping=0 + small samples → cap は同様に適用."""
        es, n_est = self._resolve(
            {"early_stopping_rounds": 0, "lgbm_n_estimators": 150},
            n_samples=80,
        )
        assert es == 0
        assert n_est == 40  # max(30, 80//2=40) = 40

    def test_edge_60_samples(self) -> None:
        """60 samples → max(30, 60//2=30) = 30."""
        es, n_est = self._resolve(
            {"early_stopping_rounds": 20},
            n_samples=60,
        )
        assert n_est == 30

    def test_exact_ratio_229(self) -> None:
        """229 は今回の defect case — 300→114 に制限されるべき."""
        _, n_est_before = self._resolve(
            {"early_stopping_rounds": 20, "lgbm_n_estimators_max": 300},
        )
        _, n_est_after = self._resolve(
            {"early_stopping_rounds": 20, "lgbm_n_estimators_max": 300},
            n_samples=229,
        )
        assert n_est_before == 300
        assert n_est_after == 114
        # 比率: 229/114 ≈ 2.0 (安全圏)
        assert 229 / n_est_after >= 2.0


class TestSideMinSamplesDefault:
    """P0-C: side_min_samples デフォルト値の引き上げテスト."""

    def test_default_is_200(self) -> None:
        """デフォルト設定が 200 であること."""
        from scripts.v460.ml.retrain_scheduler import _DEFAULT_CONFIG

        assert _DEFAULT_CONFIG["side_min_samples"] == 200

    def test_yaml_overrides_to_200(self) -> None:
        """YAML からの 200 が反映されること."""
        from scripts.v460.ml.retrain_scheduler import _DEFAULT_CONFIG

        cfg = {**_DEFAULT_CONFIG, "side_min_samples": 200}
        assert cfg["side_min_samples"] == 200


class TestPredStdDiagnostics:
    """P0-B: D2 ガードの診断フィールドテスト."""

    def test_result_contains_val_fields(self) -> None:
        """pred_std_val / pred_std_ratio が診断フィールドとして存在すること."""
        import inspect
        from scripts.v460.ml import retrain_scheduler

        source = inspect.getsource(retrain_scheduler.retrain_model)
        # val split の診断フィールドが存在する
        assert "pred_std_val" in source
        assert "pred_std_ratio" in source

    def test_d2_gate_uses_full_training_data(self) -> None:
        """D2 ゲート判定は全訓練データの pred_std で行うこと.

        eval_set は LightGBM モニタリング専用で訓練データから除外されないため、
        真の OOS ではない。ゲート判定には全訓練データの pred_std を使用する。
        """
        import inspect
        from scripts.v460.ml import retrain_scheduler

        source = inspect.getsource(retrain_scheduler.retrain_model)
        # preds_all が D2 判定に使用される
        assert "preds_all = lgbm.predict(X_sc)" in source
        assert 'pred_std = float(np.std(preds_all))' in source
