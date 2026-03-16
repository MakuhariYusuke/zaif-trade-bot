"""057# ML モジュールのテスト."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import scripts.v460.ml.data_loader as ml_data_loader
from scripts.v460.ml.as_classifier import (
    ASModelMetrics,
    evaluate_skip_policy,
    train_as_classifier,
)
from scripts.v460.ml.data_loader import (
    build_as_features,
    build_fill_features,
    clear_fill_records_cache,
    get_fill_records_cache_stats,
    load_fill_records,
)
from scripts.v460.ml.fill_classifier import FillModelMetrics, train_fill_classifier
from tests.unit.v460._real_data_test_helpers import (
    latest_fill_records_file,
    write_minimum_feature_ready_fill_sample,
)


# ======================================================================
# Fixtures
# ======================================================================


@lru_cache(maxsize=1)
def _cached_synthetic_fill_df() -> pd.DataFrame:
    """合成 fill records: 100件のテストデータ."""
    rng = np.random.RandomState(42)
    n = 50
    timestamps = np.arange(1700000000, 1700000000 + n * 120, 120, dtype=float)
    sides = rng.choice(["buy", "sell"], n)
    prices = 14_500_000 + rng.randn(n) * 10_000
    fill_prices = prices + rng.randn(n) * 500

    # AS: queue_wait < 20s → 70% AS, >= 20s → 30% AS
    queue_waits = rng.exponential(25, n) + 5
    as_probs = np.where(queue_waits < 20, 0.7, 0.3)
    adverse = rng.binomial(1, as_probs).astype(bool)

    mid_at_fill = fill_prices + rng.randn(n) * 100
    pnl_30s = np.where(adverse, -rng.exponential(3, n), rng.exponential(2, n))
    mid_30s = mid_at_fill + np.where(
        sides == "buy",
        pnl_30s / 10000 * mid_at_fill,
        -pnl_30s / 10000 * mid_at_fill,
    )

    filled = np.ones(n, dtype=bool)
    filled[rng.choice(n, 15, replace=False)] = False  # 15% timeout

    rows = []
    for i in range(n):
        row: dict[str, Any] = {
            "cycle_id": f"test_{i}",
            "timestamp": timestamps[i],
            "side": sides[i],
            "order_price": prices[i],
            "order_quantity": 0.001,
            "filled": bool(filled[i]),
            "fill_price": float(fill_prices[i]) if filled[i] else None,
            "cancelled": not bool(filled[i]),
            "cancel_reason": None if filled[i] else "timeout",
            "queue_wait_sec": float(queue_waits[i]),
            "adverse_selected_raw": bool(adverse[i]) if filled[i] else None,
            "adverse_selected": bool(adverse[i]) if filled[i] else None,
            "mid_at_fill": float(mid_at_fill[i]) if filled[i] else None,
            "mid_30s_after": float(mid_30s[i]) if filled[i] else None,
            "post_fill_30s_pnl": float(pnl_30s[i]) if filled[i] else None,
            "spread_at_order": float(rng.uniform(500, 3000)),
            "spread_offset_ratio": float(rng.uniform(0.03, 0.15)),
            "regime": rng.choice(["trending", "ranging", "high_vol", "unknown"]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_fill_df() -> pd.DataFrame:
    return _cached_synthetic_fill_df().copy(deep=True)


@pytest.fixture
def as_training_data_small(
    synthetic_fill_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """AS classifier 向けの軽量共有学習データ."""
    X, y = build_as_features(synthetic_fill_df)
    X = X.head(20)
    y = y.loc[X.index]
    return X, y


@pytest.fixture
def fill_training_data_small(
    synthetic_fill_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fill classifier 向けの軽量共有学習データ."""
    X, y = build_fill_features(synthetic_fill_df)
    X = X.head(20)
    y = y.loc[X.index]
    return X, y


_REAL_DATA_CANDIDATE_LIMITS = (94, 100, 160, 220)


@lru_cache(maxsize=1)
def _cached_latest_fill_records_file() -> Path | None:
    return latest_fill_records_file()


def _load_minimum_real_as_fill_df(tmp_path: Path) -> pd.DataFrame:
    latest_file = _cached_latest_fill_records_file()
    if latest_file is None:
        return pd.DataFrame()
    return write_minimum_feature_ready_fill_sample(
        latest_file=latest_file,
        tmp_path=tmp_path,
        load_fn=lambda path: load_fill_records(path, max_files=1),
        feature_builder=build_as_features,
        candidate_limits=_REAL_DATA_CANDIDATE_LIMITS,
    )


# ======================================================================
# Data Loader Tests
# ======================================================================


class Test057DataLoader:
    """data_loader のテスト."""

    def test_build_as_features_shape(self, synthetic_fill_df: pd.DataFrame) -> None:
        """AS 特徴量の shape が妥当."""
        X, y = build_as_features(synthetic_fill_df)
        assert len(X) > 0
        assert len(X) == len(y)
        assert X.shape[1] >= 5  # 最低5特徴量

    def test_build_as_features_no_nan(self, synthetic_fill_df: pd.DataFrame) -> None:
        """AS 特徴量に NaN がない."""
        X, y = build_as_features(synthetic_fill_df)
        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_build_as_features_labels_binary(self, synthetic_fill_df: pd.DataFrame) -> None:
        """ラベルが 0/1 の二値."""
        _, y = build_as_features(synthetic_fill_df)
        assert set(y.unique()).issubset({0, 1})

    def test_build_as_features_require_spread(self, synthetic_fill_df: pd.DataFrame) -> None:
        """require_spread=True で spread 必須."""
        X, y = build_as_features(synthetic_fill_df, require_spread=True)
        assert "spread_jpy" in X.columns

    def test_build_fill_features_shape(self, synthetic_fill_df: pd.DataFrame) -> None:
        """Fill 特徴量の shape."""
        X, y = build_fill_features(synthetic_fill_df)
        assert len(X) >= len(synthetic_fill_df) * 0.5
        assert X.shape[1] >= 3

    def test_build_fill_features_labels(self, synthetic_fill_df: pd.DataFrame) -> None:
        """Fill ラベルが 0/1."""
        _, y = build_fill_features(synthetic_fill_df)
        assert 0 in y.values
        assert 1 in y.values

    def test_build_as_insufficient_data(self) -> None:
        """データ不足で ValueError."""
        df = pd.DataFrame({
            "filled": [True],
            "adverse_selected_raw": [True],
            "queue_wait_sec": [10.0],
            "side": ["buy"],
            "timestamp": [1.0],
            "fill_price": [1.0],
            "mid_at_fill": [1.0],
        })
        with pytest.raises(ValueError, match="Insufficient"):
            build_as_features(df)

    def test_load_fill_records_excludes_emergency_and_deduplicates(self, tmp_path: Path) -> None:
        """data_loader は emergency を読まず、primary 間の重複は除外する."""
        primary_a = {
            "cycle_id": "dup_1",
            "timestamp": 1700000000.0,
            "side": "buy",
            "order_price": 15000000.0,
            "order_quantity": 0.001,
            "filled": True,
            "adverse_selected_raw": True,
            "queue_wait_sec": 10.0,
        }
        primary_b = {
            "cycle_id": "uniq_1",
            "timestamp": 1700000060.0,
            "side": "sell",
            "order_price": 15000010.0,
            "order_quantity": 0.001,
            "filled": False,
            "adverse_selected_raw": None,
            "queue_wait_sec": 12.0,
        }
        emergency = {
            "cycle_id": "emg_1",
            "timestamp": 1700000120.0,
            "side": "buy",
            "order_price": 15000020.0,
            "order_quantity": 0.001,
            "filled": True,
            "adverse_selected_raw": False,
            "queue_wait_sec": 8.0,
        }

        (tmp_path / "fill_records_20260101.jsonl").write_text(
            json.dumps(primary_a) + "\n",
            encoding="utf-8",
        )
        (tmp_path / "fill_records_20260102.jsonl").write_text(
            "\n".join([json.dumps(primary_a), json.dumps(primary_b)]) + "\n",
            encoding="utf-8",
        )
        emergency_dir = tmp_path / "emergency"
        emergency_dir.mkdir()
        (emergency_dir / "emergency_20260101.jsonl").write_text(
            json.dumps(emergency) + "\n",
            encoding="utf-8",
        )

        df = load_fill_records(tmp_path)
        assert sorted(df["cycle_id"].tolist()) == ["dup_1", "uniq_1"]


# ======================================================================
# AS Classifier Tests
# ======================================================================


class Test057ASClassifier:
    """AS 分類器のテスト."""

    def test_train_returns_metrics(
        self,
        as_training_data_small: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """学習が完了し metrics を返す."""
        X, y = as_training_data_small
        metrics, model, scaler, oof_probs = train_as_classifier(X, y, model_type="lr", n_splits=2)
        assert isinstance(metrics, ASModelMetrics)
        assert metrics.n_samples == len(X)
        assert 0 <= metrics.roc_auc_mean <= 1
        assert oof_probs is not None

    def test_train_gb_model(
        self,
        as_training_data_small: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """GradientBoosting で学習."""
        X, y = as_training_data_small
        metrics, model, scaler, _ = train_as_classifier(
            X,
            y,
            model_type="gb",
            n_splits=2,
            gb_n_estimators=3,
        )
        assert metrics.feature_importances is not None
        assert len(metrics.feature_importances) == X.shape[1]

    def test_model_predicts(
        self,
        as_training_data_small: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """学習済みモデルが predict_proba を返す."""
        X, y = as_training_data_small
        _, model, pipeline, _ = train_as_classifier(X, y, model_type="lr", n_splits=2)
        # 059#: pipeline は完全な Pipeline (imputer + scaler + model)
        probs = pipeline.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_skip_policy_with_pnl(
        self,
        synthetic_fill_df: pd.DataFrame,
        as_training_data_small: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """PnL 付きでスキップ効果計算."""
        X, y = as_training_data_small
        pnl = synthetic_fill_df.loc[X.index, "post_fill_30s_pnl"].astype(float)
        metrics, model, scaler, oof_probs = train_as_classifier(
            X, y, pnl, model_type="lr", n_splits=2
        )
        assert isinstance(metrics.skip_top20_pnl_improvement_bps, float)

    def test_evaluate_skip_policy(
        self,
        synthetic_fill_df: pd.DataFrame,
        as_training_data_small: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """スキップポリシーの DataFrame が返る."""
        X, y = as_training_data_small
        pnl = synthetic_fill_df.loc[X.index, "post_fill_30s_pnl"].astype(float)
        oof_probs = np.linspace(0.2, 0.8, len(X), dtype=np.float64)
        result = evaluate_skip_policy(
            X,
            y,
            pnl,
            model=object(),
            scaler=object(),
            oof_probs=oof_probs,
        )
        assert isinstance(result, pd.DataFrame)
        assert "threshold" in result.columns
        assert "pnl_improvement_bps" in result.columns
        assert len(result) == 6  # default thresholds


# ======================================================================
# Fill Classifier Tests
# ======================================================================


class Test057FillClassifier:
    """Fill 分類器のテスト."""

    def test_train_returns_metrics(
        self,
        fill_training_data_small: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """学習が完了し metrics を返す."""
        X, y = fill_training_data_small
        metrics, model, scaler = train_fill_classifier(X, y, model_type="lr", n_splits=2)
        assert isinstance(metrics, FillModelMetrics)
        assert metrics.n_samples == len(X)

    def test_train_gb(
        self,
        fill_training_data_small: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """GradientBoosting で学習."""
        X, y = fill_training_data_small
        metrics, model, scaler = train_fill_classifier(
            X,
            y,
            model_type="gb",
            n_splits=2,
            gb_n_estimators=3,
        )
        assert metrics.feature_importances is not None

    def test_fill_rate_correct(
        self,
        fill_training_data_small: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """fill_rate が正しい."""
        X, y = fill_training_data_small
        metrics, _, _ = train_fill_classifier(X, y, model_type="lr", n_splits=2)
        assert abs(metrics.fill_rate - y.mean()) < 0.01


# ======================================================================
# Integration: load_fill_records (実データ)
# ======================================================================


@pytest.mark.slow
@pytest.mark.integration
class Test057Integration:
    """実データが存在する場合の統合テスト."""

    @pytest.fixture
    def real_data_available(self) -> bool:
        """実データの有無."""
        return _cached_latest_fill_records_file() is not None

    def test_load_real_data(self, real_data_available: bool, tmp_path: Path) -> None:
        """実データのロードと AS 特徴量構築."""
        if not real_data_available:
            pytest.skip("No real fill records")
        df = _load_minimum_real_as_fill_df(tmp_path)
        assert len(df) >= 30
        X, y = build_as_features(df)
        assert len(X) >= 10


class Test057DataLoaderCache:
    """load_fill_records のファイル更新連動キャッシュ."""

    def test_cache_is_bounded_and_clearable(self, tmp_path: Path) -> None:
        clear_fill_records_cache()
        for idx in range(3):
            subdir = tmp_path / f"run_{idx}"
            subdir.mkdir()
            payload = {
                "cycle_id": f"c{idx}",
                "timestamp": 1700000000.0 + idx,
                "side": "buy",
                "order_price": 15000000.0 + idx,
                "order_quantity": 0.001,
                "filled": True,
                "adverse_selected_raw": True,
                "queue_wait_sec": 10.0,
            }
            (subdir / "fill_records_20260101.jsonl").write_text(
                json.dumps(payload) + "\n",
                encoding="utf-8",
            )

        with patch.object(ml_data_loader, "_FILL_RECORDS_CACHE_MAX_ENTRIES", 2):
            for idx in range(3):
                load_fill_records(tmp_path / f"run_{idx}")

        assert get_fill_records_cache_stats()["fill_records_cache_entries"] <= 2
        clear_fill_records_cache()
        assert get_fill_records_cache_stats()["fill_records_cache_entries"] == 0

    def test_cache_invalidates_when_file_changes(self, tmp_path: Path) -> None:
        p = tmp_path / "fill_records_20260101.jsonl"
        first = {
            "cycle_id": "c1",
            "timestamp": 1700000000.0,
            "side": "buy",
            "order_price": 15000000.0,
            "order_quantity": 0.001,
            "filled": True,
            "adverse_selected_raw": True,
            "queue_wait_sec": 10.0,
        }
        second = {
            "cycle_id": "c2",
            "timestamp": 1700000060.0,
            "side": "sell",
            "order_price": 15000010.0,
            "order_quantity": 0.001,
            "filled": False,
            "adverse_selected_raw": None,
            "queue_wait_sec": 12.0,
        }
        p.write_text(json.dumps(first) + "\n", encoding="utf-8")

        df1 = load_fill_records(tmp_path)
        df2 = load_fill_records(tmp_path)
        assert len(df1) == 1
        assert len(df2) == 1
        assert df1 is not df2

        p.write_text("\n".join([json.dumps(first), json.dumps(second)]) + "\n", encoding="utf-8")
        df3 = load_fill_records(tmp_path)
        assert len(df3) == 2

    def test_max_files_loads_latest_only(self, tmp_path: Path) -> None:
        older = {
            "cycle_id": "old_1",
            "timestamp": 1700000000.0,
            "side": "buy",
            "order_price": 15000000.0,
            "order_quantity": 0.001,
            "filled": True,
            "adverse_selected_raw": False,
            "queue_wait_sec": 11.0,
        }
        newer = {
            "cycle_id": "new_1",
            "timestamp": 1700001000.0,
            "side": "sell",
            "order_price": 15000100.0,
            "order_quantity": 0.001,
            "filled": True,
            "adverse_selected_raw": True,
            "queue_wait_sec": 9.0,
        }
        (tmp_path / "fill_records_20260101.jsonl").write_text(
            json.dumps(older) + "\n",
            encoding="utf-8",
        )
        (tmp_path / "fill_records_20260102.jsonl").write_text(
            json.dumps(newer) + "\n",
            encoding="utf-8",
        )

        df = load_fill_records(tmp_path, max_files=1)
        assert df["cycle_id"].tolist() == ["new_1"]

    def test_max_files_invalid_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_files must be >= 1"):
            load_fill_records(tmp_path, max_files=0)

    def test_as_gb_n_estimators_invalid_raises(self, synthetic_fill_df: pd.DataFrame) -> None:
        X, y = build_as_features(synthetic_fill_df)
        with pytest.raises(ValueError, match="gb_n_estimators must be >= 1"):
            train_as_classifier(X.head(20), y.head(20), model_type="gb", n_splits=2, gb_n_estimators=0)

    def test_fill_gb_n_estimators_invalid_raises(self, synthetic_fill_df: pd.DataFrame) -> None:
        X, y = build_fill_features(synthetic_fill_df)
        with pytest.raises(ValueError, match="gb_n_estimators must be >= 1"):
            train_fill_classifier(X.head(20), y.head(20), model_type="gb", n_splits=2, gb_n_estimators=0)
