"""552# update_training_data ユニットテスト."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """テスト用 OHLCV DataFrame."""
    n = 100
    timestamps = pd.date_range("2026-03-01", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    close = 14_000_000 + rng.standard_normal(n).cumsum() * 10_000
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close + rng.uniform(-5000, 5000, n),
        "high": close + abs(rng.standard_normal(n)) * 10_000,
        "low": close - abs(rng.standard_normal(n)) * 10_000,
        "close": close,
        "volume": rng.uniform(0.01, 1.0, n),
    })


@pytest.fixture
def sample_parquet(sample_ohlcv: pd.DataFrame, tmp_path: Path) -> Path:
    """テスト用 parquet ファイル."""
    path = tmp_path / "test.parquet"
    # tz を除去して保存 (本番と同じ形式)
    df = sample_ohlcv.copy()
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["price_velocity"] = np.float32(0.0)
    df["micro_trend"] = np.float32(0.0)
    df.to_parquet(path, index=False)
    return path


class TestGetParquetLastTimestamp:
    """_get_parquet_last_timestamp のテスト."""

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        from scripts.v460.ml.update_training_data import _get_parquet_last_timestamp

        result = _get_parquet_last_timestamp(tmp_path / "no.parquet")
        assert result is None

    def test_returns_last_ts(self, sample_parquet: Path) -> None:
        from scripts.v460.ml.update_training_data import _get_parquet_last_timestamp

        result = _get_parquet_last_timestamp(sample_parquet)
        assert result is not None
        assert isinstance(result, datetime)

    def test_tz_aware(self, sample_parquet: Path) -> None:
        from scripts.v460.ml.update_training_data import _get_parquet_last_timestamp

        result = _get_parquet_last_timestamp(sample_parquet)
        assert result is not None
        assert result.tzinfo is not None


class TestHoursSinceLastUpdate:
    """_hours_since_last_update のテスト."""

    def test_nonexistent_returns_inf(self, tmp_path: Path) -> None:
        from scripts.v460.ml.update_training_data import _hours_since_last_update

        result = _hours_since_last_update(tmp_path / "nope.parquet")
        assert result == float("inf")

    def test_returns_positive(self, sample_parquet: Path) -> None:
        from scripts.v460.ml.update_training_data import _hours_since_last_update

        result = _hours_since_last_update(sample_parquet)
        assert result > 0


class TestMergeIntoParquet:
    """_merge_into_parquet のテスト."""

    def test_creates_new_parquet(self, tmp_path: Path) -> None:
        from scripts.v460.ml.update_training_data import _merge_into_parquet

        path = tmp_path / "new.parquet"
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-03-01", periods=5, freq="1min"),
            "open": [1.0] * 5,
            "close": [1.0] * 5,
        })
        n_added = _merge_into_parquet(path, df)
        assert n_added == 5
        assert path.exists()

    def test_deduplicates(self, sample_parquet: Path) -> None:
        from scripts.v460.ml.update_training_data import _merge_into_parquet

        existing = pd.read_parquet(sample_parquet)
        n_before = len(existing)

        # 重複データを追加
        dup = existing.tail(10).copy()
        n_added = _merge_into_parquet(sample_parquet, dup)
        assert n_added == 0

        after = pd.read_parquet(sample_parquet)
        assert len(after) == n_before

    def test_appends_new_rows(self, sample_parquet: Path) -> None:
        from scripts.v460.ml.update_training_data import _merge_into_parquet

        existing = pd.read_parquet(sample_parquet)
        n_before = len(existing)

        new_ts = pd.date_range("2026-04-01", periods=10, freq="1min")
        new_data = pd.DataFrame({
            "timestamp": new_ts,
            "open": [1.0] * 10,
            "high": [1.0] * 10,
            "low": [1.0] * 10,
            "close": [1.0] * 10,
            "volume": [0.1] * 10,
        })
        n_added = _merge_into_parquet(sample_parquet, new_data)
        assert n_added == 10
        after = pd.read_parquet(sample_parquet)
        assert len(after) == n_before + 10


class TestEnsureDataFresh:
    """ensure_data_fresh のテスト."""

    def test_fresh_data_no_update(self, sample_parquet: Path) -> None:
        from scripts.v460.ml.update_training_data import ensure_data_fresh

        with patch(
            "scripts.v460.ml.update_training_data._hours_since_last_update",
            return_value=1.0,
        ):
            result = ensure_data_fresh(sample_parquet, max_stale_hours=48.0)
            assert result is False

    def test_stale_data_triggers_update(self, sample_parquet: Path) -> None:
        from scripts.v460.ml.update_training_data import ensure_data_fresh

        with patch(
            "scripts.v460.ml.update_training_data._hours_since_last_update",
            return_value=100.0,
        ), patch(
            "scripts.v460.ml.update_training_data.update_training_parquet",
            return_value=50,
        ) as mock_update:
            result = ensure_data_fresh(sample_parquet, max_stale_hours=48.0)
            assert result is True
            mock_update.assert_called_once_with(sample_parquet)

    def test_update_failure_returns_false(self, sample_parquet: Path) -> None:
        from scripts.v460.ml.update_training_data import ensure_data_fresh

        with patch(
            "scripts.v460.ml.update_training_data._hours_since_last_update",
            return_value=100.0,
        ), patch(
            "scripts.v460.ml.update_training_data.update_training_parquet",
            side_effect=RuntimeError("yfinance error"),
        ):
            result = ensure_data_fresh(sample_parquet, max_stale_hours=48.0)
            assert result is False


class TestGetAllParquetFeatures:
    """_get_all_parquet_features のテスト."""

    def test_nonexistent_returns_sac_features(self, tmp_path: Path) -> None:
        from scripts.v460.ml.update_training_data import (
            _SAC_FEATURES,
            _get_all_parquet_features,
        )

        result = _get_all_parquet_features(tmp_path / "no.parquet")
        assert set(result) == set(_SAC_FEATURES)

    def test_includes_sac_features(self, sample_parquet: Path) -> None:
        from scripts.v460.ml.update_training_data import (
            _SAC_FEATURES,
            _get_all_parquet_features,
        )

        result = _get_all_parquet_features(sample_parquet)
        for feat in _SAC_FEATURES:
            assert feat in result


class TestDownloadOhlcv:
    """_download_ohlcv のテスト (mocked)."""

    def test_returns_dataframe(self) -> None:
        from scripts.v460.ml.update_training_data import _download_ohlcv

        mock_hist = pd.DataFrame(
            {
                "Open": [14_000_000.0] * 5,
                "High": [14_100_000.0] * 5,
                "Low": [13_900_000.0] * 5,
                "Close": [14_050_000.0] * 5,
                "Volume": [0.5] * 5,
            },
            index=pd.date_range("2026-03-20", periods=5, freq="1min", tz="UTC"),
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_hist

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = _download_ohlcv("7d")
            assert len(result) == 5
            assert list(result.columns) == [
                "timestamp", "open", "high", "low", "close", "volume",
            ]

    def test_raises_on_empty(self) -> None:
        from scripts.v460.ml.update_training_data import _download_ohlcv

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            with pytest.raises(RuntimeError, match="empty"):
                _download_ohlcv("7d")
