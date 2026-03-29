"""552# update_training_data ユニットテスト."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from datetime import datetime, timezone
import shutil
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scripts.v460.ml.update_training_data import (
    _SAC_FEATURES,
    _download_ohlcv,
    _get_all_parquet_features,
    _get_parquet_last_timestamp,
    _hours_since_last_update,
    _merge_into_parquet,
    ensure_data_fresh,
)


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
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


def _build_sample_parquet(path: Path, sample_ohlcv: pd.DataFrame) -> None:
    # tz を除去して保存 (本番と同じ形式)
    df = sample_ohlcv.copy()
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["price_velocity"] = np.float32(0.0)
    df["micro_trend"] = np.float32(0.0)
    df.to_parquet(path, index=False)


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def sample_parquet_template(
    sample_ohlcv: pd.DataFrame,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """読み取り専用の parquet template を module 単位で共有する."""
    path = cast(Path, tmp_path_factory.mktemp("update_training_data")) / "test.parquet"
    _build_sample_parquet(path, sample_ohlcv)
    # 初回 engine 初期化を fixture 側で吸収し、individual test の first-call cost を減らす.
    pd.read_parquet(path, columns=["timestamp"])
    return path


@pytest.fixture  # type: ignore[untyped-decorator]
def sample_parquet(sample_parquet_template: Path, tmp_path: Path) -> Path:
    """各テスト用の parquet ファイルを template から複製する."""
    path = tmp_path / "test.parquet"
    shutil.copy2(sample_parquet_template, path)
    return path


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def sample_parquet_features(sample_parquet_template: Path) -> tuple[str, ...]:
    """parquet の feature 名取得を module 単位で warm up する."""
    return tuple(_get_all_parquet_features(sample_parquet_template))


class TestGetParquetLastTimestamp:
    """_get_parquet_last_timestamp のテスト."""

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = _get_parquet_last_timestamp(tmp_path / "no.parquet")
        assert result is None

    def test_returns_last_ts(self, sample_parquet: Path) -> None:
        result = _get_parquet_last_timestamp(sample_parquet)
        assert result is not None
        assert isinstance(result, datetime)

    def test_tz_aware(self, sample_parquet: Path) -> None:
        result = _get_parquet_last_timestamp(sample_parquet)
        assert result is not None
        assert result.tzinfo is not None


class TestHoursSinceLastUpdate:
    """_hours_since_last_update のテスト."""

    def test_nonexistent_returns_inf(self, tmp_path: Path) -> None:
        result = _hours_since_last_update(tmp_path / "nope.parquet")
        assert result == float("inf")

    def test_returns_positive(self, sample_parquet: Path) -> None:
        result = _hours_since_last_update(sample_parquet)
        assert result > 0


class TestMergeIntoParquet:
    """_merge_into_parquet のテスト."""

    def test_creates_new_parquet(self, tmp_path: Path) -> None:
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
        existing = pd.read_parquet(sample_parquet)
        n_before = len(existing)

        # 重複データを追加
        dup = existing.tail(10).copy()
        n_added = _merge_into_parquet(sample_parquet, dup)
        assert n_added == 0

        after = pd.read_parquet(sample_parquet)
        assert len(after) == n_before

    def test_appends_new_rows(self, sample_parquet: Path) -> None:
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
        with patch(
            "scripts.v460.ml.update_training_data._hours_since_last_update",
            return_value=1.0,
        ):
            result = ensure_data_fresh(sample_parquet, max_stale_hours=48.0)
            assert result is False

    def test_stale_data_triggers_update(self, sample_parquet: Path) -> None:
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
        result = _get_all_parquet_features(tmp_path / "no.parquet")
        assert set(result) == set(_SAC_FEATURES)

    def test_includes_sac_features(
        self,
        sample_parquet_features: tuple[str, ...],
    ) -> None:
        result = sample_parquet_features
        for feat in _SAC_FEATURES:
            assert feat in result

    def test_cache_invalidates_when_parquet_changes(
        self,
        sample_parquet: Path,
    ) -> None:
        initial = set(_get_all_parquet_features(sample_parquet))

        updated = pd.read_parquet(sample_parquet)
        updated["price_velocity"] = updated["price_velocity"].astype(np.float32)
        updated["ema_velocity_bps"] = np.float32(0.0)
        updated.to_parquet(sample_parquet, index=False)

        refreshed = set(_get_all_parquet_features(sample_parquet))
        assert initial <= refreshed
        assert "ema_velocity_bps" in refreshed


class TestDownloadOhlcv:
    """_download_ohlcv のテスト (mocked)."""

    @pytest.fixture(scope="class")  # type: ignore[untyped-decorator]
    def mock_hist(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": [14_000_000.0] * 5,
                "High": [14_100_000.0] * 5,
                "Low": [13_900_000.0] * 5,
                "Close": [14_050_000.0] * 5,
                "Volume": [0.5] * 5,
            },
            index=pd.date_range("2026-03-20", periods=5, freq="1min", tz="UTC"),
        )

    def test_returns_dataframe(self, mock_hist: pd.DataFrame) -> None:
        mock_hist = pd.DataFrame(
            mock_hist,
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_hist

        fake_yfinance = SimpleNamespace(Ticker=MagicMock(return_value=mock_ticker))
        with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
            result = _download_ohlcv("7d")
            assert len(result) == 5
            assert list(result.columns) == [
                "timestamp", "open", "high", "low", "close", "volume",
            ]

    def test_raises_on_empty(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        fake_yfinance = SimpleNamespace(Ticker=MagicMock(return_value=mock_ticker))
        with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
            with pytest.raises(RuntimeError, match="empty"):
                _download_ohlcv("7d")
