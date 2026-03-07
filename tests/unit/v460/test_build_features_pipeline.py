"""
v460 build_features パイプライン統合テスト.

proxy / real モードの特徴量生成パイプラインを検証する。
MarketDataCollector.aggregate_to_1min → add_microstructure_features → V460_FEATURES 列の連結確認。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scripts.v460.build_features import V460_FEATURES, build_proxy_features
from ztb.data.market_data_collector import MarketDataCollector
from ztb.features.microstructure import (
    MICROSTRUCTURE_FEATURES,
    add_microstructure_features,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# =====================================================================
# Helpers
# =====================================================================


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """テスト用 OHLCV DataFrame."""
    rng = np.random.RandomState(42)
    close = 15_000_000.0 + np.cumsum(rng.randn(n) * 1000)
    high = close + rng.uniform(100, 2000, n)
    low = close - rng.uniform(100, 2000, n)
    open_ = close + rng.randn(n) * 500
    volume = rng.uniform(0.1, 10.0, n)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture(scope="module")
def proxy_features_default() -> pd.DataFrame:
    """代表的な proxy feature 出力を再利用する."""
    return build_proxy_features(_make_ohlcv(200))


@pytest.fixture(scope="module")
def proxy_features_zero_volume() -> pd.DataFrame:
    """volume=0 系の出力を再利用する."""
    df = _make_ohlcv(100)
    df["volume"] = 0.0
    return build_proxy_features(df)


@pytest.fixture(scope="module")
def proxy_window_variants() -> tuple[pd.DataFrame, pd.DataFrame]:
    """window 差分比較用の2出力を返す."""
    df = _make_ohlcv(200)
    return build_proxy_features(df, window=10), build_proxy_features(df, window=30)


# =====================================================================
# Proxy Mode テスト
# =====================================================================


class TestBuildProxyFeatures:
    """build_proxy_features のテスト."""

    def test_output_columns(self, proxy_features_default: pd.DataFrame) -> None:
        """出力に close + V460_FEATURES 10 種がすべて含まれる."""
        assert "close" in proxy_features_default.columns
        for feat in V460_FEATURES:
            assert feat in proxy_features_default.columns, f"Missing feature: {feat}"

    def test_output_shape(self) -> None:
        """行数が入力と一致."""
        df = _make_ohlcv(150)
        result = build_proxy_features(df)
        assert len(result) == 150

    def test_no_inf_values(self, proxy_features_default: pd.DataFrame) -> None:
        """無限大が含まれない."""
        numeric = proxy_features_default.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Inf detected in output"

    def test_nan_limited_to_warmup(self, proxy_features_default: pd.DataFrame) -> None:
        """NaN は warmup 期間のみ (末尾は NaN なし)."""
        # 末尾 50 行は NaN がないはず (window=20 なので)
        tail = proxy_features_default.tail(50)
        nan_count = tail.isna().sum().sum()
        assert nan_count == 0, f"NaN in tail 50 rows: {nan_count}"

    def test_bid_ask_spread_positive(self, proxy_features_default: pd.DataFrame) -> None:
        """bid_ask_spread は常に ≥ 0."""
        # high >= low なので spread >= 0
        assert (proxy_features_default["bid_ask_spread"].dropna() >= 0).all()

    def test_depth_imbalance_range(self, proxy_features_default: pd.DataFrame) -> None:
        """depth_imbalance (CLV) は [-1, 1] 範囲."""
        di = proxy_features_default["depth_imbalance"].dropna()
        assert (di >= -1.01).all() and (di <= 1.01).all()

    def test_different_window(
        self,
        proxy_window_variants: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """window パラメータが変更可能."""
        r10, r30 = proxy_window_variants
        # 異なる window → 異なる rolling 計算
        assert not np.allclose(
            r10["trade_flow_imbalance"].dropna().values[-50:],
            r30["trade_flow_imbalance"].dropna().values[-50:],
        )

    def test_small_input(self) -> None:
        """最小入力 (n=5) でもクラッシュしない."""
        df = _make_ohlcv(5)
        result = build_proxy_features(df, window=3)
        assert len(result) == 5

    def test_zero_volume_handling(self, proxy_features_zero_volume: pd.DataFrame) -> None:
        """volume=0 でも除算エラーにならない (eps ガード)."""
        assert not proxy_features_zero_volume.isna().all().any(), "全 NaN 列がある"


# =====================================================================
# Real Mode パイプライン連結テスト
# =====================================================================


def _ts(minute: int, second: int = 0) -> float:
    """UTC 2026-01-01 00:{minute}:{second} の UNIX timestamp."""
    return 1767225600.0 + minute * 60 + second


def _make_raw_records(n_minutes: int = 30) -> tuple[list[dict], list[dict]]:
    """raw orderbook + trades レコードを生成."""
    rng = np.random.RandomState(42)

    ob_records = []
    tr_records = []

    for m in range(n_minutes):
        # Orderbook: 2 snapshots per minute
        for s_offset in [0, 30]:
            best_bid = 10_000_000 + rng.randint(-500, 500)
            spread = rng.randint(500, 2000)
            best_ask = best_bid + spread
            bids = [[best_bid - i * 100, 0.1 * (5 - i)] for i in range(5)]
            asks = [[best_ask + i * 100, 0.1 * (5 - i)] for i in range(5)]
            ob_records.append({
                "ts": _ts(m, s_offset),
                "bids": bids,
                "asks": asks,
                "exchange": "coincheck",
            })

        # Trades: 3-8 per minute
        n_trades = rng.randint(3, 9)
        for t in range(n_trades):
            tr_records.append({
                "ts": _ts(m, rng.randint(0, 60)),
                "price": 10_000_000 + rng.randn() * 500,
                "amount": rng.uniform(0.001, 0.1),
                "side": "buy" if rng.random() < 0.5 else "sell",
            })

    return ob_records, tr_records


def _aggregate_raw_records(
    tmp_path: Path,
    *,
    ob_records: list[dict],
    tr_records: list[dict],
) -> pd.DataFrame:
    """raw 読込と parquet 書込を patch して集約ロジックだけを見る."""
    ob_path = tmp_path / "ob.jsonl.gz"
    tr_path = tmp_path / "tr.jsonl.gz"
    out_path = tmp_path / "out.parquet"

    def _fake_read(path: Path) -> list[dict]:
        if path == ob_path:
            return ob_records
        if path == tr_path:
            return tr_records
        return []

    with patch("ztb.data.market_data_collector._read_jsonl_gz", side_effect=_fake_read):
        with patch.object(pd.DataFrame, "to_parquet", autospec=True, return_value=None):
            return MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)


@pytest.fixture(scope="class")
def real_mode_aggregate_30(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """30分相当の aggregate 出力を再利用する."""
    tmp_path = tmp_path_factory.mktemp("build_features_agg_30")
    ob_records, tr_records = _make_raw_records(30)
    return _aggregate_raw_records(tmp_path, ob_records=ob_records, tr_records=tr_records)


@pytest.fixture(scope="class")
def real_mode_micro_40(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """microstructure 追加済みの real-mode 出力を再利用する."""
    tmp_path = tmp_path_factory.mktemp("build_features_micro_40")
    ob_records, tr_records = _make_raw_records(40)
    agg = _aggregate_raw_records(tmp_path, ob_records=ob_records, tr_records=tr_records)
    if "close" not in agg.columns and "mid_price" in agg.columns:
        agg["close"] = agg["mid_price"]
    return add_microstructure_features(agg)


class TestRealModePipeline:
    """MarketDataCollector.aggregate_to_1min → add_microstructure_features の連結確認."""

    def test_aggregate_output_schema(self, real_mode_aggregate_30: pd.DataFrame) -> None:
        """aggregate_to_1min の出力カラムを確認."""
        assert isinstance(real_mode_aggregate_30, pd.DataFrame)
        assert len(real_mode_aggregate_30) > 0
        # aggregate 出力に mid_price 等が含まれる
        for col in ["best_bid", "best_ask", "mid_price"]:
            assert col in real_mode_aggregate_30.columns, f"Missing column: {col}"

    def test_microstructure_on_aggregated(self, real_mode_micro_40: pd.DataFrame) -> None:
        """aggregate_to_1min 出力 → add_microstructure_features で特徴量追加."""
        # microstructure features が追加される
        for feat in MICROSTRUCTURE_FEATURES:
            assert feat in real_mode_micro_40.columns, f"Missing microstructure feature: {feat}"

    def test_v460_features_coverage(self, real_mode_micro_40: pd.DataFrame) -> None:
        """V460_FEATURES 10 種がすべて生成される (real mode 相当)."""
        missing = [f for f in V460_FEATURES if f not in real_mode_micro_40.columns]
        assert not missing, f"Missing V460 features: {missing}"

    def test_pipeline_no_inf(self, real_mode_micro_40: pd.DataFrame) -> None:
        """パイプライン出力に Inf がない."""
        numeric = real_mode_micro_40.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_pipeline_row_count(self, real_mode_aggregate_30: pd.DataFrame) -> None:
        """aggregate 後の行数が分単位で概ね正しい."""
        # 30 分 → 最大 30 行 (端の分は欠落可能)
        assert 20 <= len(real_mode_aggregate_30) <= 31
