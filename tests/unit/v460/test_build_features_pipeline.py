"""
v460 build_features パイプライン統合テスト.

proxy / real モードの特徴量生成パイプラインを検証する。
MarketDataCollector.aggregate_to_1min → add_microstructure_features → V460_FEATURES 列の連結確認。
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


# =====================================================================
# Proxy Mode テスト
# =====================================================================


class TestBuildProxyFeatures:
    """build_proxy_features のテスト."""

    def test_output_columns(self) -> None:
        """出力に close + V460_FEATURES 10 種がすべて含まれる."""
        from scripts.v460.build_features import V460_FEATURES, build_proxy_features

        df = _make_ohlcv(200)
        result = build_proxy_features(df)

        assert "close" in result.columns
        for feat in V460_FEATURES:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_output_shape(self) -> None:
        """行数が入力と一致."""
        from scripts.v460.build_features import build_proxy_features

        df = _make_ohlcv(150)
        result = build_proxy_features(df)
        assert len(result) == 150

    def test_no_inf_values(self) -> None:
        """無限大が含まれない."""
        from scripts.v460.build_features import build_proxy_features

        df = _make_ohlcv(200)
        result = build_proxy_features(df)
        numeric = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Inf detected in output"

    def test_nan_limited_to_warmup(self) -> None:
        """NaN は warmup 期間のみ (末尾は NaN なし)."""
        from scripts.v460.build_features import build_proxy_features

        df = _make_ohlcv(200)
        result = build_proxy_features(df)
        # 末尾 50 行は NaN がないはず (window=20 なので)
        tail = result.tail(50)
        nan_count = tail.isna().sum().sum()
        assert nan_count == 0, f"NaN in tail 50 rows: {nan_count}"

    def test_bid_ask_spread_positive(self) -> None:
        """bid_ask_spread は常に ≥ 0."""
        from scripts.v460.build_features import build_proxy_features

        df = _make_ohlcv(200)
        result = build_proxy_features(df)
        # high >= low なので spread >= 0
        assert (result["bid_ask_spread"].dropna() >= 0).all()

    def test_depth_imbalance_range(self) -> None:
        """depth_imbalance (CLV) は [-1, 1] 範囲."""
        from scripts.v460.build_features import build_proxy_features

        df = _make_ohlcv(200)
        result = build_proxy_features(df)
        di = result["depth_imbalance"].dropna()
        assert (di >= -1.01).all() and (di <= 1.01).all()

    def test_different_window(self) -> None:
        """window パラメータが変更可能."""
        from scripts.v460.build_features import build_proxy_features

        df = _make_ohlcv(200)
        r10 = build_proxy_features(df, window=10)
        r30 = build_proxy_features(df, window=30)
        # 異なる window → 異なる rolling 計算
        assert not np.allclose(
            r10["trade_flow_imbalance"].dropna().values[-50:],
            r30["trade_flow_imbalance"].dropna().values[-50:],
        )

    def test_small_input(self) -> None:
        """最小入力 (n=5) でもクラッシュしない."""
        from scripts.v460.build_features import build_proxy_features

        df = _make_ohlcv(5)
        result = build_proxy_features(df, window=3)
        assert len(result) == 5

    def test_zero_volume_handling(self) -> None:
        """volume=0 でも除算エラーにならない (eps ガード)."""
        from scripts.v460.build_features import build_proxy_features

        df = _make_ohlcv(100)
        df["volume"] = 0.0
        result = build_proxy_features(df)
        assert not result.isna().all().any(), "全 NaN 列がある"


# =====================================================================
# Real Mode パイプライン連結テスト
# =====================================================================


def _ts(minute: int, second: int = 0) -> float:
    """UTC 2026-01-01 00:{minute}:{second} の UNIX timestamp."""
    return 1767225600.0 + minute * 60 + second


def _write_gz(path: Path, records: list[dict]) -> None:
    """JSONL gzip を書き出す."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_raw_data(tmp_path: Path, n_minutes: int = 30) -> tuple[Path, Path, Path]:
    """raw orderbook + trades JSONL.gz を生成し、ファイルパスを返す."""
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

    ob_path = tmp_path / "ob.jsonl.gz"
    tr_path = tmp_path / "tr.jsonl.gz"
    out_path = tmp_path / "out.parquet"
    _write_gz(ob_path, ob_records)
    _write_gz(tr_path, tr_records)
    return ob_path, tr_path, out_path


class TestRealModePipeline:
    """MarketDataCollector.aggregate_to_1min → add_microstructure_features の連結確認."""

    def test_aggregate_output_schema(self, tmp_path: Path) -> None:
        """aggregate_to_1min の出力カラムを確認."""
        from ztb.data.market_data_collector import MarketDataCollector

        ob_path, tr_path, out_path = _make_raw_data(tmp_path, 30)
        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # aggregate 出力に mid_price 等が含まれる
        for col in ["best_bid", "best_ask", "mid_price"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_microstructure_on_aggregated(self, tmp_path: Path) -> None:
        """aggregate_to_1min 出力 → add_microstructure_features で特徴量追加."""
        from ztb.data.market_data_collector import MarketDataCollector
        from ztb.features.microstructure import (
            MICROSTRUCTURE_FEATURES,
            add_microstructure_features,
        )

        ob_path, tr_path, out_path = _make_raw_data(tmp_path, 60)
        agg = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        # close 列がない場合は mid_price を代用 (build_features.py と同じロジック)
        if "close" not in agg.columns and "mid_price" in agg.columns:
            agg["close"] = agg["mid_price"]

        result = add_microstructure_features(agg)

        # microstructure features が追加される
        for feat in MICROSTRUCTURE_FEATURES:
            assert feat in result.columns, f"Missing microstructure feature: {feat}"

    def test_v460_features_coverage(self, tmp_path: Path) -> None:
        """V460_FEATURES 10 種がすべて生成される (real mode 相当)."""
        from scripts.v460.build_features import V460_FEATURES
        from ztb.data.market_data_collector import MarketDataCollector
        from ztb.features.microstructure import add_microstructure_features

        ob_path, tr_path, out_path = _make_raw_data(tmp_path, 60)
        agg = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        if "close" not in agg.columns and "mid_price" in agg.columns:
            agg["close"] = agg["mid_price"]

        result = add_microstructure_features(agg)

        missing = [f for f in V460_FEATURES if f not in result.columns]
        assert not missing, f"Missing V460 features: {missing}"

    def test_pipeline_no_inf(self, tmp_path: Path) -> None:
        """パイプライン出力に Inf がない."""
        from ztb.data.market_data_collector import MarketDataCollector
        from ztb.features.microstructure import add_microstructure_features

        ob_path, tr_path, out_path = _make_raw_data(tmp_path, 60)
        agg = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)
        if "close" not in agg.columns and "mid_price" in agg.columns:
            agg["close"] = agg["mid_price"]
        result = add_microstructure_features(agg)
        numeric = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_pipeline_row_count(self, tmp_path: Path) -> None:
        """aggregate 後の行数が分単位で概ね正しい."""
        from ztb.data.market_data_collector import MarketDataCollector

        ob_path, tr_path, out_path = _make_raw_data(tmp_path, 30)
        result = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)
        # 30 分 → 最大 30 行 (端の分は欠落可能)
        assert 20 <= len(result) <= 31
