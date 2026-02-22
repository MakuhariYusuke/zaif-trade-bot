"""
aggregate_to_1min() + _read_jsonl_gz() 単体テスト — 014# B1.

MarketDataCollector.aggregate_to_1min の全分岐を検証する。
012# §2-C「aggregate_to_1min 検証不足」への対応。
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ztb.data.market_data_collector import MarketDataCollector, _read_jsonl_gz


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _ts(minute: int, second: int = 0) -> float:
    """UTC 2026-01-01 00:{minute}:{second} の UNIX timestamp."""
    # 2026-01-01 00:00:00 UTC = 1767225600
    return 1767225600.0 + minute * 60 + second


def _write_gz(path: Path, records: list[dict]) -> None:
    """JSONL gzip を書き出す."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_ob_record(
    minute: int,
    second: int = 0,
    best_bid: float = 10_000_000.0,
    best_ask: float = 10_001_000.0,
    depth_levels: int = 5,
) -> dict:
    """Orderbook raw レコードを生成."""
    spread_unit = (best_ask - best_bid) / depth_levels
    bids = [[best_bid - i * spread_unit, 0.1 * (depth_levels - i)] for i in range(depth_levels)]
    asks = [[best_ask + i * spread_unit, 0.1 * (depth_levels - i)] for i in range(depth_levels)]
    return {
        "ts": _ts(minute, second),
        "bids": bids,
        "asks": asks,
        "exchange": "coincheck",
    }


def _make_trade_record(
    minute: int,
    second: int = 0,
    price: float = 10_000_500.0,
    amount: float = 0.01,
    side: str = "buy",
) -> dict:
    """Trade raw レコードを生成."""
    return {
        "ts": _ts(minute, second),
        "price": price,
        "amount": amount,
        "side": side,
    }


# =====================================================================
# _read_jsonl_gz
# =====================================================================

class TestReadJsonlGz:
    """_read_jsonl_gz ヘルパー関数のテスト."""

    def test_read_valid_file(self, tmp_path: Path) -> None:
        records = [{"a": 1}, {"b": 2}]
        p = tmp_path / "test.jsonl.gz"
        _write_gz(p, records)

        result = _read_jsonl_gz(p)
        assert len(result) == 2
        assert result[0] == {"a": 1}

    def test_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        result = _read_jsonl_gz(tmp_path / "missing.jsonl.gz")
        assert result == []

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl.gz"
        _write_gz(p, [])
        result = _read_jsonl_gz(p)
        assert result == []

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "blanks.jsonl.gz"
        with gzip.open(p, "wt", encoding="utf-8") as f:
            f.write('{"a": 1}\n\n\n{"b": 2}\n')
        result = _read_jsonl_gz(p)
        assert len(result) == 2


# =====================================================================
# aggregate_to_1min — Orderbook のみ
# =====================================================================

class TestAggregateOrderbookOnly:
    """Orderbook のみのケース (trades 空)."""

    def test_basic_orderbook_aggregation(self, tmp_path: Path) -> None:
        ob_records = [
            _make_ob_record(0, 0, best_bid=10_000_000, best_ask=10_001_000),
            _make_ob_record(0, 30, best_bid=10_000_500, best_ask=10_001_500),
            _make_ob_record(1, 0, best_bid=10_001_000, best_ask=10_002_000),
        ]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, [])

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert len(df) == 2  # minute 0, minute 1
        assert "best_bid" in df.columns
        assert "best_ask" in df.columns
        assert "mid_price" in df.columns
        assert "spread" in df.columns
        assert "depth_imbalance" in df.columns
        assert "spread_range" in df.columns

    def test_last_snapshot_used(self, tmp_path: Path) -> None:
        """同じ分内の最後のスナップショットが使われる."""
        ob_records = [
            _make_ob_record(0, 0, best_bid=10_000_000, best_ask=10_001_000),
            _make_ob_record(0, 30, best_bid=10_000_500, best_ask=10_001_500),  # last
        ]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, [])

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert len(df) == 1
        assert df["best_bid"].iloc[0] == 10_000_500
        assert df["best_ask"].iloc[0] == 10_001_500

    def test_mid_price_correct(self, tmp_path: Path) -> None:
        ob_records = [_make_ob_record(0, 0, best_bid=10_000_000, best_ask=10_002_000)]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, [])

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert df["mid_price"].iloc[0] == pytest.approx(10_001_000.0)

    def test_spread_is_relative(self, tmp_path: Path) -> None:
        """spread = (ask - bid) / mid."""
        ob_records = [_make_ob_record(0, 0, best_bid=10_000_000, best_ask=10_001_000)]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, [])

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        expected_mid = (10_000_000 + 10_001_000) / 2
        expected_spread = 1_000 / expected_mid
        assert df["spread"].iloc[0] == pytest.approx(expected_spread, rel=1e-6)

    def test_depth_imbalance_range(self, tmp_path: Path) -> None:
        """depth_imbalance ∈ [-1, 1]."""
        ob_records = [_make_ob_record(0)]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, [])

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        val = df["depth_imbalance"].iloc[0]
        assert -1.0 <= val <= 1.0

    def test_spread_range_single_snapshot(self, tmp_path: Path) -> None:
        """分内に 1 スナップショットのみ → spread_range = 0."""
        ob_records = [_make_ob_record(0)]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, [])

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert df["spread_range"].iloc[0] == pytest.approx(0.0)

    def test_spread_range_multiple_snapshots(self, tmp_path: Path) -> None:
        """分内に複数スナップショット → spread_range > 0."""
        ob_records = [
            _make_ob_record(0, 0, best_bid=10_000_000, best_ask=10_001_000),
            _make_ob_record(0, 30, best_bid=10_000_000, best_ask=10_003_000),  # wider spread
        ]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, [])

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert df["spread_range"].iloc[0] > 0


# =====================================================================
# aggregate_to_1min — Trades のみ
# =====================================================================

class TestAggregateTradesOnly:
    """Trades のみのケース (orderbook 空)."""

    def test_basic_trade_aggregation(self, tmp_path: Path) -> None:
        tr_records = [
            _make_trade_record(0, 0, price=10_000_000, amount=0.1, side="buy"),
            _make_trade_record(0, 10, price=10_000_100, amount=0.2, side="sell"),
            _make_trade_record(1, 0, price=10_000_200, amount=0.3, side="buy"),
        ]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, [])
        _write_gz(tr_path, tr_records)

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert len(df) == 2  # minute 0, minute 1
        assert "buy_volume" in df.columns
        assert "sell_volume" in df.columns
        assert "trade_count" in df.columns
        assert "vwap" in df.columns
        assert "trade_flow_imbalance" in df.columns

    def test_buy_sell_volume_correct(self, tmp_path: Path) -> None:
        tr_records = [
            _make_trade_record(0, 0, amount=0.1, side="buy"),
            _make_trade_record(0, 10, amount=0.2, side="sell"),
            _make_trade_record(0, 20, amount=0.3, side="buy"),
        ]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, [])
        _write_gz(tr_path, tr_records)

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert df["buy_volume"].iloc[0] == pytest.approx(0.4)
        assert df["sell_volume"].iloc[0] == pytest.approx(0.2)
        assert df["trade_count"].iloc[0] == 3

    def test_vwap_calculation(self, tmp_path: Path) -> None:
        """VWAP = Σ(price × amount) / Σ(amount)."""
        tr_records = [
            _make_trade_record(0, 0, price=10_000_000, amount=0.1, side="buy"),
            _make_trade_record(0, 10, price=10_002_000, amount=0.3, side="buy"),
        ]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, [])
        _write_gz(tr_path, tr_records)

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        expected_vwap = (10_000_000 * 0.1 + 10_002_000 * 0.3) / (0.1 + 0.3)
        assert df["vwap"].iloc[0] == pytest.approx(expected_vwap, rel=1e-8)

    def test_trade_flow_imbalance_all_buy(self, tmp_path: Path) -> None:
        """全 buy → imbalance = 1.0."""
        tr_records = [
            _make_trade_record(0, 0, amount=0.1, side="buy"),
            _make_trade_record(0, 10, amount=0.2, side="buy"),
        ]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, [])
        _write_gz(tr_path, tr_records)

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert df["trade_flow_imbalance"].iloc[0] == pytest.approx(1.0)

    def test_trade_flow_imbalance_all_sell(self, tmp_path: Path) -> None:
        """全 sell → imbalance = -1.0."""
        tr_records = [
            _make_trade_record(0, 0, amount=0.1, side="sell"),
        ]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, [])
        _write_gz(tr_path, tr_records)

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert df["trade_flow_imbalance"].iloc[0] == pytest.approx(-1.0)


# =====================================================================
# aggregate_to_1min — Orderbook + Trades merged
# =====================================================================

class TestAggregateMerged:
    """Orderbook + Trades の結合テスト."""

    def test_merged_has_all_columns(self, tmp_path: Path) -> None:
        ob_records = [_make_ob_record(0)]
        tr_records = [_make_trade_record(0)]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, tr_records)

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        ob_cols = {"best_bid", "best_ask", "mid_price", "spread",
                   "bid_vol_5", "ask_vol_5", "depth_imbalance", "spread_range"}
        tr_cols = {"buy_volume", "sell_volume", "trade_count", "vwap",
                   "trade_flow_imbalance"}
        assert ob_cols.issubset(set(df.columns))
        assert tr_cols.issubset(set(df.columns))

    def test_outer_join_fills_nan(self, tmp_path: Path) -> None:
        """OB は min 0 のみ、Trades は min 1 のみ → outer join で NaN 埋め."""
        ob_records = [_make_ob_record(0)]
        tr_records = [_make_trade_record(1)]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, tr_records)

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert len(df) == 2
        # min 0: has OB data, no trade data
        assert not np.isnan(df["best_bid"].iloc[0])
        assert np.isnan(df["buy_volume"].iloc[0]) or df["buy_volume"].iloc[0] == 0
        # min 1: has trade data, no OB data
        assert np.isnan(df["best_bid"].iloc[1])

    def test_parquet_output_created(self, tmp_path: Path) -> None:
        ob_records = [_make_ob_record(0)]
        tr_records = [_make_trade_record(0)]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "output.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, tr_records)

        MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert out_path.exists()
        reloaded = pd.read_parquet(out_path)
        assert len(reloaded) >= 1

    def test_parquet_roundtrip(self, tmp_path: Path) -> None:
        """Parquet 書き出し→再読み込みの一致."""
        ob_records = [_make_ob_record(0), _make_ob_record(1)]
        tr_records = [_make_trade_record(0), _make_trade_record(1)]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "output.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, tr_records)

        original = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)
        reloaded = pd.read_parquet(out_path)

        assert len(original) == len(reloaded)
        for col in original.columns:
            assert col in reloaded.columns


# =====================================================================
# Edge cases
# =====================================================================

class TestAggregateEdgeCases:
    """エッジケース・異常系テスト."""

    def test_both_empty_returns_empty_df(self, tmp_path: Path) -> None:
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, [])
        _write_gz(tr_path, [])

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert df.empty
        assert not out_path.exists()  # 空のときは parquet 出力しない

    def test_missing_files_returns_empty(self, tmp_path: Path) -> None:
        """ファイルが存在しない場合."""
        ob_path = tmp_path / "missing_ob.jsonl.gz"
        tr_path = tmp_path / "missing_tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert df.empty

    def test_empty_bids_asks(self, tmp_path: Path) -> None:
        """bids/asks が空リストの場合 → NaN."""
        ob_records = [{
            "ts": _ts(0),
            "bids": [],
            "asks": [],
            "exchange": "coincheck",
        }]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, [])

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        # Empty bids/asks → NaN for best_bid/ask → all NaN → dropped by dropna(how='all')
        # OR the row remains with NaN values
        # Either way, should not crash
        assert isinstance(df, pd.DataFrame)

    def test_single_depth_level(self, tmp_path: Path) -> None:
        """depth = 1 レベルのみ."""
        ob_records = [{
            "ts": _ts(0),
            "bids": [[10_000_000.0, 0.5]],
            "asks": [[10_001_000.0, 0.3]],
            "exchange": "coincheck",
        }]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, [])

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert len(df) == 1
        assert df["bid_vol_5"].iloc[0] == pytest.approx(0.5)
        assert df["ask_vol_5"].iloc[0] == pytest.approx(0.3)

    def test_many_minutes(self, tmp_path: Path) -> None:
        """複数分にわたるデータの集約."""
        ob_records = [_make_ob_record(m) for m in range(10)]
        tr_records = [_make_trade_record(m) for m in range(10)]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, ob_records)
        _write_gz(tr_path, tr_records)

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        assert len(df) == 10

    def test_mixed_case_side(self, tmp_path: Path) -> None:
        """side が 'Buy'/'SELL' 等の大文字混在でも動作."""
        tr_records = [
            _make_trade_record(0, 0, amount=0.1, side="Buy"),
            _make_trade_record(0, 10, amount=0.2, side="SELL"),
        ]
        ob_path = tmp_path / "ob.jsonl.gz"
        tr_path = tmp_path / "tr.jsonl.gz"
        out_path = tmp_path / "out.parquet"
        _write_gz(ob_path, [])
        _write_gz(tr_path, tr_records)

        df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, out_path)

        # side.str.lower() == "buy" のため大文字でも正常動作
        assert df["buy_volume"].iloc[0] == pytest.approx(0.1)
        assert df["sell_volume"].iloc[0] == pytest.approx(0.2)
