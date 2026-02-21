"""135# テスト: TradesRecorder + trades_health + per-run Gate + OBRecorder refactor.

P0-03/04: TradesRecorder JSONL.gz 蓄積 + 重複排除
P2-09→P1: trades 健全性チェック
P0-07: per-run Gate 評価
P0-12: run_gate_check.py CLI 統一
OBRecorder: ztb/io/jsonl_gz.py 共通化リファクタ
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from ztb.io.jsonl_gz import append_jsonl_gz, read_jsonl_gz


# =====================================================================
# §1 ztb/io/jsonl_gz.py — 共通 JSONL.gz ユーティリティ
# =====================================================================


class TestAppendJsonlGz:
    """append_jsonl_gz の基本動作."""

    def test_append_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl.gz"
        n = append_jsonl_gz(path, [{"a": 1}, {"b": 2}])
        assert n == 2
        assert path.exists()

    def test_append_empty_returns_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl.gz"
        assert append_jsonl_gz(path, []) == 0
        assert not path.exists()

    def test_append_multiple_calls(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl.gz"
        append_jsonl_gz(path, [{"x": 1}])
        append_jsonl_gz(path, [{"x": 2}, {"x": 3}])
        records = read_jsonl_gz(path)
        assert len(records) == 3
        assert records[2]["x"] == 3

    def test_read_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.jsonl.gz"
        assert read_jsonl_gz(path) == []


# =====================================================================
# §2 ztb/data/trades_recorder.py — TradesRecorder
# =====================================================================


class TestTradesRecorderBasic:
    """TradesRecorder の基本動作."""

    def test_record_adds_to_buffer(self) -> None:
        from ztb.data.trades_recorder import TradesRecorder
        rec = TradesRecorder(enabled=True)
        n = rec.record_trades([
            {"ts": 1000.0, "price": 14500000.0, "amount": 0.01, "side": "buy"},
        ])
        assert n == 1
        assert rec.buffer_size == 1

    def test_record_disabled_noop(self) -> None:
        from ztb.data.trades_recorder import TradesRecorder
        rec = TradesRecorder(enabled=False)
        n = rec.record_trades([
            {"ts": 1000.0, "price": 14500000.0, "amount": 0.01, "side": "buy"},
        ])
        assert n == 0
        assert rec.buffer_size == 0

    def test_flush_empty_returns_zero(self) -> None:
        from ztb.data.trades_recorder import TradesRecorder
        rec = TradesRecorder(enabled=True)
        assert rec.flush() == 0

    def test_total_written_tracks_cumulative(self, tmp_path: Path) -> None:
        from ztb.data.trades_recorder import TradesRecorder
        rec = TradesRecorder(raw_dir=tmp_path, enabled=True)
        rec.record_trades([
            {"ts": 1000.0, "price": 14500000.0, "amount": 0.01, "side": "buy"},
            {"ts": 1001.0, "price": 14500100.0, "amount": 0.02, "side": "sell"},
        ])
        rec.flush()
        assert rec.total_written == 2
        assert rec.buffer_size == 0


class TestTradesRecorderDedup:
    """TradesRecorder の重複排除."""

    def test_dedup_same_trade(self) -> None:
        from ztb.data.trades_recorder import TradesRecorder
        rec = TradesRecorder(enabled=True)
        trade = {"ts": 1000.0, "price": 14500000.0, "amount": 0.01, "side": "buy"}
        rec.record_trades([trade, trade, trade])
        assert rec.buffer_size == 1

    def test_dedup_old_trades_skipped(self, tmp_path: Path) -> None:
        from ztb.data.trades_recorder import TradesRecorder
        rec = TradesRecorder(raw_dir=tmp_path, enabled=True)
        # 最初のバッチ
        rec.record_trades([
            {"ts": 1000.0, "price": 14500000.0, "amount": 0.01, "side": "buy"},
            {"ts": 1001.0, "price": 14500100.0, "amount": 0.02, "side": "sell"},
        ])
        rec.flush()
        # 2回目: ts <= 1001.0 の trade は重複スキップ
        n = rec.record_trades([
            {"ts": 1000.0, "price": 14500000.0, "amount": 0.01, "side": "buy"},
            {"ts": 1001.0, "price": 14500100.0, "amount": 0.02, "side": "sell"},
            {"ts": 1002.0, "price": 14500200.0, "amount": 0.03, "side": "buy"},
        ])
        assert n == 1  # 新規は ts=1002 の1件のみ


class TestTradesRecorderFlush:
    """TradesRecorder の flush → JSONL.gz 書き出し."""

    def test_flush_creates_jsonl_gz(self, tmp_path: Path) -> None:
        from ztb.data.trades_recorder import TradesRecorder
        rec = TradesRecorder(raw_dir=tmp_path, enabled=True)
        rec.record_trades([
            {"ts": 1000.5, "price": 14500000.0, "amount": 0.01, "side": "buy"},
        ])
        n = rec.flush()
        assert n == 1
        tr_dir = tmp_path / "trades"
        assert tr_dir.exists()
        files = list(tr_dir.glob("*.jsonl.gz"))
        assert len(files) == 1
        with gzip.open(files[0], "rt", encoding="utf-8") as f:
            data = json.loads(f.readline())
        assert data["ts"] == 1000.5
        assert data["price"] == 14500000.0
        assert data["side"] == "buy"


class TestTradesRecorderFromAdapter:
    """record_from_adapter: TradeRecord オブジェクトからの記録."""

    def test_record_from_adapter_duck_typing(self) -> None:
        from ztb.data.trades_recorder import TradesRecorder

        @dataclass
        class FakeTradeRecord:
            timestamp: float
            price: float
            amount: float
            side: str

        rec = TradesRecorder(enabled=True)
        trades = [
            FakeTradeRecord(timestamp=1000.0, price=14500000.0, amount=0.01, side="buy"),
            FakeTradeRecord(timestamp=1001.0, price=14500100.0, amount=0.02, side="sell"),
        ]
        n = rec.record_from_adapter(trades)
        assert n == 2
        assert rec.buffer_size == 2


# =====================================================================
# §3 ztb/data/trades_health.py — trades 健全性チェック
# =====================================================================


class TestTradesHealth:
    """trades_health.check_trades_health の検証."""

    def test_healthy_when_all_present(self, tmp_path: Path) -> None:
        from ztb.data.trades_health import check_trades_health
        tr_dir = tmp_path / "trades"
        tr_dir.mkdir(parents=True)
        # 直近3日分のファイルを作成
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        for i in range(3):
            day = (now - timedelta(days=i)).strftime("%Y%m%d")
            path = tr_dir / f"{day}.jsonl.gz"
            append_jsonl_gz(path, [{"ts": time.time(), "price": 1.0, "amount": 0.01, "side": "buy"}])
        result = check_trades_health(raw_dir=tmp_path, lookback_days=3)
        assert result.healthy is True
        assert len(result.missing_days) == 0

    def test_unhealthy_when_missing(self, tmp_path: Path) -> None:
        from ztb.data.trades_health import check_trades_health
        tr_dir = tmp_path / "trades"
        tr_dir.mkdir(parents=True)
        # 1日分のみ (3日必要)
        from datetime import datetime, timezone
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = tr_dir / f"{day}.jsonl.gz"
        append_jsonl_gz(path, [{"ts": time.time(), "price": 1.0, "amount": 0.01, "side": "buy"}])
        result = check_trades_health(raw_dir=tmp_path, lookback_days=3)
        assert result.healthy is False
        assert len(result.missing_days) > 0

    def test_stale_detection(self, tmp_path: Path) -> None:
        import os

        from ztb.data.trades_health import check_trades_health
        tr_dir = tmp_path / "trades"
        tr_dir.mkdir(parents=True)
        # 古いファイル (stale): mtime を48時間前に設定
        path = tr_dir / "20260101.jsonl.gz"
        append_jsonl_gz(path, [{"ts": 1.0, "price": 1.0, "amount": 0.01, "side": "buy"}])
        old_mtime = time.time() - 48 * 3600
        os.utime(path, (old_mtime, old_mtime))
        result = check_trades_health(
            raw_dir=tmp_path,
            expected_days=["20260101"],
            stale_threshold_hours=24.0,
        )
        # stale で unhealthy
        assert result.healthy is False
        assert result.stale_hours > 24.0

    def test_no_trades_dir(self, tmp_path: Path) -> None:
        from ztb.data.trades_health import check_trades_health
        result = check_trades_health(raw_dir=tmp_path, lookback_days=1)
        assert result.healthy is False
        assert len(result.available_days) == 0


# =====================================================================
# §4 gate_judgment.py — per-run Gate 評価 (P0-07)
# =====================================================================


class TestPerRunGateFiltering:
    """_filter_by_run_id / _get_unique_run_ids の検証."""

    def _make_record(
        self, run_id: str, ts: float = 1000.0, side: str = "buy"
    ) -> "FillRecord":
        from ztb.metrics.fill_quality import FillRecord
        return FillRecord(
            cycle_id=f"test_{ts}",
            timestamp=ts,
            side=side,
            order_price=14500000.0,
            order_quantity=0.001,
            filled=True,
            fill_price=14500000.0,
            queue_wait_sec=5.0,
            mid_at_fill=14500100.0,
            mid_30s_after=14500200.0,
            post_fill_30s_pnl=0.5,
            adverse_selected=False,
            git_sha="abc123",
            run_id=run_id,
        )

    def test_filter_by_specific_run_id(self) -> None:
        from scripts.v460.gate_judgment import _filter_by_run_id
        records = [
            self._make_record("run_A", ts=1000.0),
            self._make_record("run_B", ts=1001.0),
            self._make_record("run_A", ts=1002.0),
        ]
        filtered = _filter_by_run_id(records, run_id="run_A")
        assert len(filtered) == 2
        assert all(r.run_id == "run_A" for r in filtered)

    def test_filter_latest_run(self) -> None:
        from scripts.v460.gate_judgment import _filter_by_run_id
        records = [
            self._make_record("run_A", ts=1000.0),
            self._make_record("run_B", ts=2000.0),  # latest
            self._make_record("run_A", ts=1500.0),
        ]
        filtered = _filter_by_run_id(records, latest=True)
        assert len(filtered) == 1
        assert filtered[0].run_id == "run_B"

    def test_filter_no_args_returns_all(self) -> None:
        from scripts.v460.gate_judgment import _filter_by_run_id
        records = [
            self._make_record("run_A", ts=1000.0),
            self._make_record("run_B", ts=2000.0),
        ]
        filtered = _filter_by_run_id(records)
        assert len(filtered) == 2

    def test_get_unique_run_ids_sorted(self) -> None:
        from scripts.v460.gate_judgment import _get_unique_run_ids
        records = [
            self._make_record("run_B", ts=2000.0),
            self._make_record("run_A", ts=1000.0),
            self._make_record("run_B", ts=2001.0),
        ]
        ids = _get_unique_run_ids(records)
        assert ids == ["run_A", "run_B"]  # timestamp asc


# =====================================================================
# §5 OBRecorder — jsonl_gz 共通化後の回帰テスト
# =====================================================================


class TestOBRecorderRefactored:
    """OBRecorder が ztb/io/jsonl_gz 共通化後も正常動作すること."""

    def test_flush_creates_jsonl_gz(self, tmp_path: Path) -> None:
        from scripts.v460.lib.ob_recorder import OBRecorder
        rec = OBRecorder(raw_dir=tmp_path, enabled=True)
        rec.record([[14500000, 0.1]], [[14501000, 0.1]], timestamp=1000.5)
        n = rec.flush()
        assert n == 1
        ob_dir = tmp_path / "orderbook"
        files = list(ob_dir.glob("*.jsonl.gz"))
        assert len(files) == 1
        with gzip.open(files[0], "rt", encoding="utf-8") as f:
            data = json.loads(f.readline())
        assert data["ts"] == 1000.5
        assert data["exchange"] == "coincheck"

    def test_append_to_existing(self, tmp_path: Path) -> None:
        from scripts.v460.lib.ob_recorder import OBRecorder
        rec = OBRecorder(raw_dir=tmp_path, enabled=True)
        rec.record([[100, 1.0]], [[101, 1.0]], timestamp=1000.0)
        rec.flush()
        rec.record([[100, 2.0]], [[101, 2.0]], timestamp=1001.0)
        rec.flush()
        ob_dir = tmp_path / "orderbook"
        records = read_jsonl_gz(list(ob_dir.glob("*.jsonl.gz"))[0])
        assert len(records) == 2
