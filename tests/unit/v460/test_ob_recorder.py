"""129# OBRecorder + MakerPriceCalculator OB snapshot キャッシュのテスト.

- OBRecorder: record → buffer → flush → JSONL.gz 書き出し
- MakerPriceCalculator: compute_imbalance で _last_ob_snapshot を保持
"""

from __future__ import annotations

import gzip
import inspect
import json
import math
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator
from scripts.v460.lib.ob_recorder import OBRecorder
from scripts.v460.ml.feature_enricher import _find_nearest_ob, load_raw_orderbook
from ztb.trading.live.exchanges.base.broker_interfaces import OrderBookSnapshot


# ===== OBRecorder テスト =====


def _flush_and_capture(rec: OBRecorder) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    def _capture(_path: Path, rows: list[dict[str, object]]) -> None:
        captured.extend(json.loads(json.dumps(rows)))

    with patch("scripts.v460.lib.ob_recorder.append_jsonl_gz", side_effect=_capture):
        rec.flush()
    return captured


class TestOBRecorderBasic:
    """OBRecorder の基本動作."""

    def test_record_adds_to_buffer(self) -> None:
        rec = OBRecorder(enabled=True)
        rec.record([[100, 1.0]], [[101, 1.0]], timestamp=1000.0)
        assert rec.buffer_size == 1

    def test_record_disabled_noop(self) -> None:
        rec = OBRecorder(enabled=False)
        rec.record([[100, 1.0]], [[101, 1.0]], timestamp=1000.0)
        assert rec.buffer_size == 0

    def test_flush_empty_returns_zero(self) -> None:
        rec = OBRecorder(enabled=True)
        assert rec.flush() == 0

    def test_total_written_tracks_cumulative(self) -> None:
        rec = OBRecorder(raw_dir=Path("/tmp/test_ob_recorder_tw"), enabled=True)
        rec.record([[100, 1.0]], [[101, 1.0]], timestamp=1000.0)
        rec.record([[100, 2.0]], [[101, 2.0]], timestamp=1001.0)
        rec.flush()
        assert rec.total_written == 2
        assert rec.buffer_size == 0

    def test_snapshot_stats_reports_buffer_and_failures(self) -> None:
        rec = OBRecorder(enabled=True)
        rec.record([[100, 1.0]], [[101, 1.0]], timestamp=1000.0)
        assert rec.snapshot_stats() == {
            "buffer_size": 1,
            "total_written": 0,
            "flush_fail_count": 0,
        }

    def test_record_preserves_zero_timestamp(self) -> None:
        rec = OBRecorder(enabled=True)
        rec.record([[100, 1.0]], [[101, 1.0]], timestamp=0.0)
        data = _flush_and_capture(rec)[0]
        assert data["ts"] == 0.0

    def test_record_ignores_malformed_snapshot(self) -> None:
        rec = OBRecorder(enabled=True)
        rec.record([MagicMock()], [[101, 1.0]], timestamp=1000.0)
        assert rec.buffer_size == 0

    def test_record_sanitizes_magicmock_timestamp(self) -> None:
        rec = OBRecorder(enabled=True)
        rec.record([[100, 1.0]], [[101, 1.0]], timestamp=MagicMock())
        data = _flush_and_capture(rec)[0]
        assert isinstance(data["ts"], float)
        assert math.isfinite(data["ts"])


class TestOBRecorderFlush:
    """OBRecorder の flush → JSONL.gz 書き出し."""

    def test_flush_creates_jsonl_gz(self, tmp_path: Path) -> None:
        rec = OBRecorder(raw_dir=tmp_path, enabled=True)
        bids = [[14500000, 0.1], [14499000, 0.2]]
        asks = [[14501000, 0.1], [14502000, 0.2]]
        rec.record(bids, asks, timestamp=1000.5)
        n = rec.flush()
        assert n == 1
        # ファイルが存在することを確認
        ob_dir = tmp_path / "orderbook"
        assert ob_dir.exists()
        files = list(ob_dir.glob("*.jsonl.gz"))
        assert len(files) == 1
        # 中身を検証
        with gzip.open(files[0], "rt", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["ts"] == 1000.5
        assert data["bids"] == bids
        assert data["asks"] == asks
        assert data["exchange"] == "coincheck"

    def test_flush_appends_to_existing(self, tmp_path: Path) -> None:
        rec = OBRecorder(raw_dir=tmp_path, enabled=True)
        rec.record([[100, 1.0]], [[101, 1.0]], timestamp=1000.0)
        rec.flush()
        rec.record([[100, 2.0]], [[101, 2.0]], timestamp=1001.0)
        rec.flush()
        ob_dir = tmp_path / "orderbook"
        files = list(ob_dir.glob("*.jsonl.gz"))
        assert len(files) == 1
        with gzip.open(files[0], "rt", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_flush_tuple_bids_asks_normalized(self, tmp_path: Path) -> None:
        """tuple 形式の bids/asks が list に変換されること."""
        rec = OBRecorder(raw_dir=tmp_path, enabled=True)
        rec.record([(100.0, 1.0)], [(101.0, 1.0)], timestamp=1000.0)
        rec.flush()
        ob_dir = tmp_path / "orderbook"
        with gzip.open(list(ob_dir.glob("*.jsonl.gz"))[0], "rt") as f:
            data = json.loads(f.readline())
        assert data["bids"] == [[100.0, 1.0]]
        assert data["asks"] == [[101.0, 1.0]]

    def test_flush_splits_records_by_snapshot_utc_day(self, tmp_path: Path) -> None:
        rec = OBRecorder(raw_dir=tmp_path, enabled=True)
        rec.record([[100, 1.0]], [[101, 1.0]], timestamp=86399.0)
        rec.record([[200, 1.0]], [[201, 1.0]], timestamp=86401.0)

        flushed = rec.flush()

        assert flushed == 2
        files = sorted((tmp_path / "orderbook").glob("*.jsonl.gz"))
        assert [path.name for path in files] == [
            "19700101.jsonl.gz",
            "19700102.jsonl.gz",
        ]
        with gzip.open(files[0], "rt", encoding="utf-8") as handle:
            day1 = [json.loads(line) for line in handle if line.strip()]
        with gzip.open(files[1], "rt", encoding="utf-8") as handle:
            day2 = [json.loads(line) for line in handle if line.strip()]
        assert [row["ts"] for row in day1] == [86399.0]
        assert [row["ts"] for row in day2] == [86401.0]


class TestOBRecorderAutoFlush:
    """flush_interval 経過時の自動 flush."""

    def test_auto_flush_on_interval(self, tmp_path: Path) -> None:
        rec = OBRecorder(raw_dir=tmp_path, flush_interval=0, enabled=True)
        # flush_interval=0 なので record 時に即座に flush
        rec._last_flush = time.time() - 1  # 過去にする
        rec.record([[100, 1.0]], [[101, 1.0]], timestamp=1000.0)
        # バッファは flush されているはず
        assert rec.buffer_size == 0
        assert rec.total_written == 1

    def test_shutdown_clears_buffer_after_flush_failure(self) -> None:
        rec = OBRecorder(enabled=True)
        rec.record([[100, 1.0]], [[101, 1.0]], timestamp=1000.0)
        with patch("scripts.v460.lib.ob_recorder.append_jsonl_gz", side_effect=OSError("disk full")):
            rec.shutdown()
        assert rec.buffer_size == 0
        assert rec.snapshot_stats()["flush_fail_count"] == 0


class TestOBRecorderFormat:
    """MarketDataCollector と同一フォーマットであることの検証."""

    def test_format_compatible_with_feature_enricher(self, tmp_path: Path) -> None:
        """feature_enricher.load_raw_orderbook() で読めるフォーマット."""
        rec = OBRecorder(raw_dir=tmp_path, enabled=True)
        bids = [[14500000.0, 0.1], [14499000.0, 0.2], [14498000.0, 0.3],
                [14497000.0, 0.4], [14496000.0, 0.5]]
        asks = [[14501000.0, 0.1], [14502000.0, 0.2], [14503000.0, 0.3],
                [14504000.0, 0.4], [14505000.0, 0.5]]
        rec.record(bids, asks, timestamp=1000.0)
        rec.flush()

        # feature_enricher で読み込めることを検証
        ob_df = load_raw_orderbook(tmp_path)
        assert len(ob_df) == 1
        assert ob_df.iloc[0]["ts"] == 1000.0
        assert ob_df.iloc[0]["best_bid"] == 14500000.0
        assert ob_df.iloc[0]["best_ask"] == 14501000.0
        assert ob_df.iloc[0]["bid_vol_5"] == pytest.approx(0.1 + 0.2 + 0.3 + 0.4 + 0.5)


# ===== MakerPriceCalculator OB キャッシュ テスト =====


class TestMakerPriceOBCache:
    """compute_imbalance で _last_ob_snapshot が保持されること."""

    @pytest.mark.asyncio
    async def test_last_ob_snapshot_stored(self) -> None:
        config = FillTestConfig()
        ffd = FastFillDefense(
            config=FastFillDefenseConfig(),
            base_offset_ratio=config.spread_offset_ratio,
        )
        calc = MakerPriceCalculator(
            config=config,
            fast_fill_defense=ffd,
            regime_detector=None,
            base_offset_ratio=config.spread_offset_ratio,
        )
        assert calc._last_ob_snapshot is None

        # Mock adapter
        mock_ob = OrderBookSnapshot(
            timestamp=1000.0,
            bids=[(14500000, 0.1), (14499000, 0.2)],
            asks=[(14501000, 0.1), (14502000, 0.2)],
            exchange="coincheck",
        )
        adapter = AsyncMock()
        adapter.get_orderbook = AsyncMock(return_value=mock_ob)

        result = await calc.compute_imbalance(adapter, "btc_jpy", depth=5)
        assert calc._last_ob_snapshot is mock_ob
        assert result.imbalance == 0.0  # equal volumes → 0

    @pytest.mark.asyncio
    async def test_ob_snapshot_has_bids_asks(self) -> None:
        """キャッシュされた OB の bids/asks が OBRecorder に渡せる形式."""
        config = FillTestConfig()
        ffd = FastFillDefense(
            config=FastFillDefenseConfig(),
            base_offset_ratio=config.spread_offset_ratio,
        )
        calc = MakerPriceCalculator(
            config=config,
            fast_fill_defense=ffd,
            regime_detector=None,
            base_offset_ratio=config.spread_offset_ratio,
        )

        mock_ob = OrderBookSnapshot(
            timestamp=1000.0,
            bids=[(14500000, 0.1)],
            asks=[(14501000, 0.3)],
            exchange="coincheck",
        )
        adapter = AsyncMock()
        adapter.get_orderbook = AsyncMock(return_value=mock_ob)

        await calc.compute_imbalance(adapter, "btc_jpy", depth=5)
        ob = calc._last_ob_snapshot
        assert hasattr(ob, "bids")
        assert hasattr(ob, "asks")
        assert hasattr(ob, "timestamp")


# ===== 統合テスト: OBRecorder + MakerPriceCalculator =====


class TestOBRecorderIntegration:
    """OBRecorder が MakerPriceCalculator の OB スナップショットを記録できること."""

    @pytest.mark.asyncio
    async def test_end_to_end_record_and_enrich(self, tmp_path: Path) -> None:
        """OB record → flush → feature_enricher で OB matched > 0."""
        config = FillTestConfig()
        ffd = FastFillDefense(
            config=FastFillDefenseConfig(),
            base_offset_ratio=config.spread_offset_ratio,
        )
        calc = MakerPriceCalculator(
            config=config,
            fast_fill_defense=ffd,
            regime_detector=None,
            base_offset_ratio=config.spread_offset_ratio,
        )

        # OBRecorder
        rec = OBRecorder(raw_dir=tmp_path, enabled=True)

        # 5 level OB
        bids = [(14500000, 0.1), (14499000, 0.2), (14498000, 0.3),
                (14497000, 0.4), (14496000, 0.5)]
        asks = [(14501000, 0.1), (14502000, 0.2), (14503000, 0.3),
                (14504000, 0.4), (14505000, 0.5)]
        mock_ob = OrderBookSnapshot(
            timestamp=1000.0, bids=bids, asks=asks, exchange="coincheck",
        )
        adapter = AsyncMock()
        adapter.get_orderbook = AsyncMock(return_value=mock_ob)

        # Simulate cycle: fetch OB, record
        await calc.compute_imbalance(adapter, "btc_jpy", depth=5)
        ob = calc._last_ob_snapshot
        assert ob is not None
        rec.record(ob.bids, ob.asks, ob.timestamp)
        rec.flush()

        # Verify feature_enricher can match
        ob_df = load_raw_orderbook(tmp_path)
        assert len(ob_df) == 1
        features = _find_nearest_ob(ob_df, 1000.0, tolerance_sec=5)
        assert not (features["spread_bps_ob"] != features["spread_bps_ob"])  # not NaN
