"""554# CalibrationMap offline batch + raw gap fill テスト."""

from __future__ import annotations

import gzip
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ════════════════════════════════════════════════════════════
# Raw trades → OHLCV
# ════════════════════════════════════════════════════════════

class TestRawTradesToOhlcv:
    """_raw_trades_to_ohlcv_1min のテスト."""

    def _make_trades_gz(self, tmp_path: Path, records: list[dict]) -> Path:
        path = tmp_path / "20260315.jsonl.gz"
        with gzip.open(path, "wt") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def test_basic_conversion(self, tmp_path: Path) -> None:
        from scripts.v460.ml.update_training_data import _raw_trades_to_ohlcv_1min

        base_ts = 1773532800.0  # 2026-03-13 00:00:00 UTC
        records = [
            {"ts": base_ts + 0, "price": 100.0, "amount": 1.0, "side": "buy"},
            {"ts": base_ts + 10, "price": 110.0, "amount": 2.0, "side": "sell"},
            {"ts": base_ts + 30, "price": 95.0, "amount": 0.5, "side": "buy"},
            {"ts": base_ts + 50, "price": 105.0, "amount": 1.5, "side": "buy"},
        ]
        path = self._make_trades_gz(tmp_path, records)
        result = _raw_trades_to_ohlcv_1min(path)

        assert len(result) == 1  # All within same minute
        assert list(result.columns) == [
            "timestamp", "open", "high", "low", "close", "volume",
        ]
        row = result.iloc[0]
        assert row["open"] == 100.0
        assert row["high"] == 110.0
        assert row["low"] == 95.0
        assert row["close"] == 105.0
        assert row["volume"] == pytest.approx(5.0)

    def test_multiple_minutes(self, tmp_path: Path) -> None:
        from scripts.v460.ml.update_training_data import _raw_trades_to_ohlcv_1min

        base_ts = 1773532800.0
        records = [
            {"ts": base_ts, "price": 100.0, "amount": 1.0, "side": "buy"},
            {"ts": base_ts + 60, "price": 200.0, "amount": 2.0, "side": "sell"},
            {"ts": base_ts + 120, "price": 300.0, "amount": 3.0, "side": "buy"},
        ]
        path = self._make_trades_gz(tmp_path, records)
        result = _raw_trades_to_ohlcv_1min(path)

        assert len(result) == 3

    def test_empty_file(self, tmp_path: Path) -> None:
        from scripts.v460.ml.update_training_data import _raw_trades_to_ohlcv_1min

        path = self._make_trades_gz(tmp_path, [])
        result = _raw_trades_to_ohlcv_1min(path)
        assert result.empty


class TestFillGapFromRaw:
    """fill_gap_from_raw のテスト."""

    def test_no_raw_dir(self, tmp_path: Path) -> None:
        from scripts.v460.ml.update_training_data import fill_gap_from_raw

        result = fill_gap_from_raw(
            parquet_path=tmp_path / "test.parquet",
            raw_trades_dir=tmp_path / "nonexistent",
        )
        assert result == 0


# ════════════════════════════════════════════════════════════
# CalibrationMap offline batch
# ════════════════════════════════════════════════════════════

class TestSideToAction:
    """_side_to_action のテスト."""

    def test_buy(self) -> None:
        from scripts.v460.ml.calibration_batch import _side_to_action
        assert _side_to_action("buy") == 0.3  # 559# fix: Buy bin に統一

    def test_sell(self) -> None:
        from scripts.v460.ml.calibration_batch import _side_to_action
        assert _side_to_action("sell") == -0.3  # 559# fix: Sell bin に統一


class TestBuildCalibrationMap:
    """build_calibration_map のテスト."""

    def _make_fill_records(self, tmp_path: Path, records: list[dict]) -> Path:
        d = tmp_path / "fill_test"
        d.mkdir(parents=True, exist_ok=True)
        path = d / "fill_records_20260322.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return d

    def test_builds_from_records(self, tmp_path: Path) -> None:
        from scripts.v460.ml.calibration_batch import build_calibration_map

        records = []
        base_ts = 1774000000.0
        for i in range(100):
            records.append({
                "cycle_id": f"cycle_{i}",
                "timestamp": base_ts + i * 60,
                "side": "buy" if i % 2 == 0 else "sell",
                "filled": True,
                "post_fill_30s_pnl": 10.0 if i % 3 == 0 else -5.0,
                "regime": "ranging",
                "regime_at_order": "ranging",
                "regime_confidence": 1.0,
                "order_price": 10000000,
                "order_quantity": 0.001,
                "fill_price": 10000000,
                "spread_at_order": 2000.0,
                "spread_bps": 2.0,
            })

        d = self._make_fill_records(tmp_path, records)
        output = tmp_path / "cal.json"

        export = build_calibration_map(
            results_dir=d,
            output_path=output,
        )

        assert output.exists()
        assert export["meta"]["n_records_used"] == 100
        assert "stats" in export
        assert "global" in export["stats"]

    def test_unfilled_records_skipped(self, tmp_path: Path) -> None:
        from scripts.v460.ml.calibration_batch import build_calibration_map

        records = [
            {
                "cycle_id": "c1", "timestamp": 1774000000.0,
                "side": "buy", "filled": False,
                "post_fill_30s_pnl": None,
                "regime": "ranging", "regime_at_order": "ranging",
                "regime_confidence": 1.0,
                "order_price": 10000000, "order_quantity": 0.001,
                "fill_price": 10000000, "spread_at_order": 2000.0,
                "spread_bps": 2.0,
            },
            {
                "cycle_id": "c2", "timestamp": 1774000060.0,
                "side": "sell", "filled": True,
                "post_fill_30s_pnl": 5.0,
                "regime": "ranging", "regime_at_order": "ranging",
                "regime_confidence": 1.0,
                "order_price": 10000000, "order_quantity": 0.001,
                "fill_price": 10000000, "spread_at_order": 2000.0,
                "spread_bps": 2.0,
            },
        ]

        d = self._make_fill_records(tmp_path, records)
        output = tmp_path / "cal.json"

        export = build_calibration_map(
            results_dir=d,
            output_path=output,
        )

        assert export["meta"]["n_records_used"] == 1

    def test_days_filter(self, tmp_path: Path) -> None:
        from scripts.v460.ml.calibration_batch import build_calibration_map

        now_ts = datetime.now(timezone.utc).timestamp()
        records = [
            {
                "cycle_id": f"c{i}", "timestamp": now_ts - (30 - i) * 86400,
                "side": "buy", "filled": True,
                "post_fill_30s_pnl": 10.0,
                "regime": "ranging", "regime_at_order": "ranging",
                "regime_confidence": 1.0,
                "order_price": 10000000, "order_quantity": 0.001,
                "fill_price": 10000000, "spread_at_order": 2000.0,
                "spread_bps": 2.0,
            }
            for i in range(30)
        ]

        d = self._make_fill_records(tmp_path, records)
        output = tmp_path / "cal.json"

        # Only last 7 days
        export = build_calibration_map(
            results_dir=d, output_path=output, days=7,
        )
        assert export["meta"]["n_records_used"] < 30
        assert export["meta"]["days_filter"] == 7


class TestLoadCalibrationState:
    """load_calibration_state のテスト."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        from scripts.v460.ml.calibration_batch import load_calibration_state

        result = load_calibration_state(tmp_path / "no.json")
        assert result is None

    def test_roundtrip(self, tmp_path: Path) -> None:
        from scripts.v460.ml.calibration_batch import (
            build_calibration_map,
            load_calibration_state,
        )

        records = [
            {
                "cycle_id": f"c{i}", "timestamp": 1774000000.0 + i * 60,
                "side": "buy", "filled": True,
                "post_fill_30s_pnl": 10.0 if i % 2 == 0 else -5.0,
                "regime": "trending", "regime_at_order": "trending",
                "regime_confidence": 1.0,
                "order_price": 10000000, "order_quantity": 0.001,
                "fill_price": 10000000, "spread_at_order": 2000.0,
                "spread_bps": 2.0,
            }
            for i in range(50)
        ]

        d = tmp_path / "fill_test"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "fill_records_20260322.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        output = tmp_path / "cal.json"
        build_calibration_map(results_dir=d, output_path=output)

        cal_map = load_calibration_state(output)
        assert cal_map is not None

        stats = cal_map.get_stats("trending", 0.3)  # 559# fix: Buy bin に統一
        assert "l1" in stats
        assert "fallback" in stats
        assert stats["l1"]["n_eff"] > 0
