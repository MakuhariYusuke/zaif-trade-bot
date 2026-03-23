"""analysis_common.py のユニットテスト."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from scripts.v460.analysis.analysis_common import (
    AS_THRESHOLD_BPS,
    DEFAULT_RESULTS_DIR,
    PNL_FIELD_PRIORITY,
    SEVERE_AS_THRESHOLD_BPS,
    Record,
    add_common_filter_args,
    add_output_args,
    add_side_regime_args,
    extract_filled,
    extract_pnl_array,
    extract_pnl_list,
    get_pnl,
    load_and_filter_records,
    load_records_from_args,
    record_to_utc_hour,
    write_json_output,
    write_output,
)


# ======================================================================
# 定数
# ======================================================================


class TestConstants:
    def test_default_results_dir(self) -> None:
        assert DEFAULT_RESULTS_DIR == "results/v460/fill_test"

    def test_as_thresholds(self) -> None:
        assert AS_THRESHOLD_BPS == -3.0
        assert SEVERE_AS_THRESHOLD_BPS == -10.0

    def test_pnl_field_priority(self) -> None:
        assert PNL_FIELD_PRIORITY == (
            "ev_weighted_pnl",
            "post_fill_30s_pnl",
            "pnl_bps",
        )


# ======================================================================
# CLI 引数ビルダー
# ======================================================================


class TestCLIArgBuilders:
    def test_add_common_filter_args(self) -> None:
        parser = argparse.ArgumentParser()
        add_common_filter_args(parser)
        args = parser.parse_args([])
        assert args.results_dir == DEFAULT_RESULTS_DIR
        assert args.date_from is None
        assert args.date_to is None
        assert args.git_sha is None
        assert args.run_id is None

    def test_add_common_filter_args_with_values(self) -> None:
        parser = argparse.ArgumentParser()
        add_common_filter_args(parser)
        args = parser.parse_args([
            "--results-dir", "/tmp/data",
            "--date-from", "2026-03-01",
            "--date-to", "2026-03-15",
            "--git-sha", "abc1234",
            "--run-id", "run_001",
        ])
        assert args.results_dir == "/tmp/data"
        assert args.date_from == "2026-03-01"
        assert args.date_to == "2026-03-15"
        assert args.git_sha == "abc1234"
        assert args.run_id == "run_001"

    def test_add_side_regime_args(self) -> None:
        parser = argparse.ArgumentParser()
        add_side_regime_args(parser)
        args = parser.parse_args([])
        assert args.side is None
        assert args.regime is None

    def test_add_side_regime_args_with_values(self) -> None:
        parser = argparse.ArgumentParser()
        add_side_regime_args(parser)
        args = parser.parse_args(["--side", "buy", "--regime", "trending"])
        assert args.side == "buy"
        assert args.regime == "trending"

    def test_add_side_regime_invalid_side(self) -> None:
        parser = argparse.ArgumentParser()
        add_side_regime_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--side", "invalid"])

    def test_add_output_args(self) -> None:
        parser = argparse.ArgumentParser()
        add_output_args(parser)
        args = parser.parse_args([])
        assert args.output is None
        assert args.json is False

    def test_add_output_args_with_values(self) -> None:
        parser = argparse.ArgumentParser()
        add_output_args(parser)
        args = parser.parse_args(["--output", "report.txt", "--json"])
        assert args.output == "report.txt"
        assert args.json is True

    def test_all_args_combined(self) -> None:
        """3つのビルダーを同一 parser に追加しても衝突しない."""
        parser = argparse.ArgumentParser()
        add_common_filter_args(parser)
        add_side_regime_args(parser)
        add_output_args(parser)
        args = parser.parse_args([
            "--results-dir", "data/",
            "--side", "sell",
            "--json",
        ])
        assert args.results_dir == "data/"
        assert args.side == "sell"
        assert args.json is True


# ======================================================================
# PnL 抽出
# ======================================================================


class TestGetPnl:
    def test_ev_weighted_pnl_priority(self) -> None:
        r: Record = {"ev_weighted_pnl": 1.5, "post_fill_30s_pnl": 2.0}
        assert get_pnl(r) == 1.5

    def test_fallback_to_post_fill(self) -> None:
        r: Record = {"post_fill_30s_pnl": 2.5}
        assert get_pnl(r) == 2.5

    def test_fallback_to_pnl_bps(self) -> None:
        r: Record = {"pnl_bps": 3.0}
        assert get_pnl(r) == 3.0

    def test_none_when_no_field(self) -> None:
        r: Record = {"side": "buy"}
        assert get_pnl(r) is None

    def test_skip_nan(self) -> None:
        r: Record = {"ev_weighted_pnl": float("nan"), "post_fill_30s_pnl": 1.0}
        assert get_pnl(r) == 1.0

    def test_skip_inf(self) -> None:
        r: Record = {"ev_weighted_pnl": float("inf"), "post_fill_30s_pnl": 1.0}
        assert get_pnl(r) == 1.0

    def test_none_value_skipped(self) -> None:
        r: Record = {"ev_weighted_pnl": None, "post_fill_30s_pnl": 5.0}
        assert get_pnl(r) == 5.0


class TestExtractPnlArray:
    def test_basic_extraction(self) -> None:
        records: list[Record] = [
            {"post_fill_30s_pnl": 1.0},
            {"post_fill_30s_pnl": -2.0},
            {"post_fill_30s_pnl": 3.0},
        ]
        arr = extract_pnl_array(records)
        np.testing.assert_array_equal(arr, [1.0, -2.0, 3.0])
        assert arr.dtype == np.float64

    def test_skip_none(self) -> None:
        records: list[Record] = [
            {"post_fill_30s_pnl": 1.0},
            {"post_fill_30s_pnl": None},
            {"other": 5.0},
        ]
        arr = extract_pnl_array(records)
        np.testing.assert_array_equal(arr, [1.0])

    def test_empty_input(self) -> None:
        arr = extract_pnl_array([])
        assert len(arr) == 0
        assert arr.dtype == np.float64

    def test_custom_key(self) -> None:
        records: list[Record] = [
            {"ev_weighted_pnl": 10.0},
            {"ev_weighted_pnl": 20.0},
        ]
        arr = extract_pnl_array(records, key="ev_weighted_pnl")
        np.testing.assert_array_equal(arr, [10.0, 20.0])

    def test_nan_filtered(self) -> None:
        records: list[Record] = [
            {"post_fill_30s_pnl": float("nan")},
            {"post_fill_30s_pnl": 1.0},
        ]
        arr = extract_pnl_array(records)
        np.testing.assert_array_equal(arr, [1.0])


class TestExtractPnlList:
    def test_fallback_chain(self) -> None:
        records: list[Record] = [
            {"ev_weighted_pnl": 1.0},
            {"post_fill_30s_pnl": 2.0},
            {"pnl_bps": 3.0},
            {"side": "buy"},
        ]
        result = extract_pnl_list(records)
        assert result == [1.0, 2.0, 3.0]


# ======================================================================
# フィルタヘルパー
# ======================================================================


class TestExtractFilled:
    def test_basic_filter(self) -> None:
        records: list[Record] = [
            {"filled": True, "side": "buy"},
            {"filled": False, "side": "sell"},
            {"filled": True, "side": "sell"},
        ]
        result = extract_filled(records)
        assert len(result) == 2

    def test_side_filter(self) -> None:
        records: list[Record] = [
            {"filled": True, "side": "buy"},
            {"filled": True, "side": "sell"},
            {"filled": True, "side": "buy"},
        ]
        result = extract_filled(records, side="buy")
        assert len(result) == 2
        assert all(r["side"] == "buy" for r in result)

    def test_empty_input(self) -> None:
        assert extract_filled([]) == []

    def test_no_filled(self) -> None:
        records: list[Record] = [{"filled": False}, {"filled": False}]
        assert extract_filled(records) == []


# ======================================================================
# タイムスタンプヘルパー
# ======================================================================


class TestRecordToUtcHour:
    def test_numeric_timestamp(self) -> None:
        # 2026-01-01 15:30:00 UTC
        r: Record = {"timestamp": 1767275400.0}
        h = record_to_utc_hour(r)
        assert h is not None
        assert 0 <= h <= 23

    def test_string_timestamp(self) -> None:
        r: Record = {"timestamp": "2026-01-01T15:30:00+00:00"}
        assert record_to_utc_hour(r) == 15

    def test_string_with_z(self) -> None:
        r: Record = {"timestamp": "2026-01-01T08:00:00Z"}
        assert record_to_utc_hour(r) == 8

    def test_none_timestamp(self) -> None:
        r: Record = {"side": "buy"}
        assert record_to_utc_hour(r) is None

    def test_invalid_type(self) -> None:
        r: Record = {"timestamp": [1, 2, 3]}
        assert record_to_utc_hour(r) is None


# ======================================================================
# データ読み込み
# ======================================================================


class TestLoadAndFilterRecords:
    def test_nonexistent_dir_exits(self) -> None:
        with pytest.raises(SystemExit):
            load_and_filter_records("/nonexistent/path")

    def test_nonexistent_dir_no_exit(self) -> None:
        result = load_and_filter_records(
            "/nonexistent/path", exit_on_empty=False,
        )
        assert result == []


class TestLoadRecordsFromArgs:
    def test_delegates_to_load_and_filter(self) -> None:
        parser = argparse.ArgumentParser()
        add_common_filter_args(parser)
        add_side_regime_args(parser)
        args = parser.parse_args(["--results-dir", "/fake/dir"])

        with pytest.raises(SystemExit):
            load_records_from_args(args)

    def test_uses_defaults(self) -> None:
        """引数未指定でも DEFAULT_RESULTS_DIR が使われる."""
        args = argparse.Namespace()
        with patch(
            "scripts.v460.analysis.analysis_common.load_and_filter_records",
            return_value=[{"filled": True}],
        ) as mock:
            result = load_records_from_args(args, exit_on_empty=False)
            mock.assert_called_once()
            call_args = mock.call_args
            assert call_args[0][0] == DEFAULT_RESULTS_DIR


# ======================================================================
# 出力ヘルパー
# ======================================================================


class TestWriteOutput:
    def test_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        write_output("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_file_output(self, tmp_path: Path) -> None:
        out = tmp_path / "test.txt"
        write_output("content", out)
        assert out.read_text() == "content"

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "dir" / "test.txt"
        write_output("content", out)
        assert out.read_text() == "content"


class TestWriteJsonOutput:
    def test_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        write_json_output({"key": "value"})
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data == {"key": "value"}

    def test_file_output(self, tmp_path: Path) -> None:
        out = tmp_path / "result.json"
        write_json_output({"a": 1}, out)
        data = json.loads(out.read_text())
        assert data == {"a": 1}
