"""Tests for scripts/v460/analysis/analyze_fill_logs.py (162# P0)."""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime, timezone

import pytest

from scripts.v460.analysis.analysis_common import (
    Record,
    load_and_filter_records,
)
from scripts.v460.analysis.analyze_fill_logs import (
    build_json_summary,
    build_parser,
    section_basic,
    section_header,
    section_inventory_health,
    section_mcb_impact,
    section_model_used,
    section_roundtrip,
    section_side,
    section_spread_fill_quality,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_record(
    *,
    side: str = "buy",
    filled: bool = True,
    git_sha: str = "abc123",
    run_id: str = "run_001",
    regime: str = "ranging",
    timestamp: float = 1771900000.0,
    pnl30: float = 1.5,
) -> Record:
    return {
        "side": side,
        "filled": filled,
        "git_sha": git_sha,
        "run_id": run_id,
        "regime": regime,
        "timestamp": timestamp,
        "post_fill_30s_pnl": pnl30 if filled else None,
        "cancel_reason": None if filled else "timeout",
        "skip_gate_skipped": False,
        "adverse_selected": False,
    }


@pytest.fixture
def sample_records() -> list[Record]:
    """12 records spanning 2 days, 2 shas, 2 sides."""
    base_ts = datetime(2026, 2, 20, 0, 0, tzinfo=timezone.utc).timestamp()
    recs = []
    for i in range(12):
        recs.append(
            _make_record(
                side="buy" if i % 2 == 0 else "sell",
                filled=i % 3 != 0,  # 2/3 filled
                git_sha="sha_A" if i < 6 else "sha_B",
                run_id="run_1" if i < 6 else "run_2",
                regime="ranging" if i < 8 else "trending_up",
                timestamp=base_ts + i * 3600,  # 1h apart
                pnl30=float(i) - 5.0,
            )
        )
    return recs


@pytest.fixture
def tmp_data_dir(tmp_path: pathlib.Path, sample_records: list[Record]) -> pathlib.Path:
    """Write sample records as JSONL to a temp dir."""
    p = tmp_path / "fill_records_20260220.jsonl"
    lines = [json.dumps(r, ensure_ascii=False) for r in sample_records]
    p.write_text("\n".join(lines), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: CLI parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.data_dir == "results/v460/fill_test"
        assert args.run_id is None
        assert args.git_sha is None

    def test_all_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--data-dir", "/tmp/test",
            "--run-id", "run_001",
            "--git-sha", "abc123",
            "--date-from", "2026-02-20",
            "--date-to", "2026-02-21",
            "--side", "buy",
            "--regime", "ranging",
            "--output", "out.txt",
            "--json",
        ])
        assert args.data_dir == "/tmp/test"
        assert args.run_id == "run_001"
        assert args.git_sha == "abc123"
        assert args.date_from == "2026-02-20"
        assert args.date_to == "2026-02-21"
        assert args.side == "buy"
        assert args.regime == "ranging"
        assert args.output == "out.txt"
        assert args.json is True


# ---------------------------------------------------------------------------
# Tests: Filtering
# ---------------------------------------------------------------------------

class TestApplyFilters:
    """load_and_filter_records の内部フィルタロジックをテスト (analysis_common 経由)."""

    def _filter(
        self,
        records: list[Record],
        *,
        run_id: str | None = None,
        git_sha: str | None = None,
        side: str | None = None,
        regime: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[Record]:
        """テスト用ヘルパー: analysis_common のフィルタ API を直接呼び出す."""
        from ztb.metrics.fill_quality import apply_fill_record_filters as _apply
        filtered, _ = _apply(
            records,
            run_id=run_id,
            git_sha=git_sha,
            date_from=date_from,
            date_to=date_to,
        )
        if side:
            filtered = [r for r in filtered if r.get("side") == side]
        if regime:
            filtered = [r for r in filtered if r.get("regime") == regime]
        return filtered

    def test_no_filter(self, sample_records: list[Record]) -> None:
        result = self._filter(sample_records)
        assert len(result) == 12

    def test_run_id_filter(self, sample_records: list[Record]) -> None:
        result = self._filter(sample_records, run_id="run_1")
        assert len(result) == 6
        assert all(r["run_id"] == "run_1" for r in result)

    def test_git_sha_prefix(self, sample_records: list[Record]) -> None:
        result = self._filter(sample_records, git_sha="sha_A")
        assert len(result) == 6

    def test_git_sha_prefix_partial(self, sample_records: list[Record]) -> None:
        result = self._filter(sample_records, git_sha="sha_")
        assert len(result) == 12  # both sha_A and sha_B match

    def test_side_filter(self, sample_records: list[Record]) -> None:
        result = self._filter(sample_records, side="buy")
        assert all(r["side"] == "buy" for r in result)
        assert len(result) == 6

    def test_regime_filter(self, sample_records: list[Record]) -> None:
        result = self._filter(sample_records, regime="trending_up")
        assert len(result) == 4
        assert all(r["regime"] == "trending_up" for r in result)

    def test_date_from_filter(self, sample_records: list[Record]) -> None:
        # Records span 12h from base. Filter to last 6h.
        mid_date = "2026-02-20"  # same day, so all should pass
        result = self._filter(sample_records, date_from=mid_date)
        assert len(result) == 12

    def test_date_to_filter_exclusive(self, sample_records: list[Record]) -> None:
        """date_to is inclusive of the day."""
        result = self._filter(sample_records, date_to="2026-02-19")
        assert len(result) == 0  # all records are Feb 20

    def test_combined_filters(self, sample_records: list[Record]) -> None:
        result = self._filter(
            sample_records,
            git_sha="sha_A",
            side="buy",
        )
        assert len(result) == 3  # first 6 records, even indices (0, 2, 4)


# ---------------------------------------------------------------------------
# Tests: Data Loading
# ---------------------------------------------------------------------------

class TestLoadRecords:
    def test_basic_load(self, tmp_data_dir: pathlib.Path) -> None:
        records = load_and_filter_records(str(tmp_data_dir))
        assert len(records) == 12

    def test_date_filter_match(self, tmp_data_dir: pathlib.Path) -> None:
        records = load_and_filter_records(
            str(tmp_data_dir), date_from="2026-02-20", date_to="2026-02-20",
        )
        assert len(records) == 12


# ---------------------------------------------------------------------------
# Tests: Sections
# ---------------------------------------------------------------------------

class TestSections:
    def test_section_header(self, sample_records: list[Record]) -> None:
        args = argparse.Namespace(
            data_dir="test", run_id=None, git_sha=None,
            date_from=None, date_to=None, side=None, regime=None,
        )
        lines = section_header(sample_records, args)
        text = "\n".join(lines)
        assert "フィルタ条件" in text
        assert "run_id" in text
        assert "git_sha_unique" in text

    def test_section_basic(self, sample_records: list[Record]) -> None:
        lines = section_basic(sample_records)
        text = "\n".join(lines)
        assert "Total: 12" in text
        assert "Filled: 8" in text  # 2/3 of 12

    def test_section_side(self, sample_records: list[Record]) -> None:
        lines = section_side(sample_records)
        text = "\n".join(lines)
        assert "buy:" in text
        assert "sell:" in text


# ---------------------------------------------------------------------------
# Tests: JSON output
# ---------------------------------------------------------------------------

class TestJsonSummary:
    def test_basic_json(self, sample_records: list[Record]) -> None:
        args = argparse.Namespace(
            data_dir="test", run_id=None, git_sha=None,
            date_from=None, date_to=None, side=None, regime=None,
        )
        result = build_json_summary(sample_records, args)
        assert result["total_records"] == 12
        assert result["filled"] == 8
        assert "sides" in result
        assert "buy" in result["sides"]
        assert "sell" in result["sides"]
        assert "git_sha_distribution" in result

    def test_filters_in_json(self, sample_records: list[dict]) -> None:
        args = argparse.Namespace(
            data_dir="test", run_id="run_1", git_sha="sha_A",
            date_from="2026-02-20", date_to="2026-02-20", side=None, regime=None,
        )
        result = build_json_summary(sample_records, args)
        assert result["filters"]["run_id"] == "run_1"
        assert result["filters"]["git_sha"] == "sha_A"


# ---------------------------------------------------------------------------
# Tests: section_model_used (165# 7.3)
# ---------------------------------------------------------------------------


class TestSectionModelUsed:
    def test_basic(self, sample_records: list[dict]) -> None:
        """Returns lines with model used data."""
        # Add skip_gate_model_used to records
        for r in sample_records:
            r["skip_gate_model_used"] = "primary:side_sell" if r["side"] == "sell" else "none"
        lines = section_model_used(sample_records)
        assert any("Model Used" in l for l in lines)

    def test_empty_no_crash(self) -> None:
        lines = section_model_used([])
        assert any("no fills" in l for l in lines)

    def test_multiple_models(self) -> None:
        recs = [
            _make_record(filled=True, pnl30=1.0),
            _make_record(filled=True, pnl30=-2.0),
            _make_record(filled=True, pnl30=3.0),
        ]
        recs[0]["skip_gate_model_used"] = "none"
        recs[1]["skip_gate_model_used"] = "primary:side_sell"
        recs[2]["skip_gate_model_used"] = "primary:unified"
        lines = section_model_used(recs)
        joined = "\n".join(lines)
        assert "none" in joined
        assert "primary:side_sell" in joined
        assert "primary:unified" in joined


# ---------------------------------------------------------------------------
# Tests: section_roundtrip (650#)
# ---------------------------------------------------------------------------


class TestSectionRoundtrip:
    def test_empty(self) -> None:
        lines = section_roundtrip([])
        assert any("insufficient" in l for l in lines)

    def test_basic_pairing(self) -> None:
        """buy → sell ペアが正しく認識され PnL が計算される."""
        recs = [
            {**_make_record(side="buy", filled=True, pnl30=1.0, timestamp=1000.0),
             "fill_price": 10_000_000},
            {**_make_record(side="sell", filled=True, pnl30=-1.0, timestamp=1060.0),
             "fill_price": 10_010_000},
        ]
        lines = section_roundtrip(recs)
        text = "\n".join(lines)
        assert "RTs: 1" in text
        assert "Win: 1" in text

    def test_same_side_skipped(self) -> None:
        """同一 side が連続する場合はスキップされる."""
        recs = [
            {**_make_record(side="buy", filled=True, pnl30=1.0, timestamp=1000.0),
             "fill_price": 10_000_000},
            {**_make_record(side="buy", filled=True, pnl30=2.0, timestamp=1060.0),
             "fill_price": 10_001_000},
            {**_make_record(side="sell", filled=True, pnl30=-1.0, timestamp=1120.0),
             "fill_price": 10_005_000},
        ]
        lines = section_roundtrip(recs)
        text = "\n".join(lines)
        assert "RTs: 1" in text

    def test_worst_roundtrips_shown(self) -> None:
        recs = [
            {**_make_record(side="sell", filled=True, pnl30=-5.0, timestamp=1000.0),
             "fill_price": 10_000_000},
            {**_make_record(side="buy", filled=True, pnl30=0.0, timestamp=1060.0),
             "fill_price": 10_020_000},
        ]
        lines = section_roundtrip(recs)
        text = "\n".join(lines)
        assert "Worst Roundtrips" in text


# ---------------------------------------------------------------------------
# Tests: section_inventory_health (650#)
# ---------------------------------------------------------------------------


class TestSectionInventoryHealth:
    def test_empty(self) -> None:
        lines = section_inventory_health([])
        assert any("no records" in l for l in lines)

    def test_preflight_counting(self) -> None:
        recs = [
            _make_record(side="buy", filled=False),
            _make_record(side="buy", filled=False),
            _make_record(side="sell", filled=True, pnl30=1.0),
        ]
        recs[0]["cancel_reason"] = "preflight_insufficient"
        recs[1]["cancel_reason"] = "preflight_insufficient"
        lines = section_inventory_health(recs)
        text = "\n".join(lines)
        assert "preflight_insufficient: 2/3" in text

    def test_balance_snapshot(self) -> None:
        rec = _make_record(side="sell", filled=True, pnl30=1.0, timestamp=2000.0)
        rec["balance_jpy_at_order"] = 500.0
        rec["balance_btc_at_order"] = 0.002
        rec["mid_at_order"] = 10_000_000
        lines = section_inventory_health([rec])
        text = "\n".join(lines)
        assert "JPY ratio" in text
        assert "Inventory imbalance" in text


# ---------------------------------------------------------------------------
# Tests: section_mcb_impact (650#)
# ---------------------------------------------------------------------------


class TestSectionMcbImpact:
    def test_no_halts(self) -> None:
        recs = [_make_record(filled=True, pnl30=1.0)]
        lines = section_mcb_impact(recs)
        text = "\n".join(lines)
        assert "MCB halts: 0" in text

    def test_with_halts(self) -> None:
        recs = [
            _make_record(side="buy", filled=True, pnl30=2.0, timestamp=1000.0),
            _make_record(side="sell", filled=False, timestamp=1500.0),
            _make_record(side="buy", filled=True, pnl30=-3.0, timestamp=2000.0),
        ]
        recs[1]["cancel_reason"] = "mcb_halt"
        lines = section_mcb_impact(recs)
        text = "\n".join(lines)
        assert "MCB halts: 1" in text


# ---------------------------------------------------------------------------
# Tests: section_spread_fill_quality (650#)
# ---------------------------------------------------------------------------


class TestSectionSpreadFillQuality:
    def test_insufficient(self) -> None:
        recs = [_make_record(filled=True, pnl30=1.0)]
        lines = section_spread_fill_quality(recs)
        assert any("insufficient" in l for l in lines)

    def test_quartile_output(self) -> None:
        recs = []
        for i in range(8):
            r = _make_record(
                side="buy" if i % 2 == 0 else "sell",
                filled=True,
                pnl30=float(i) - 3.0,
                timestamp=1000.0 + i * 60,
            )
            r["spread_bps"] = 1.0 + i * 0.5
            r["adverse_selected"] = i < 2
            r["queue_wait_sec"] = 10.0 + i
            recs.append(r)
        lines = section_spread_fill_quality(recs)
        text = "\n".join(lines)
        assert "Q1" in text or "Q2" in text
        assert "avg_pnl30" in text
