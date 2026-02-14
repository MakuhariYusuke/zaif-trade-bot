"""
G1.1-exec Fill Quality 単体テスト — 009# P2-4 準拠.

fill_quality.py の FillRecord / FillMetrics / compute / judgment を検証する。
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest


# =====================================================================
# FillRecord
# =====================================================================

class TestFillRecord:
    """FillRecord の serialize / deserialize テスト."""

    def test_round_trip(self) -> None:
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="test_001",
            timestamp=1700000000.0,
            side="buy",
            order_price=15000000.0,
            order_quantity=0.001,
            fill_price=15000001.0,
            filled=True,
            cancelled=False,
            queue_wait_sec=12.5,
            mid_at_fill=15000050.0,
            mid_30s_after=15000100.0,
            post_fill_30s_pnl=0.33,
            adverse_selected=False,
        )
        d = r.to_dict()
        r2 = FillRecord.from_dict(d)
        assert r2.cycle_id == r.cycle_id
        assert r2.filled is True
        assert r2.post_fill_30s_pnl == pytest.approx(0.33)

    def test_from_dict_extra_keys_ignored(self) -> None:
        from ztb.metrics.fill_quality import FillRecord

        d = {
            "cycle_id": "x",
            "timestamp": 0.0,
            "side": "sell",
            "order_price": 100.0,
            "order_quantity": 0.01,
            "extra_key": "should_be_ignored",
        }
        r = FillRecord.from_dict(d)
        assert r.cycle_id == "x"
        assert not hasattr(r, "extra_key")

    def test_defaults(self) -> None:
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="d",
            timestamp=0.0,
            side="buy",
            order_price=1.0,
            order_quantity=0.001,
        )
        assert r.filled is False
        assert r.cancelled is False
        assert r.fill_price is None
        assert r.post_fill_30s_pnl is None

    def test_cancel_reason_field(self) -> None:
        """CM-2: cancel_reason フィールドの追加確認."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="cr",
            timestamp=0.0,
            side="buy",
            order_price=1.0,
            order_quantity=0.001,
            cancelled=True,
            cancel_reason="post_only_reject",
        )
        assert r.cancel_reason == "post_only_reject"
        d = r.to_dict()
        assert d["cancel_reason"] == "post_only_reject"
        r2 = FillRecord.from_dict(d)
        assert r2.cancel_reason == "post_only_reject"

    def test_cancel_reason_default_none(self) -> None:
        """CM-2: cancel_reason はデフォルト None."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="cr2",
            timestamp=0.0,
            side="buy",
            order_price=1.0,
            order_quantity=0.001,
        )
        assert r.cancel_reason is None

    def test_new_fields_020(self) -> None:
        """020# O4/O5: run_id, git_sha, adverse_selected_raw フィールド."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="o20",
            timestamp=0.0,
            side="buy",
            order_price=1.0,
            order_quantity=0.001,
            adverse_selected=False,
            adverse_selected_raw=True,
            run_id="1234_abc",
            git_sha="abc1234",
        )
        assert r.adverse_selected_raw is True
        assert r.run_id == "1234_abc"
        assert r.git_sha == "abc1234"
        # round-trip
        d = r.to_dict()
        assert d["adverse_selected_raw"] is True
        assert d["run_id"] == "1234_abc"
        r2 = FillRecord.from_dict(d)
        assert r2.adverse_selected_raw is True
        assert r2.run_id == "1234_abc"
        assert r2.git_sha == "abc1234"

    def test_new_fields_020_defaults_none(self) -> None:
        """020# フィールドはデフォルト None."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="d020",
            timestamp=0.0,
            side="buy",
            order_price=1.0,
            order_quantity=0.001,
        )
        assert r.adverse_selected_raw is None
        assert r.run_id is None
        assert r.git_sha is None

    def test_new_fields_031(self) -> None:
        """031# spread_at_order, error_message, spread_offset_ratio フィールド."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="o31",
            timestamp=0.0,
            side="buy",
            order_price=15000000.0,
            order_quantity=0.001,
            spread_at_order=200.0,
            error_message="test error",
            spread_offset_ratio=0.05,
        )
        assert r.spread_at_order == 200.0
        assert r.error_message == "test error"
        assert r.spread_offset_ratio == 0.05
        # round-trip
        d = r.to_dict()
        assert d["spread_at_order"] == 200.0
        assert d["error_message"] == "test error"
        assert d["spread_offset_ratio"] == 0.05
        r2 = FillRecord.from_dict(d)
        assert r2.spread_at_order == 200.0
        assert r2.error_message == "test error"
        assert r2.spread_offset_ratio == 0.05

    def test_new_fields_031_defaults_none(self) -> None:
        """031# フィールドはデフォルト None."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="d031",
            timestamp=0.0,
            side="buy",
            order_price=1.0,
            order_quantity=0.001,
        )
        assert r.spread_at_order is None
        assert r.error_message is None
        assert r.spread_offset_ratio is None

    def test_new_fields_031_backward_compat(self) -> None:
        """031# 旧データ (フィールドなし) からの読み込み互換性."""
        from ztb.metrics.fill_quality import FillRecord

        old_data = {
            "cycle_id": "old",
            "timestamp": 0.0,
            "side": "buy",
            "order_price": 1.0,
            "order_quantity": 0.001,
            "filled": True,
        }
        r = FillRecord.from_dict(old_data)
        assert r.spread_at_order is None
        assert r.error_message is None
        assert r.spread_offset_ratio is None


# =====================================================================
# compute_fill_metrics
# =====================================================================

class TestComputeFillMetrics:
    """compute_fill_metrics の指標算出テスト."""

    def _make_records(
        self,
        n: int = 100,
        fill_rate: float = 0.9,
        queue_wait: float = 20.0,
        pnl_mean: float = 0.5,
        adverse_rate: float = 0.1,
        days: int = 3,
    ) -> list:
        """テスト用の FillRecord リストを生成."""
        from ztb.metrics.fill_quality import FillRecord

        records = []
        base_ts = 1700000000.0
        for i in range(n):
            day_offset = (i % days) * 86400
            filled = (i / n) < fill_rate
            pnl = np.random.normal(pnl_mean, 1.0) if filled else None
            adverse = pnl < 0 if pnl is not None else None

            records.append(FillRecord(
                cycle_id=f"cycle_{i}",
                timestamp=base_ts + day_offset + i * 120,
                side="buy" if i % 2 == 0 else "sell",
                order_price=15000000.0 + i,
                order_quantity=0.001,
                fill_price=15000000.0 + i if filled else None,
                filled=filled,
                cancelled=not filled,
                queue_wait_sec=queue_wait + np.random.uniform(-5, 5) if filled else 0.0,
                mid_at_fill=15000050.0 if filled else None,
                mid_30s_after=15000050.0 + pnl * 1.5 if filled and pnl is not None else None,
                post_fill_30s_pnl=pnl,
                adverse_selected=adverse,
            ))
        return records

    def test_empty_records(self) -> None:
        from ztb.metrics.fill_quality import FillMetrics, compute_fill_metrics

        m = compute_fill_metrics([])
        assert m.total_orders == 0
        assert m.fill_rate_p90 == 0.0

    def test_all_filled(self) -> None:
        from ztb.metrics.fill_quality import compute_fill_metrics

        records = self._make_records(n=50, fill_rate=1.0, days=2)
        m = compute_fill_metrics(records)
        assert m.total_orders == 50
        assert m.filled_orders == 50
        assert m.cancelled_orders == 0
        assert m.fill_rate_p90 >= 0.9
        assert m.cancel_ratio == 0.0

    def test_partial_fill(self) -> None:
        from ztb.metrics.fill_quality import compute_fill_metrics

        records = self._make_records(n=100, fill_rate=0.7, days=5)
        m = compute_fill_metrics(records)
        assert m.filled_orders == 70
        assert m.cancelled_orders == 30
        assert m.cancel_ratio == pytest.approx(0.3)

    def test_queue_wait_median(self) -> None:
        from ztb.metrics.fill_quality import FillRecord, compute_fill_metrics

        records = []
        base_ts = 1700000000.0
        for i in range(10):
            records.append(FillRecord(
                cycle_id=f"qw_{i}",
                timestamp=base_ts + i * 120,
                side="buy",
                order_price=100.0,
                order_quantity=0.001,
                filled=True,
                queue_wait_sec=float(10 + i * 2),  # 10, 12, ..., 28
            ))
        m = compute_fill_metrics(records)
        # median of [10, 12, 14, 16, 18, 20, 22, 24, 26, 28] = 19
        assert m.queue_wait_median_sec == pytest.approx(19.0)

    def test_adverse_selection(self) -> None:
        from ztb.metrics.fill_quality import FillRecord, compute_fill_metrics

        records = []
        base_ts = 1700000000.0
        for i in range(10):
            records.append(FillRecord(
                cycle_id=f"as_{i}",
                timestamp=base_ts + i * 120,
                side="buy",
                order_price=100.0,
                order_quantity=0.001,
                filled=True,
                adverse_selected=(i < 3),  # 30% adverse
                post_fill_30s_pnl=-1.0 if i < 3 else 1.0,
            ))
        m = compute_fill_metrics(records)
        assert m.adverse_selection_ratio == pytest.approx(0.3)

    def test_adverse_selection_raw_020(self) -> None:
        """020# O5: E5-raw (deadzone 非適用) が並行計算されることを検証."""
        from ztb.metrics.fill_quality import FillRecord, compute_fill_metrics

        records = []
        base_ts = 1700000000.0
        for i in range(10):
            records.append(FillRecord(
                cycle_id=f"asr_{i}",
                timestamp=base_ts + i * 120,
                side="buy",
                order_price=100.0,
                order_quantity=0.001,
                filled=True,
                adverse_selected=(i < 2),       # 20% (deadzone 適用後)
                adverse_selected_raw=(i < 4),    # 40% (raw)
                post_fill_30s_pnl=-1.0 if i < 2 else 1.0,
            ))
        m = compute_fill_metrics(records)
        assert m.adverse_selection_ratio == pytest.approx(0.2)
        assert m.adverse_selection_ratio_raw == pytest.approx(0.4)

    def test_sample_sufficient_true(self) -> None:
        """020# O1: n>=200 & 3暦日 → sample_sufficient=True."""
        from ztb.metrics.fill_quality import FillRecord, compute_fill_metrics

        records = []
        # UTC midnight-aligned timestamp to avoid date-boundary issues
        base_ts = 1700006400.0  # 2023-11-15 00:00:00 UTC
        for day in range(3):
            for i in range(70):
                records.append(FillRecord(
                    cycle_id=f"ss_d{day}_{i}",
                    timestamp=base_ts + day * 86400 + i * 120,
                    side="buy",
                    order_price=100.0,
                    order_quantity=0.001,
                    filled=True,
                    queue_wait_sec=10.0,
                    post_fill_30s_pnl=0.5,
                    adverse_selected=False,
                ))
        m = compute_fill_metrics(records)
        assert m.total_orders == 210
        assert m.measurement_days >= 3
        assert m.sample_sufficient is True

    def test_sample_sufficient_false_n(self) -> None:
        """020# O1: n<200 → sample_sufficient=False."""
        from ztb.metrics.fill_quality import FillRecord, compute_fill_metrics

        records = []
        base_ts = 1700000000.0
        for day in range(3):
            for i in range(30):
                records.append(FillRecord(
                    cycle_id=f"ssf_d{day}_{i}",
                    timestamp=base_ts + day * 86400 + i * 120,
                    side="buy",
                    order_price=100.0,
                    order_quantity=0.001,
                    filled=True,
                ))
        m = compute_fill_metrics(records)
        assert m.total_orders == 90
        assert m.sample_sufficient is False

    def test_daily_fill_rates(self) -> None:
        from ztb.metrics.fill_quality import FillRecord, compute_fill_metrics

        records = []
        base_ts = 1700000000.0
        # Day 1: 10 orders, 9 filled
        for i in range(10):
            records.append(FillRecord(
                cycle_id=f"d1_{i}",
                timestamp=base_ts + i * 120,
                side="buy",
                order_price=100.0,
                order_quantity=0.001,
                filled=(i < 9),
                cancelled=(i >= 9),
            ))
        # Day 2: 10 orders, 5 filled
        for i in range(10):
            records.append(FillRecord(
                cycle_id=f"d2_{i}",
                timestamp=base_ts + 86400 + i * 120,
                side="buy",
                order_price=100.0,
                order_quantity=0.001,
                filled=(i < 5),
                cancelled=(i >= 5),
            ))
        m = compute_fill_metrics(records)
        assert m.measurement_days == 2
        assert len(m.daily_fill_rates) == 2
        assert m.daily_fill_rates[0] == pytest.approx(0.9)
        assert m.daily_fill_rates[1] == pytest.approx(0.5)
        # P90 (10th percentile of sorted daily rates) = 0.5 + 0.1*(0.9-0.5) = 0.54
        assert m.fill_rate_p90 == pytest.approx(0.54)


# =====================================================================
# g1_1_judgment
# =====================================================================

class TestG11Judgment:
    """G1.1 Gate 合否判定テスト."""

    def test_all_pass(self) -> None:
        from ztb.metrics.fill_quality import FillMetrics, g1_1_judgment

        metrics = FillMetrics(
            total_orders=100,
            filled_orders=95,
            cancelled_orders=5,
            fill_rate_p90=0.92,
            cancel_ratio=0.05,
            queue_wait_median_sec=15.0,
            post_fill_30s_pnl_mean=0.5,
            post_fill_30s_pnl_pvalue=0.8,
            adverse_selection_ratio=0.10,
        )
        thresholds = {
            "min_fill_rate_p90": 0.90,
            "max_cancel_ratio": 0.30,
            "max_queue_wait_median_sec": 60,
            "min_post_fill_30s_pnl": 0.0,
            "max_adverse_selection_ratio": 0.20,
        }
        result = g1_1_judgment(metrics, thresholds)
        assert result["gate_result"] == "PASS"
        assert all(c["pass"] for c in result["checks"].values())

    def test_fill_rate_fail(self) -> None:
        from ztb.metrics.fill_quality import FillMetrics, g1_1_judgment

        metrics = FillMetrics(
            fill_rate_p90=0.50,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=1.0,
            adverse_selection_ratio=0.05,
        )
        thresholds = {"min_fill_rate_p90": 0.90}
        result = g1_1_judgment(metrics, thresholds)
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["E1_fill_rate_p90"]["pass"] is False

    def test_adverse_selection_fail(self) -> None:
        from ztb.metrics.fill_quality import FillMetrics, g1_1_judgment

        metrics = FillMetrics(
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=0.5,
            adverse_selection_ratio=0.35,
        )
        thresholds = {"max_adverse_selection_ratio": 0.20}
        result = g1_1_judgment(metrics, thresholds)
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["E5_adverse_selection"]["pass"] is False

    def test_pnl_negative_but_not_significant(self) -> None:
        """E4: mean < 0 だが p >= 0.05 → PASS (009# §2.4)."""
        from ztb.metrics.fill_quality import FillMetrics, g1_1_judgment

        metrics = FillMetrics(
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=-0.1,
            post_fill_30s_pnl_pvalue=0.3,  # Not significant
            adverse_selection_ratio=0.05,
        )
        thresholds = {"min_post_fill_30s_pnl": 0.0}
        result = g1_1_judgment(metrics, thresholds)
        assert result["checks"]["E4_post_fill_pnl"]["pass"] is True

    def test_pnl_negative_and_significant(self) -> None:
        """E4: mean < 0 かつ p < 0.05 → FAIL (systemic adverse selection)."""
        from ztb.metrics.fill_quality import FillMetrics, g1_1_judgment

        metrics = FillMetrics(
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=-2.0,
            post_fill_30s_pnl_pvalue=0.001,  # Highly significant
            adverse_selection_ratio=0.05,
        )
        thresholds = {"min_post_fill_30s_pnl": 0.0}
        result = g1_1_judgment(metrics, thresholds)
        assert result["checks"]["E4_post_fill_pnl"]["pass"] is False

    def test_queue_wait_fail(self) -> None:
        from ztb.metrics.fill_quality import FillMetrics, g1_1_judgment

        metrics = FillMetrics(
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=120.0,
            post_fill_30s_pnl_mean=1.0,
            adverse_selection_ratio=0.05,
        )
        thresholds = {"max_queue_wait_median_sec": 60}
        result = g1_1_judgment(metrics, thresholds)
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["E3_queue_wait_median"]["pass"] is False

    def test_judgment_type_provisional(self) -> None:
        """020# O1: sample_sufficient=False → judgment_type=PROVISIONAL."""
        from ztb.metrics.fill_quality import FillMetrics, g1_1_judgment

        metrics = FillMetrics(
            total_orders=95,
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=0.5,
            adverse_selection_ratio=0.05,
            sample_sufficient=False,
        )
        thresholds = {
            "min_fill_rate_p90": 0.90,
            "max_cancel_ratio": 0.30,
            "max_queue_wait_median_sec": 60,
            "min_post_fill_30s_pnl": 0.0,
            "max_adverse_selection_ratio": 0.20,
        }
        result = g1_1_judgment(metrics, thresholds)
        assert result["judgment_type"] == "PROVISIONAL"
        assert result["sample_sufficient"] is False

    def test_judgment_type_final(self) -> None:
        """020# O1: sample_sufficient=True → judgment_type=FINAL."""
        from ztb.metrics.fill_quality import FillMetrics, g1_1_judgment

        metrics = FillMetrics(
            total_orders=250,
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=0.5,
            adverse_selection_ratio=0.05,
            sample_sufficient=True,
        )
        thresholds = {
            "min_fill_rate_p90": 0.90,
            "max_cancel_ratio": 0.30,
            "max_queue_wait_median_sec": 60,
            "min_post_fill_30s_pnl": 0.0,
            "max_adverse_selection_ratio": 0.20,
        }
        result = g1_1_judgment(metrics, thresholds)
        assert result["judgment_type"] == "FINAL"
        assert result["sample_sufficient"] is True

    def test_e5_raw_informational(self) -> None:
        """020# O5: E5-raw チェックは informational で gate に影響しない."""
        from ztb.metrics.fill_quality import FillMetrics, g1_1_judgment

        metrics = FillMetrics(
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=10.0,
            post_fill_30s_pnl_mean=0.5,
            adverse_selection_ratio=0.05,
            adverse_selection_ratio_raw=0.85,  # 超過しているが informational
            sample_sufficient=True,
        )
        thresholds = {"max_adverse_selection_ratio": 0.20}
        result = g1_1_judgment(metrics, thresholds)
        # E5 (deadzone) は PASS
        assert result["checks"]["E5_adverse_selection"]["pass"] is True
        # E5-raw は informational (gate に影響しない)
        assert result["checks"]["E5_adverse_selection_raw"]["informational"] is True
        # gate は PASS のまま
        assert result["gate_result"] == "PASS"


# =====================================================================
# I/O
# =====================================================================

class TestFillRecordIO:
    """FillRecord の JSONL I/O テスト."""

    def test_save_load_roundtrip(self) -> None:
        from ztb.metrics.fill_quality import FillRecord, load_fill_records, save_fill_records

        records = [
            FillRecord(
                cycle_id=f"io_{i}",
                timestamp=1700000000.0 + i * 120,
                side="buy" if i % 2 == 0 else "sell",
                order_price=15000000.0,
                order_quantity=0.001,
                filled=True,
            )
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            save_fill_records(records, path)
            loaded = load_fill_records(path)
            assert len(loaded) == 5
            assert loaded[0].cycle_id == "io_0"
            assert loaded[4].side == "buy"

    def test_load_nonexistent(self) -> None:
        from ztb.metrics.fill_quality import load_fill_records

        records = load_fill_records("/nonexistent/path.jsonl")
        assert records == []

    def test_glob_load(self) -> None:
        from ztb.metrics.fill_quality import (
            FillRecord,
            load_fill_records_glob,
            save_fill_records,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            for day in ["20260101", "20260102"]:
                records = [
                    FillRecord(
                        cycle_id=f"{day}_{i}",
                        timestamp=1700000000.0 + i,
                        side="buy",
                        order_price=100.0,
                        order_quantity=0.001,
                    )
                    for i in range(3)
                ]
                save_fill_records(records, Path(tmpdir) / f"fill_records_{day}.jsonl")

            all_records = load_fill_records_glob(tmpdir)
            assert len(all_records) == 6


# =====================================================================
# run_gate_check.py G1.1 integration
# =====================================================================

class TestGateCheckG11:
    """run_gate_check.py の G1.1 対応テスト."""

    def test_g1_1_no_data(self) -> None:
        from scripts.v460.run_gate_check import run_g1_1

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_g1_1(tmpdir)
            assert result["gate_result"] == "NO_DATA"

    def test_g1_1_with_data(self) -> None:
        from ztb.metrics.fill_quality import FillRecord, save_fill_records
        from scripts.v460.run_gate_check import run_g1_1

        with tempfile.TemporaryDirectory() as tmpdir:
            # 高 fill rate のデータを作成
            records = []
            base_ts = 1700000000.0
            for day in range(3):
                for i in range(100):
                    records.append(FillRecord(
                        cycle_id=f"d{day}_{i}",
                        timestamp=base_ts + day * 86400 + i * 120,
                        side="buy" if i % 2 == 0 else "sell",
                        order_price=15000000.0,
                        order_quantity=0.001,
                        filled=(i < 95),  # 95% fill rate
                        cancelled=(i >= 95),
                        queue_wait_sec=15.0 if i < 95 else 0.0,
                        post_fill_30s_pnl=0.5 if i < 95 else None,
                        adverse_selected=False if i < 95 else None,
                    ))
            save_fill_records(
                records,
                Path(tmpdir) / "fill_records_20260101.jsonl",
            )
            result = run_g1_1(tmpdir)
            assert result["gate"] == "G1.1-exec"
            # fill_rate_p90 = 95% → E1 PASS
            assert result["checks"]["E1_fill_rate_p90"]["pass"] is True


# =====================================================================
# Trade dedup (F8)
# =====================================================================

class TestTradeDedupTuple:
    """F8: trade dedup がタプル比較で正しく動作するテスト."""

    def test_dedup_skips_older(self) -> None:
        from ztb.data.market_data_collector import MarketDataCollector
        from ztb.trading.live.exchanges.base.broker_interfaces import TradeRecord

        collector = MarketDataCollector.__new__(MarketDataCollector)
        collector._tr_buffer = []
        collector._last_trade_id = None

        trades_batch1 = [
            TradeRecord(timestamp=100.0, price=50000.0, amount=0.1, side="buy"),
            TradeRecord(timestamp=200.0, price=50001.0, amount=0.2, side="sell"),
        ]
        collector._append_raw_trades(trades_batch1)
        assert len(collector._tr_buffer) == 2

        # Same batch again → should be deduped
        collector._append_raw_trades(trades_batch1)
        assert len(collector._tr_buffer) == 2  # No duplicates

    def test_dedup_numeric_comparison(self) -> None:
        """F8 の本質: 文字列比較だと "10.5" < "9.5" になるバグの修正確認."""
        from ztb.data.market_data_collector import MarketDataCollector
        from ztb.trading.live.exchanges.base.broker_interfaces import TradeRecord

        collector = MarketDataCollector.__new__(MarketDataCollector)
        collector._tr_buffer = []
        collector._last_trade_id = None

        # timestamp=9.5 を先に登録
        batch1 = [TradeRecord(timestamp=9.5, price=100.0, amount=0.1, side="buy")]
        collector._append_raw_trades(batch1)
        assert len(collector._tr_buffer) == 1

        # timestamp=10.5 → 新しいので追加されるべき
        batch2 = [TradeRecord(timestamp=10.5, price=100.0, amount=0.1, side="buy")]
        collector._append_raw_trades(batch2)
        assert len(collector._tr_buffer) == 2  # Must be 2 (not deduped)


# =====================================================================
# CoincheckAdapter real mode stubs
# =====================================================================

class TestAdapterRealModeP20:
    """P2-0: CoincheckAdapter real mode メソッドが NotImplementedError を出さないテスト."""

    def test_get_current_price_has_no_not_implemented(self) -> None:
        """get_current_price がreal modeで NotImplementedError を投げないことの静的確認."""
        import inspect
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

        source = inspect.getsource(CoincheckAdapter.get_current_price)
        # 009# P2-0: real mode should NOT raise NotImplementedError
        assert "NotImplementedError" not in source

    def test_get_order_status_has_no_not_implemented(self) -> None:
        import inspect
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

        source = inspect.getsource(CoincheckAdapter.get_order_status)
        assert "NotImplementedError" not in source

    def test_get_open_orders_has_no_not_implemented(self) -> None:
        import inspect
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

        source = inspect.getsource(CoincheckAdapter.get_open_orders)
        assert "NotImplementedError" not in source

    def test_get_positions_has_no_not_implemented(self) -> None:
        import inspect
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

        source = inspect.getsource(CoincheckAdapter.get_positions)
        assert "NotImplementedError" not in source


# =====================================================================
# 024# R1-R4: FillTestRunner 保存耐障害性テスト
# =====================================================================

class TestFillTestRunnerSaveResilience:
    """024# R1-R4: _try_save_batch / _save_batch_by_date / _emergency_dump のテスト."""

    def _make_runner(self, tmp_path: Path) -> "FillTestRunner":
        """テスト用の FillTestRunner を作成 (adapter は mock)."""
        from unittest.mock import MagicMock as Mock
        from scripts.v460.run_fill_test import FillTestRunner, FillTestConfig

        adapter = Mock()
        config = FillTestConfig(results_dir=str(tmp_path / "results"))
        runner = FillTestRunner(adapter, config)
        return runner

    def _make_record(self, ts: float = 1700000000.0, side: str = "buy") -> "FillRecord":
        from ztb.metrics.fill_quality import FillRecord
        return FillRecord(
            cycle_id=f"test_{int(ts)}",
            timestamp=ts,
            side=side,
            order_price=15000000.0,
            order_quantity=0.001,
            filled=True,
            fill_price=15000000.0,
        )

    def test_try_save_batch_success(self, tmp_path: Path) -> None:
        """正常保存時に True を返し、ファイルが作成される."""
        runner = self._make_runner(tmp_path)
        batch = [self._make_record(ts=1700000000.0 + i) for i in range(5)]

        result = runner._try_save_batch(batch)

        assert result is True
        assert runner._save_fail_count == 0
        # JSONL ファイルが作成されている
        jsonl_files = list((tmp_path / "results").glob("fill_records_*.jsonl"))
        assert len(jsonl_files) >= 1

    def test_try_save_batch_retry_on_failure(self, tmp_path: Path) -> None:
        """保存失敗時にリトライし、最終的に失敗を返す."""
        from unittest.mock import patch
        runner = self._make_runner(tmp_path)
        batch = [self._make_record()]

        with patch.object(runner, "_save_batch_by_date", side_effect=IOError("disk full")):
            result = runner._try_save_batch(batch)

        assert result is False
        assert runner._save_fail_count == 1
        assert len(runner._unsaved_batch) == 1

    def test_try_save_batch_emergency_dump_after_3_failures(self, tmp_path: Path) -> None:
        """3回連続失敗で緊急ダンプが発動する."""
        from unittest.mock import patch
        runner = self._make_runner(tmp_path)
        runner._save_fail_count = 2  # 既に2回失敗
        batch = [self._make_record()]

        with patch.object(runner, "_save_batch_by_date", side_effect=IOError("disk full")):
            result = runner._try_save_batch(batch)

        # 緊急ダンプが発動 → True を返す
        assert result is True
        assert runner._save_fail_count == 0
        # emergency ディレクトリにファイルが作成
        emergency_files = list((tmp_path / "results" / "emergency").glob("emergency_*.jsonl"))
        assert len(emergency_files) >= 1

    def test_save_batch_by_date_groups_by_utc_date(self, tmp_path: Path) -> None:
        """024# R4: record.timestamp 由来で日付別ファイル分割."""
        runner = self._make_runner(tmp_path)

        # 2つの異なる UTC 日付のレコード
        # 2023-11-14 23:59 UTC と 2023-11-15 00:01 UTC
        batch = [
            self._make_record(ts=1700006340.0),  # 2023-11-14 23:59 UTC
            self._make_record(ts=1700006460.0),  # 2023-11-15 00:01 UTC
        ]

        runner._save_batch_by_date(batch)

        results_dir = tmp_path / "results"
        f1 = results_dir / "fill_records_20231114.jsonl"
        f2 = results_dir / "fill_records_20231115.jsonl"
        assert f1.exists(), f"Expected {f1}"
        assert f2.exists(), f"Expected {f2}"

    def test_emergency_dump_creates_file(self, tmp_path: Path) -> None:
        """緊急ダンプがファイルを作成する."""
        runner = self._make_runner(tmp_path)
        batch = [self._make_record()]

        runner._emergency_dump(batch, "test_reason")

        emergency_files = list((tmp_path / "results" / "emergency").glob("emergency_test_reason_*.jsonl"))
        assert len(emergency_files) == 1

    def test_cleanup_sync_saves_unsaved_batch(self, tmp_path: Path) -> None:
        """atexit で未保存バッチが退避される."""
        runner = self._make_runner(tmp_path)
        runner._unsaved_batch = [self._make_record()]

        runner._cleanup_sync()

        assert runner._unsaved_batch == []
        emergency_files = list((tmp_path / "results" / "emergency").glob("emergency_atexit_*.jsonl"))
        assert len(emergency_files) == 1

    def test_cleanup_sync_no_unsaved_no_dump(self, tmp_path: Path) -> None:
        """未保存バッチが空なら緊急ダンプは作成されない."""
        runner = self._make_runner(tmp_path)

        runner._cleanup_sync()

        emergency_dir = tmp_path / "results" / "emergency"
        if emergency_dir.exists():
            assert len(list(emergency_dir.glob("*.jsonl"))) == 0


class TestUnknownFillHandling:
    """025# F6: status_order is None 時に filled 扱いしない安全策のテスト."""

    def _make_runner(self, tmp_path: Path) -> "FillTestRunner":
        """テスト用の FillTestRunner を作成 (adapter は AsyncMock)."""
        from unittest.mock import AsyncMock, MagicMock
        from scripts.v460.run_fill_test import FillTestRunner, FillTestConfig

        adapter = AsyncMock()
        # get_orderbook の戻り値を設定
        ob_mock = MagicMock()
        ob_mock.bids = [(15000000.0, 0.1)]
        ob_mock.asks = [(15001000.0, 0.1)]
        adapter.get_orderbook.return_value = ob_mock
        # place_order の戻り値
        order_mock = MagicMock()
        order_mock.order_id = "test_order_123"
        adapter.place_order.return_value = order_mock

        config = FillTestConfig(
            results_dir=str(tmp_path / "results"),
            order_timeout_sec=10.0,
            poll_interval_sec=0.01,
            post_fill_wait_sec=0.01,
        )
        runner = FillTestRunner(adapter, config)
        return runner

    @pytest.mark.asyncio
    async def test_status_none_twice_becomes_cancelled_status_unknown(
        self, tmp_path: Path,
    ) -> None:
        """get_order_status が 2 回 None → cancelled, cancel_reason=status_unknown."""
        runner = self._make_runner(tmp_path)
        # 初回も retry も None
        runner.adapter.get_order_status.return_value = None

        record = await runner.run_single_cycle()

        assert record.filled is False
        assert record.cancelled is True
        assert record.cancel_reason == "status_unknown"

    @pytest.mark.asyncio
    async def test_status_none_then_filled_on_retry(
        self, tmp_path: Path,
    ) -> None:
        """get_order_status が None → retry で filled → filled=True."""
        from unittest.mock import MagicMock

        runner = self._make_runner(tmp_path)
        filled_order = MagicMock()
        filled_order.status = "filled"
        filled_order.price = 15000500.0
        # 1 回目 None, 2 回目 filled
        runner.adapter.get_order_status.side_effect = [None, filled_order]

        record = await runner.run_single_cycle()

        assert record.filled is True
        assert record.fill_price == 15000500.0
        assert record.cancel_reason is None

    @pytest.mark.asyncio
    async def test_status_filled_directly(
        self, tmp_path: Path,
    ) -> None:
        """get_order_status が直接 filled → 通常の filled 処理."""
        from unittest.mock import MagicMock

        runner = self._make_runner(tmp_path)
        filled_order = MagicMock()
        filled_order.status = "filled"
        filled_order.price = 15000200.0
        runner.adapter.get_order_status.return_value = filled_order

        record = await runner.run_single_cycle()

        assert record.filled is True
        assert record.fill_price == 15000200.0
        assert record.cancel_reason is None
