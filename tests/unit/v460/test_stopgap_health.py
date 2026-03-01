"""165# Unit tests for stopgap_health module.

162# P1: 日次 per-regime 3指標 + Stopgap 退出判定の単体テスト.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pytest

from scripts.v460.lib.stopgap_health import (
    AlertItem,
    DailyHealthReport,
    DailyMetrics,
    ExitVerdict,
    ModelUsedMetrics,
    StopgapExitCheck,
    apply_filters,
    compute_daily_metrics,
    compute_model_used_metrics,
    evaluate_stopgap_exit,
    generate_alerts,
    generate_health_report,
    load_fill_records,
    _filter_window,
    _get_day,
    _check_2a_trending_sell_skip,
    _check_2c_sell_dynamic_kill,
    _check_2d_sell_guard,
    _check_6a_unknown_regime_skip,
)


# ======================================================================
# Helpers
# ======================================================================


def _make_record(
    *,
    filled: bool = True,
    side: str = "buy",
    regime: str = "ranging",
    pnl30: float | None = 1.0,
    adverse_selected: bool = False,
    cancel_reason: str | None = None,
    timestamp: float | None = None,
    cancelled: bool = False,
) -> dict:
    """テスト用 fill record 生成."""
    ts = timestamp or datetime(2026, 2, 24, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    r: dict = {
        "filled": filled,
        "cancelled": cancelled,
        "side": side,
        "regime": regime,
        "post_fill_30s_pnl": pnl30 if filled else None,
        "adverse_selected": adverse_selected,
        "cancel_reason": cancel_reason,
        "timestamp": ts,
    }
    return r


def _records_batch(
    n: int,
    *,
    side: str = "buy",
    regime: str = "ranging",
    pnl30: float = 1.0,
    as_rate: float = 0.0,
    fill_rate: float = 1.0,
) -> list[dict]:
    """バッチ生成."""
    records = []
    n_filled = int(n * fill_rate)
    n_as = int(n_filled * as_rate)
    base_ts = datetime(2026, 2, 24, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    for i in range(n):
        filled = i < n_filled
        is_as = filled and i < n_as
        records.append(_make_record(
            filled=filled,
            cancelled=not filled,
            side=side,
            regime=regime,
            pnl30=pnl30 if filled else None,
            adverse_selected=is_as,
            timestamp=base_ts + i * 60,
        ))
    return records


# ======================================================================
# ExitVerdict enum
# ======================================================================


class TestExitVerdict:
    def test_values(self):
        assert ExitVerdict.CAN_EXIT.value == "can_exit"
        assert ExitVerdict.KEEP.value == "keep"
        assert ExitVerdict.INSUFFICIENT.value == "insufficient"


# ======================================================================
# DailyMetrics
# ======================================================================


class TestDailyMetrics:
    def test_defaults(self):
        m = DailyMetrics(day="20260224", regime="all", side="all")
        assert m.n_total == 0
        assert m.n_filled == 0
        assert m.as_rate == 0.0
        assert math.isnan(m.avg_pnl30_bps)

    def test_full_init(self):
        m = DailyMetrics(
            day="20260224", regime="ranging", side="sell",
            n_total=100, n_filled=50, fill_rate=0.5,
            avg_pnl30_bps=-1.5, downside_p10_bps=-5.0,
            as_rate=0.25,
        )
        assert m.fill_rate == 0.5
        assert m.as_rate == 0.25


# ======================================================================
# _get_day  
# ======================================================================


class TestGetDay:
    def test_valid_timestamp(self):
        ts = datetime(2026, 2, 24, 15, 30, tzinfo=timezone.utc).timestamp()
        assert _get_day({"timestamp": ts}) == "20260224"

    def test_none_timestamp(self):
        assert _get_day({"timestamp": None}) == "unknown"
        assert _get_day({}) == "unknown"


# ======================================================================
# _filter_window
# ======================================================================


class TestFilterWindow:
    def test_zero_window(self):
        recs = [_make_record()]
        assert len(_filter_window(recs, 0)) == 1

    def test_old_records_filtered(self):
        old_ts = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        recs = [_make_record(timestamp=old_ts)]
        assert len(_filter_window(recs, 24)) == 0


# ======================================================================
# compute_daily_metrics
# ======================================================================


class TestComputeDailyMetrics:
    def test_empty(self):
        assert compute_daily_metrics([]) == []

    def test_single_record(self):
        recs = [_make_record(side="buy", regime="ranging")]
        results = compute_daily_metrics(recs)
        # (day, all, all) + (day, ranging, all) + (day, all, buy) + (day, ranging, buy)
        assert len(results) == 4
        # Check all/all
        all_all = [r for r in results if r.regime == "all" and r.side == "all"]
        assert len(all_all) == 1
        assert all_all[0].n_filled == 1
        assert all_all[0].fill_rate == 1.0

    def test_multi_regime_side(self):
        recs = [
            _make_record(side="buy", regime="ranging"),
            _make_record(side="sell", regime="trending"),
        ]
        results = compute_daily_metrics(recs)
        regimes = {r.regime for r in results}
        sides = {r.side for r in results}
        assert "ranging" in regimes
        assert "trending" in regimes
        assert "buy" in sides
        assert "sell" in sides

    def test_as_rate(self):
        recs = _records_batch(10, as_rate=0.3)
        results = compute_daily_metrics(recs)
        all_all = [r for r in results if r.regime == "all" and r.side == "all"]
        assert all_all[0].as_rate == pytest.approx(0.3, abs=0.01)

    def test_dynamic_kill_count(self):
        recs = [
            _make_record(filled=False, cancelled=True, cancel_reason="sell_dynamic_kill"),
            _make_record(filled=True),
        ]
        results = compute_daily_metrics(recs)
        all_all = [r for r in results if r.regime == "all" and r.side == "all"]
        assert all_all[0].dynamic_kill_count == 1

    def test_velocity_skip_count(self):
        recs = [
            _make_record(filled=False, cancelled=True,
                        cancel_reason="skip_gate_rule_velocity_sell"),
            _make_record(filled=True),
        ]
        results = compute_daily_metrics(recs)
        all_all = [r for r in results if r.regime == "all" and r.side == "all"]
        assert all_all[0].velocity_skip_count == 1


# ======================================================================
# 2-A: trending_sell_skip
# ======================================================================


class TestCheck2ATrendingSellSkip:
    def test_insufficient(self):
        recs = _records_batch(5, side="sell", regime="trending")
        result = _check_2a_trending_sell_skip(recs)
        assert result.verdict == ExitVerdict.INSUFFICIENT

    def test_can_exit(self):
        """AS_rate < 35%, total PnL > 0."""
        recs = _records_batch(30, side="sell", regime="trending",
                              pnl30=2.0, as_rate=0.2)
        result = _check_2a_trending_sell_skip(recs)
        assert result.verdict == ExitVerdict.CAN_EXIT
        assert result.metrics["as_rate"] < 0.35

    def test_keep_high_as(self):
        """AS_rate >= 35%."""
        recs = _records_batch(30, side="sell", regime="trending",
                              pnl30=2.0, as_rate=0.5)
        result = _check_2a_trending_sell_skip(recs)
        assert result.verdict == ExitVerdict.KEEP

    def test_keep_negative_pnl(self):
        """total PnL <= 0."""
        recs = _records_batch(30, side="sell", regime="trending",
                              pnl30=-2.0, as_rate=0.1)
        result = _check_2a_trending_sell_skip(recs)
        assert result.verdict == ExitVerdict.KEEP

    def test_rollback_flag(self):
        """AS_rate > 50% triggers rollback."""
        recs = _records_batch(30, side="sell", regime="trending",
                              pnl30=-1.0, as_rate=0.6)
        result = _check_2a_trending_sell_skip(recs)
        assert result.metrics["rollback_triggered"] is True


# ======================================================================
# 2-C: sell_dynamic_kill
# ======================================================================


class TestCheck2CSellDynamicKill:
    def test_can_exit_no_kills(self):
        recs = _records_batch(100)
        result = _check_2c_sell_dynamic_kill(recs, window_days=7.0)
        assert result.verdict == ExitVerdict.CAN_EXIT
        assert result.metrics["kills_per_day"] == 0.0

    def test_keep_frequent_kills(self):
        recs = [_make_record(filled=False, cancelled=True,
                            cancel_reason="sell_dynamic_kill")
                for _ in range(20)]
        result = _check_2c_sell_dynamic_kill(recs, window_days=7.0)
        assert result.verdict == ExitVerdict.KEEP
        assert result.metrics["kills_per_day"] > 1.0

    def test_rollback(self):
        recs = [_make_record(filled=False, cancelled=True,
                            cancel_reason="sell_dynamic_kill")
                for _ in range(30)]
        result = _check_2c_sell_dynamic_kill(recs, window_days=7.0)
        assert result.metrics["rollback_triggered"] is True


# ======================================================================
# 2-D: sell_guard
# ======================================================================


class TestCheck2DSellGuard:
    def test_insufficient(self):
        recs = _records_batch(5, side="sell")
        result = _check_2d_sell_guard(recs)
        assert result.verdict == ExitVerdict.INSUFFICIENT

    def test_can_exit(self):
        recs = _records_batch(50, side="sell", pnl30=1.0, fill_rate=0.95)
        result = _check_2d_sell_guard(recs)
        assert result.verdict == ExitVerdict.CAN_EXIT

    def test_keep_high_cancel(self):
        recs = _records_batch(50, side="sell", pnl30=1.0, fill_rate=0.3)
        result = _check_2d_sell_guard(recs)
        assert result.verdict == ExitVerdict.KEEP


# ======================================================================
# 6-A: unknown_regime_skip
# ======================================================================


class TestCheck6AUnknownRegimeSkip:
    def test_insufficient(self):
        recs = _records_batch(10, regime="unknown")
        result = _check_6a_unknown_regime_skip(recs)
        assert result.verdict == ExitVerdict.INSUFFICIENT

    def test_can_exit(self):
        recs = _records_batch(100, regime="ranging")
        result = _check_6a_unknown_regime_skip(recs)
        assert result.verdict == ExitVerdict.CAN_EXIT

    def test_keep(self):
        recs = _records_batch(100, regime="unknown")
        result = _check_6a_unknown_regime_skip(recs)
        assert result.verdict == ExitVerdict.KEEP
        assert result.metrics["unknown_rate"] == 1.0


# ======================================================================
# evaluate_stopgap_exit
# ======================================================================


class TestEvaluateStopgapExit:
    def test_returns_4_checks(self):
        recs = _records_batch(100, side="sell", regime="trending")
        checks = evaluate_stopgap_exit(recs, window_hours=0)
        assert len(checks) == 4
        ids = {c.stopgap_id for c in checks}
        assert ids == {"2-A", "2-C", "2-D", "6-A"}


# ======================================================================
# generate_health_report
# ======================================================================


class TestGenerateHealthReport:
    def test_basic_report(self):
        recs = _records_batch(50, side="buy", regime="ranging")
        report = generate_health_report(recs, window_hours=0, daily_limit=7)
        assert isinstance(report, DailyHealthReport)
        assert report.total_records == 50
        assert report.total_filled > 0
        assert len(report.daily_metrics) > 0
        assert len(report.stopgap_checks) == 4

    def test_empty_records(self):
        report = generate_health_report([], window_hours=0)
        assert report.total_records == 0
        assert report.daily_metrics == []

    def test_daily_limit(self):
        """Daily limit caps output."""
        recs = _records_batch(50, side="buy", regime="ranging")
        report = generate_health_report(recs, window_hours=0, daily_limit=1)
        # All records are same day, so limit doesn't drop any
        assert len(report.daily_metrics) > 0

    def test_report_serializable(self):
        """Report can be serialized to JSON."""
        recs = _records_batch(50, side="buy", regime="ranging")
        report = generate_health_report(recs, window_hours=0)
        from dataclasses import asdict
        data = asdict(report)
        json_str = json.dumps(data, default=str)
        assert "daily_metrics" in json_str
        assert "stopgap_checks" in json_str

    def test_unknown_regime_count_serialized(self):
        """集計済みの unknown_regime_count が report に含まれる."""
        recs = [
            _make_record(regime="unknown"),
            _make_record(regime="ranging"),
        ]
        report = generate_health_report(recs, window_hours=0)
        all_all = [
            d for d in report.daily_metrics
            if d["regime"] == "all" and d["side"] == "all"
        ]
        assert all_all[0]["unknown_regime_count"] == 1


# ======================================================================
# apply_filters (165# 7.5 P0 再現性固定)
# ======================================================================


class TestApplyFilters:
    def test_no_filter(self):
        recs = _records_batch(10)
        r, f = apply_filters(recs)
        assert len(r) == 10
        assert all(v is None for v in f.values())

    def test_run_id_filter(self):
        recs = [
            {**_make_record(), "run_id": "abc"},
            {**_make_record(), "run_id": "def"},
            {**_make_record(), "run_id": "abc"},
        ]
        r, f = apply_filters(recs, run_id="abc")
        assert len(r) == 2
        assert f["run_id"] == "abc"

    def test_git_sha_prefix(self):
        recs = [
            {**_make_record(), "git_sha": "abc123def"},
            {**_make_record(), "git_sha": "abc999fff"},
            {**_make_record(), "git_sha": "def000000"},
        ]
        r, f = apply_filters(recs, git_sha="abc")
        assert len(r) == 2
        assert f["git_sha"] == "abc"

    def test_date_from(self):
        ts1 = datetime(2026, 2, 20, 0, 0, tzinfo=timezone.utc).timestamp()
        ts2 = datetime(2026, 2, 22, 0, 0, tzinfo=timezone.utc).timestamp()
        recs = [
            {**_make_record(), "timestamp": ts1},
            {**_make_record(), "timestamp": ts2},
        ]
        r, _ = apply_filters(recs, date_from="2026-02-21")
        assert len(r) == 1

    def test_date_to_inclusive(self):
        ts1 = datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc).timestamp()
        ts2 = datetime(2026, 2, 21, 12, 0, tzinfo=timezone.utc).timestamp()
        recs = [
            {**_make_record(), "timestamp": ts1},
            {**_make_record(), "timestamp": ts2},
        ]
        r, _ = apply_filters(recs, date_to="2026-02-20")
        assert len(r) == 1  # ts1 is within 2/20 (inclusive end = +86400s)

    def test_combined(self):
        ts1 = datetime(2026, 2, 20, 0, 0, tzinfo=timezone.utc).timestamp()
        recs = [
            {**_make_record(), "run_id": "r1", "git_sha": "abc", "timestamp": ts1},
            {**_make_record(), "run_id": "r2", "git_sha": "abc", "timestamp": ts1},
            {**_make_record(), "run_id": "r1", "git_sha": "def", "timestamp": ts1},
        ]
        r, _ = apply_filters(recs, run_id="r1", git_sha="abc")
        assert len(r) == 1

    def test_invalid_timestamp_does_not_raise(self):
        recs = [
            {**_make_record(), "timestamp": "bad"},
            {**_make_record(), "timestamp": datetime(2026, 2, 22, 0, 0, tzinfo=timezone.utc).timestamp()},
        ]
        r, _ = apply_filters(recs, date_from="2026-02-21")
        assert len(r) == 1


# ======================================================================
# load_fill_records
# ======================================================================


class TestLoadFillRecords:
    def test_loader_handles_bom_and_ignores_non_object_lines(self, tmp_path):
        path = tmp_path / "fill_records_test.jsonl"
        path.write_text(
            "\n".join([
                "\ufeff{\"filled\": true, \"timestamp\": 1}",
                "[]",
                "42",
                "\"text\"",
                "{invalid",
            ]),
            encoding="utf-8",
        )

        records = load_fill_records(tmp_path)

        assert len(records) == 1
        assert records[0]["filled"] is True

    def test_loader_deduplicates_cycle_id_across_files(self, tmp_path):
        (tmp_path / "fill_records_20260224.jsonl").write_text(
            json.dumps({"cycle_id": "dup_1", "filled": True, "timestamp": 1}) + "\n",
            encoding="utf-8",
        )
        (tmp_path / "fill_records_20260225.jsonl").write_text(
            "\n".join([
                json.dumps({"cycle_id": "dup_1", "filled": False, "timestamp": 2}),
                json.dumps({"cycle_id": "dup_2", "filled": True, "timestamp": 3}),
            ]) + "\n",
            encoding="utf-8",
        )

        records = load_fill_records(tmp_path)

        assert [record["cycle_id"] for record in records] == ["dup_1", "dup_2"]


# ======================================================================
# ModelUsedMetrics / compute_model_used_metrics (165# 7.3)
# ======================================================================


class TestComputeModelUsedMetrics:
    def test_empty(self):
        result = compute_model_used_metrics([])
        assert result == []

    def test_single_model(self):
        recs = [
            {**_make_record(pnl30=2.0), "skip_gate_model_used": "primary:side_sell"},
            {**_make_record(pnl30=-3.0, adverse_selected=True), "skip_gate_model_used": "primary:side_sell"},
        ]
        result = compute_model_used_metrics(recs)
        assert len(result) == 1
        m = result[0]
        assert m.model_used == "primary:side_sell"
        assert m.n_filled == 2
        assert m.as_count == 1
        assert m.as_rate == 0.5

    def test_multiple_models(self):
        recs = [
            {**_make_record(pnl30=1.0), "skip_gate_model_used": "none"},
            {**_make_record(pnl30=2.0), "skip_gate_model_used": "primary:side_buy"},
            {**_make_record(pnl30=-1.0, adverse_selected=True), "skip_gate_model_used": "primary:unified"},
        ]
        result = compute_model_used_metrics(recs)
        assert len(result) == 3
        models = [m.model_used for m in result]
        assert "none" in models
        assert "primary:side_buy" in models
        assert "primary:unified" in models

    def test_unfilled_excluded(self):
        recs = [
            {**_make_record(filled=False), "skip_gate_model_used": "none"},
            {**_make_record(pnl30=1.0), "skip_gate_model_used": "none"},
        ]
        result = compute_model_used_metrics(recs)
        assert len(result) == 1
        assert result[0].n_filled == 1

    def test_none_model_default(self):
        recs = [_make_record(pnl30=1.0)]  # no skip_gate_model_used key
        result = compute_model_used_metrics(recs)
        assert result[0].model_used == "none"


# ======================================================================
# generate_alerts (165# 7.5 P0)
# ======================================================================


class TestGenerateAlerts:
    def test_keep_warning(self):
        checks = [StopgapExitCheck(
            stopgap_id="2-A",
            name="Test",
            verdict=ExitVerdict.KEEP,
            detail="keep detail",
            metrics={}
        )]
        alerts = generate_alerts(checks)
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"

    def test_can_exit_info(self):
        checks = [StopgapExitCheck(
            stopgap_id="2-C",
            name="Test",
            verdict=ExitVerdict.CAN_EXIT,
            detail="exit detail",
            metrics={}
        )]
        alerts = generate_alerts(checks)
        assert len(alerts) == 1
        assert alerts[0].severity == "info"

    def test_rollback_critical(self):
        checks = [StopgapExitCheck(
            stopgap_id="2-D",
            name="Test",
            verdict=ExitVerdict.KEEP,
            detail="rollback",
            metrics={"rollback_triggered": True}
        )]
        alerts = generate_alerts(checks)
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"

    def test_empty(self):
        alerts = generate_alerts([])
        assert alerts == []


# ======================================================================
# Report includes new fields (165# 7.3/7.5 integration)
# ======================================================================


class TestReportNewFields:
    def test_report_has_model_used(self):
        recs = [
            {**_make_record(pnl30=1.0), "skip_gate_model_used": "primary:side_sell"},
            {**_make_record(pnl30=-1.0, adverse_selected=True), "skip_gate_model_used": "none"},
        ]
        report = generate_health_report(recs, window_hours=0)
        assert len(report.model_used_breakdown) > 0
        mu_models = {m["model_used"] for m in report.model_used_breakdown}
        assert "primary:side_sell" in mu_models
        assert "none" in mu_models

    def test_report_has_alerts(self):
        recs = _records_batch(20, as_rate=0.5, pnl30=-5.0)
        report = generate_health_report(recs, window_hours=0)
        assert len(report.alerts) > 0
        severities = {a["severity"] for a in report.alerts}
        assert len(severities) > 0

    def test_report_has_filters(self):
        recs = _records_batch(10)
        report = generate_health_report(
            recs, window_hours=0,
            filters_applied={"run_id": "test123", "git_sha": None, "date_from": None, "date_to": None}
        )
        assert report.filters_applied["run_id"] == "test123"

    def test_report_json_includes_new_fields(self):
        recs = _records_batch(20)
        report = generate_health_report(recs, window_hours=0)
        from dataclasses import asdict
        data = asdict(report)
        s = json.dumps(data, default=str)
        assert "model_used_breakdown" in s
        assert "alerts" in s
        assert "filters_applied" in s
