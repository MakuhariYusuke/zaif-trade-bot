"""
gate_judgment.py 単体テスト — 122# B1 + E10 Monte Carlo 統合.

scripts/v460/gate_judgment.py の run_gate_judgment() コア関数を検証。
CLI (main) は subprocess テストではなく、関数レベルで検証する。
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scripts.v460.gate_judgment import _load_all_records, _side_metrics, run_gate_judgment

from ztb.metrics.fill_quality import FillRecord, save_fill_records


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_record(
    cycle_id: str = "test_001",
    filled: bool = True,
    pnl_bps: float | None = 0.5,
    adverse: bool = False,
    price: float = 10_300_000.0,
    wait: float = 5.0,
    side: str = "buy",
    ts: float = 1770975573.0,
    skip_gate: bool = False,
    cancel_reason: str | None = None,
) -> FillRecord:
    """テスト用 FillRecord ファクトリ."""
    return FillRecord(
        cycle_id=cycle_id,
        timestamp=ts,
        side=side,
        order_price=price,
        order_quantity=0.001,
        fill_price=price if filled else None,
        filled=filled,
        cancelled=not filled,
        queue_wait_sec=wait,
        mid_at_fill=price + 100 if filled else None,
        mid_30s_after=price + 200 if filled else None,
        post_fill_30s_pnl=pnl_bps if filled else None,
        adverse_selected=adverse if filled else None,
        cancel_reason=cancel_reason if not filled else None,
        skip_gate_skipped=skip_gate if skip_gate else None,
        git_sha="abcdef1234567890",
        run_id="test_run_001",
    )


def _make_records_mixed(
    n_filled: int = 50,
    n_cancelled: int = 10,
    days: int = 3,
) -> list[FillRecord]:
    """複数日にまたがる典型的なレコードセット."""
    recs: list[FillRecord] = []
    base_ts = 1770975573.0
    idx = 0
    pnl_values = [0.5, 1.0, -0.3, 0.8, -0.1, 0.3, 0.7, -0.5, 1.2, 0.2]
    for day in range(days):
        per_day_filled = n_filled // days
        per_day_cancel = n_cancelled // days
        for i in range(per_day_filled):
            pnl = pnl_values[idx % len(pnl_values)]
            recs.append(_make_record(
                cycle_id=f"fill_d{day}_{i:03d}",
                filled=True,
                pnl_bps=pnl,
                adverse=(pnl < 0),
                side="buy" if i % 2 == 0 else "sell",
                ts=base_ts + day * 86400 + i * 120,
                wait=3.0 + i * 0.5,
            ))
            idx += 1
        for i in range(per_day_cancel):
            recs.append(_make_record(
                cycle_id=f"cancel_d{day}_{i:03d}",
                filled=False,
                side="buy" if i % 2 == 0 else "sell",
                ts=base_ts + day * 86400 + (per_day_filled + i) * 120,
                cancel_reason="timeout",
            ))
    return recs


# ---------------------------------------------------------------------------
# Helper: default gate config
# ---------------------------------------------------------------------------

def _default_gate_cfg() -> dict:
    """テスト用デフォルト gate_thresholds."""
    return {
        "g1_1_quick_exec": {
            "attempted_fill_rate": 0.60,
            "queue_wait_median_sec": 120.0,
            "post_fill_30s_pnl_mean": -0.8,
            "skip_gate_ratio": 0.25,
        },
        "g1_2_full_exec": {
            "attempted_fill_rate": 0.70,
            "overall_fill_rate": 0.62,
            "queue_wait_median_sec": 60.0,
            "adverse_selection_ratio": 0.30,
            "skip_gate_ratio": 0.20,
            "calendar_coverage_days": 7,
            "n_attempted_min": 500,
            "pnl_mean_floor_bps": -1.0,  # 123# テスト用: 緩い floor
            "pnl_mean_hard_floor_bps": -5.0,
        },
    }


# =====================================================================
# Core: run_gate_judgment
# =====================================================================

class TestRunGateJudgment:
    """run_gate_judgment 関数のテスト."""

    def test_basic_result_structure(self) -> None:
        """基本的な result dict の構造を検証."""
        records = _make_records_mixed(n_filled=50, n_cancelled=10, days=3)
        result = run_gate_judgment(records, _default_gate_cfg())

        assert "data_summary" in result
        assert "metrics" in result
        assert "g1_1_quick" in result
        assert "g1_2_full" in result

    def test_data_summary_counts(self) -> None:
        """data_summary のレコード数が正確."""
        records = _make_records_mixed(n_filled=48, n_cancelled=9, days=3)
        result = run_gate_judgment(records, _default_gate_cfg())

        ds = result["data_summary"]
        assert ds["total_records"] == len(records)
        assert ds["clean_records"] + ds["quarantine_records"] == len(records)
        assert ds["elapsed_hours"] > 0

    def test_metrics_keys(self) -> None:
        """metrics に必要な全キーが含まれる."""
        records = _make_records_mixed(n_filled=50, n_cancelled=10)
        result = run_gate_judgment(records, _default_gate_cfg())

        expected_keys = {
            "attempted_fill_rate", "overall_fill_rate",
            "attempted_cancel_ratio", "queue_wait_median_sec",
            "pnl_30s_mean", "pnl_30s_pvalue",
            "pnl_60s_mean", "pnl_60s_pvalue",
            "pnl_120s_mean", "pnl_120s_pvalue",
            "pnl_ci_upper", "as_ratio", "skip_gate_ratio",
        }
        assert expected_keys.issubset(set(result["metrics"].keys()))

    def test_gate_judgment_has_gate_result(self) -> None:
        """G1.1/G1.2 に gate_result キーがある."""
        records = _make_records_mixed(n_filled=50, n_cancelled=10)
        result = run_gate_judgment(records, _default_gate_cfg())

        assert "gate_result" in result["g1_1_quick"]
        assert "gate_result" in result["g1_2_full"]
        # gate_result は PASS/FAIL/WATCH のいずれか
        assert result["g1_1_quick"]["gate_result"] in ("PASS", "FAIL", "WATCH")

    def test_side_breakdown(self) -> None:
        """side_breakdown=True で buy/sell 別メトリクスが含まれる."""
        records = _make_records_mixed(n_filled=50, n_cancelled=10)
        result = run_gate_judgment(
            records, _default_gate_cfg(), side_breakdown=True,
        )

        assert "side_breakdown" in result
        assert "buy" in result["side_breakdown"]
        assert "sell" in result["side_breakdown"]
        assert result["side_breakdown"]["buy"]["n"] > 0
        assert result["side_breakdown"]["sell"]["n"] > 0

    def test_no_side_breakdown_by_default(self) -> None:
        """side_breakdown=False (default) では含まれない."""
        records = _make_records_mixed(n_filled=50, n_cancelled=10)
        result = run_gate_judgment(records, _default_gate_cfg())

        assert "side_breakdown" not in result

    def test_json_serializable(self) -> None:
        """result は JSON シリアライズ可能."""
        records = _make_records_mixed(n_filled=50, n_cancelled=10)
        result = run_gate_judgment(
            records, _default_gate_cfg(), side_breakdown=True,
        )

        # json.dumps が例外を投げなければ OK
        serialized = json.dumps(result, ensure_ascii=False)
        assert len(serialized) > 100

    def test_elapsed_hours_zero_for_single_record(self) -> None:
        """レコード 1 件の場合 elapsed_hours=0."""
        records = [_make_record(cycle_id="single")]
        result = run_gate_judgment(records, _default_gate_cfg())

        assert result["data_summary"]["elapsed_hours"] == 0.0


# =====================================================================
# E10: Monte Carlo 統合テスト
# =====================================================================

class TestGateJudgmentMonteCarlo:
    """E10: Monte Carlo PnL シミュレーションの gate_judgment 統合."""

    def test_monte_carlo_disabled_by_default(self) -> None:
        """monte_carlo=False ではキーが含まれない."""
        records = _make_records_mixed(n_filled=50, n_cancelled=10)
        result = run_gate_judgment(records, _default_gate_cfg())

        assert "monte_carlo" not in result

    def test_monte_carlo_enabled(self) -> None:
        """monte_carlo=True で MC 結果が含まれる."""
        mc_simulations = 60
        records = _make_records_mixed(n_filled=50, n_cancelled=10)
        result = run_gate_judgment(
            records, _default_gate_cfg(),
            monte_carlo=True,
            mc_simulations=mc_simulations,
        )

        assert "monte_carlo" in result
        mc = result["monte_carlo"]
        assert "error" not in mc
        assert "pnl_mean_jpy" in mc
        assert "var_95_jpy" in mc
        assert "prob_loss" in mc
        assert "prob_profit" in mc
        assert mc["n_simulations"] == mc_simulations

    def test_monte_carlo_pnl_consistency(self) -> None:
        """MC の PnL mean が observed mean と整合性がある."""
        np.random.seed(42)  # 123# Gemini review: seed 固定で Flaky 防止

        # 正 PnL のデータ
        records = [
            _make_record(cycle_id=f"pos_{i}", pnl_bps=1.0, ts=1770975573.0 + i * 120)
            for i in range(30)
        ]
        result = run_gate_judgment(
            records, _default_gate_cfg(),
            monte_carlo=True,
            mc_simulations=120,
        )

        mc = result["monte_carlo"]
        # observed_pnl_mean_bps should be close to 1.0
        assert mc["observed_pnl_mean_bps"] == pytest.approx(1.0, abs=0.01)
        # monthly PnL should be positive
        assert mc["pnl_mean_jpy"] > 0

    def test_monte_carlo_risk_metrics(self) -> None:
        """MC の risk metrics が妥当な範囲."""
        np.random.seed(77)  # 123# Gemini review: seed 固定で Flaky 防止

        records = _make_records_mixed(n_filled=50, n_cancelled=10)
        result = run_gate_judgment(
            records, _default_gate_cfg(),
            monte_carlo=True,
            mc_simulations=80,
        )

        mc = result["monte_carlo"]
        assert 0.0 <= mc["prob_loss"] <= 1.0
        assert 0.0 <= mc["prob_profit"] <= 1.0
        assert mc["prob_loss"] + mc["prob_profit"] <= 1.0 + 1e-6

    def test_monte_carlo_custom_lot(self) -> None:
        """mc_lot パラメータが反映される."""
        np.random.seed(123)  # 123# Gemini review: seed 固定で Flaky 防止

        records = _make_records_mixed(n_filled=50, n_cancelled=10)
        result_small = run_gate_judgment(
            records, _default_gate_cfg(),
            monte_carlo=True, mc_simulations=60, mc_lot=0.001,
        )
        result_large = run_gate_judgment(
            records, _default_gate_cfg(),
            monte_carlo=True, mc_simulations=60, mc_lot=0.01,
        )

        mc_small = result_small["monte_carlo"]
        mc_large = result_large["monte_carlo"]
        # lot 10x → PnL std should scale approximately
        # (not exact due to MC randomness, but should be roughly proportional)
        assert abs(mc_large["pnl_std_jpy"]) > abs(mc_small["pnl_std_jpy"])

    def test_monte_carlo_json_serializable(self) -> None:
        """MC 結果を含む result が JSON シリアライズ可能."""
        records = _make_records_mixed(n_filled=50, n_cancelled=10)
        result = run_gate_judgment(
            records, _default_gate_cfg(),
            monte_carlo=True, mc_simulations=60,
        )

        serialized = json.dumps(result, ensure_ascii=False)
        parsed = json.loads(serialized)
        assert "monte_carlo" in parsed


# =====================================================================
# _load_all_records
# =====================================================================

class TestLoadAllRecords:
    """_load_all_records の読み込みテスト."""

    def test_empty_directory(self) -> None:
        """空ディレクトリでは空リスト返却."""
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _load_all_records(Path(tmpdir))
            assert records == []

    def test_load_single_file(self) -> None:
        """JSONL 1 ファイルからの読み込み."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recs = _make_records_mixed(n_filled=10, n_cancelled=2, days=1)
            save_fill_records(recs, Path(tmpdir) / "fill_records_20260220.jsonl")

            loaded = _load_all_records(Path(tmpdir))
            assert len(loaded) == len(recs)

    def test_load_multiple_files(self) -> None:
        """複数 JSONL ファイルからの読み込み."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recs1 = _make_records_mixed(n_filled=5, n_cancelled=1, days=1)
            recs2 = [
                replace(record, cycle_id=f"alt_{record.cycle_id}")
                for record in _make_records_mixed(n_filled=5, n_cancelled=1, days=1)
            ]
            save_fill_records(recs1, Path(tmpdir) / "fill_records_20260220.jsonl")
            save_fill_records(recs2, Path(tmpdir) / "fill_records_20260221.jsonl")

            loaded = _load_all_records(Path(tmpdir))
            assert len(loaded) == len(recs1) + len(recs2)


# =====================================================================
# _side_metrics
# =====================================================================

class TestSideMetrics:
    """_side_metrics のテスト."""

    def test_buy_metrics(self) -> None:
        """buy 側のメトリクスが算出される."""
        records = _make_records_mixed(n_filled=50, n_cancelled=10)
        buy_m = _side_metrics(records, "buy")

        assert buy_m["n"] > 0
        assert "fill_rate" in buy_m
        assert "pnl_30s_mean" in buy_m

    def test_empty_side(self) -> None:
        """該当 side が 0 件の場合."""
        records = [_make_record(side="buy") for _ in range(5)]
        sell_m = _side_metrics(records, "sell")

        assert sell_m == {"n": 0}
