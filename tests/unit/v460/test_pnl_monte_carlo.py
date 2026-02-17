"""
PnL モンテカルロシミュレータ単体テスト — 014# T5.

ztb/risk/pnl_monte_carlo.py の全主要パスを検証する。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ztb.metrics.fill_quality import FillRecord
from ztb.risk.pnl_monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    PnLMonteCarloSimulator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_record(
    cycle_id: str = "test_001",
    filled: bool = True,
    cancelled: bool = False,
    pnl_bps: float | None = 0.5,
    adverse: bool | None = False,
    price: float = 10_300_000.0,
    wait: float = 5.0,
    side: str = "buy",
) -> FillRecord:
    """テスト用 FillRecord のファクトリ."""
    return FillRecord(
        cycle_id=cycle_id,
        timestamp=1770975573.0,
        side=side,
        order_price=price,
        order_quantity=0.001,
        fill_price=price if filled else None,
        filled=filled,
        cancelled=cancelled,
        queue_wait_sec=wait,
        mid_at_fill=price + 100 if filled else None,
        mid_30s_after=price + 200 if filled else None,
        post_fill_30s_pnl=pnl_bps if filled else None,
        adverse_selected=adverse if filled else None,
    )


def _make_records_typical(n_filled: int = 8, n_cancelled: int = 2) -> list[FillRecord]:
    """典型的なレコードセット (PnL 正分布)."""
    recs: list[FillRecord] = []
    # Filled records with diverse PnL
    pnl_values = [0.5, 1.0, -0.3, 0.8, 1.5, -0.1, 0.3, 0.7, 1.2, -0.5]
    adverse_flags = [False, False, True, False, False, True, False, False, False, True]
    for i in range(n_filled):
        pnl = pnl_values[i % len(pnl_values)]
        adv = adverse_flags[i % len(adverse_flags)]
        recs.append(_make_record(
            cycle_id=f"filled_{i:03d}",
            filled=True,
            pnl_bps=pnl,
            adverse=adv,
            wait=3.0 + i * 0.5,
        ))
    for i in range(n_cancelled):
        recs.append(_make_record(
            cycle_id=f"cancel_{i:03d}",
            filled=False,
            cancelled=True,
        ))
    return recs


def _write_jsonl(records: list[FillRecord], path: Path) -> None:
    """FillRecord リストを JSONL に書き出す."""
    lines: list[str] = []
    for r in records:
        lines.append(json.dumps(r.to_dict(), ensure_ascii=False))
    path.write_text("\n".join(lines), encoding="utf-8")


# =====================================================================
# FillRecord
# =====================================================================

class TestFillRecord:
    """FillRecord データクラスの基本テスト."""

    def test_creation_filled(self) -> None:
        r = _make_record(filled=True, pnl_bps=1.5)
        assert r.filled is True
        assert r.cancelled is False
        assert r.post_fill_30s_pnl == 1.5

    def test_creation_cancelled(self) -> None:
        r = _make_record(filled=False, cancelled=True)
        assert r.filled is False
        assert r.cancelled is True
        assert r.post_fill_30s_pnl is None
        assert r.fill_price is None


# =====================================================================
# Load fill records
# =====================================================================

class TestLoadFillRecords:
    """JSONL ファイルからの読み込みテスト."""

    def test_load_from_file(self, tmp_path: Path) -> None:
        recs = _make_records_typical(3, 1)
        f = tmp_path / "fill_records_test.jsonl"
        _write_jsonl(recs, f)

        loaded = PnLMonteCarloSimulator.load_fill_records(f)
        assert len(loaded) == 4
        assert sum(1 for r in loaded if r.filled) == 3
        assert sum(1 for r in loaded if r.cancelled) == 1

    def test_load_from_directory(self, tmp_path: Path) -> None:
        recs1 = _make_records_typical(2, 1)
        # 101# §5: cycle_id をユニークにする (dedup 対応)
        recs2_raw = _make_records_typical(3, 0)
        recs2: list[FillRecord] = []
        for r in recs2_raw:
            d = r.to_dict()
            d["cycle_id"] = f"day2_{d['cycle_id']}"
            recs2.append(FillRecord.from_dict(d))
        _write_jsonl(recs1, tmp_path / "fill_records_day1.jsonl")
        _write_jsonl(recs2, tmp_path / "fill_records_day2.jsonl")

        loaded = PnLMonteCarloSimulator.load_fill_records(tmp_path)
        assert len(loaded) == 6  # 3 + 3

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            PnLMonteCarloSimulator.load_fill_records("/nonexistent/path")

    def test_load_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "fill_records_empty.jsonl"
        f.write_text("", encoding="utf-8")
        loaded = PnLMonteCarloSimulator.load_fill_records(f)
        assert len(loaded) == 0


# =====================================================================
# MonteCarloConfig
# =====================================================================

class TestMonteCarloConfig:
    """MonteCarloConfig の設定値テスト."""

    def test_defaults(self) -> None:
        cfg = MonteCarloConfig()
        assert cfg.n_simulations == 10_000
        assert cfg.cycles_per_day == 720
        assert cfg.days_per_month == 30
        assert cfg.lot_size_btc == 0.001
        assert cfg.maker_fee_rate == 0.0

    def test_custom_values(self) -> None:
        cfg = MonteCarloConfig(n_simulations=100, btc_price_jpy=15_000_000.0)
        assert cfg.n_simulations == 100
        assert cfg.btc_price_jpy == 15_000_000.0


# =====================================================================
# Simulator constructor
# =====================================================================

class TestSimulatorInit:
    """PnLMonteCarloSimulator の初期化テスト."""

    def test_empty_records_raises(self) -> None:
        with pytest.raises(ValueError, match="No fill records"):
            PnLMonteCarloSimulator([])

    def test_accepts_records(self) -> None:
        recs = _make_records_typical(3, 1)
        sim = PnLMonteCarloSimulator(recs)
        assert len(sim.records) == 4

    def test_custom_config(self) -> None:
        recs = _make_records_typical(3, 1)
        cfg = MonteCarloConfig(n_simulations=50)
        sim = PnLMonteCarloSimulator(recs, config=cfg)
        assert sim.config.n_simulations == 50


# =====================================================================
# Core simulation — run()
# =====================================================================

class TestSimulationRun:
    """run() メソッドの各種シナリオテスト."""

    def test_basic_run_returns_result(self) -> None:
        recs = _make_records_typical(8, 2)
        cfg = MonteCarloConfig(n_simulations=100, random_seed=42)
        sim = PnLMonteCarloSimulator(recs, cfg)
        result = sim.run()

        assert isinstance(result, MonteCarloResult)
        assert result.n_records == 10
        assert result.n_filled == 8
        assert result.n_cancelled == 2
        assert result.n_simulations == 100
        assert result.cycles_per_month == 720 * 30

    def test_observed_statistics(self) -> None:
        recs = _make_records_typical(8, 2)
        cfg = MonteCarloConfig(n_simulations=100)
        sim = PnLMonteCarloSimulator(recs, cfg)
        result = sim.run()

        assert result.observed_fill_rate == pytest.approx(0.8)
        assert result.g11_cancel_ratio == pytest.approx(0.2)
        assert result.observed_pnl_mean_bps > -10  # sanity check
        assert result.observed_pnl_mean_bps < 10

    def test_percentiles_exist(self) -> None:
        recs = _make_records_typical(8, 2)
        cfg = MonteCarloConfig(n_simulations=100)
        sim = PnLMonteCarloSimulator(recs, cfg)
        result = sim.run()

        assert "5%" in result.pnl_percentiles_jpy
        assert "50%" in result.pnl_percentiles_jpy
        assert "95%" in result.pnl_percentiles_jpy
        # Percentiles should be ordered
        assert result.pnl_percentiles_jpy["5%"] <= result.pnl_percentiles_jpy["50%"]
        assert result.pnl_percentiles_jpy["50%"] <= result.pnl_percentiles_jpy["95%"]

    def test_positive_pnl_scenario(self) -> None:
        """全 filled, PnL 全正 → 月次PnL > 0 が期待される."""
        recs = [_make_record(f"pos_{i}", pnl_bps=1.0 + i * 0.1) for i in range(10)]
        cfg = MonteCarloConfig(n_simulations=500, random_seed=42)
        sim = PnLMonteCarloSimulator(recs, cfg)
        result = sim.run()

        assert result.pnl_mean_jpy > 0
        assert result.prob_profit == pytest.approx(1.0, abs=0.01)
        assert result.prob_loss == pytest.approx(0.0, abs=0.01)

    def test_negative_pnl_scenario(self) -> None:
        """全 filled, PnL 全負 → 月次PnL < 0 が期待される."""
        recs = [_make_record(f"neg_{i}", pnl_bps=-2.0 - i * 0.1) for i in range(10)]
        cfg = MonteCarloConfig(n_simulations=500, random_seed=42)
        sim = PnLMonteCarloSimulator(recs, cfg)
        result = sim.run()

        assert result.pnl_mean_jpy < 0
        assert result.prob_loss == pytest.approx(1.0, abs=0.01)

    def test_all_cancelled_scenario(self) -> None:
        """全キャンセル → PnL = 0."""
        recs = [
            _make_record(f"cancel_{i}", filled=False, cancelled=True)
            for i in range(5)
        ]
        cfg = MonteCarloConfig(n_simulations=100)
        sim = PnLMonteCarloSimulator(recs, cfg)
        result = sim.run()

        assert result.observed_fill_rate == 0.0
        assert result.pnl_mean_jpy == pytest.approx(0.0)
        assert result.n_filled == 0

    def test_single_record_works(self) -> None:
        """n=1 でも動作する."""
        recs = [_make_record("only_one", pnl_bps=1.0)]
        cfg = MonteCarloConfig(n_simulations=100)
        sim = PnLMonteCarloSimulator(recs, cfg)
        result = sim.run()

        assert result.n_records == 1
        assert result.observed_fill_rate == 1.0
        assert result.pnl_mean_jpy > 0

    def test_reproducibility_with_seed(self) -> None:
        """同じ seed → 同結果."""
        recs = _make_records_typical(8, 2)
        cfg1 = MonteCarloConfig(n_simulations=200, random_seed=12345)
        cfg2 = MonteCarloConfig(n_simulations=200, random_seed=12345)

        result1 = PnLMonteCarloSimulator(recs, cfg1).run()
        result2 = PnLMonteCarloSimulator(recs, cfg2).run()

        assert result1.pnl_mean_jpy == pytest.approx(result2.pnl_mean_jpy)
        assert result1.var_95_jpy == pytest.approx(result2.var_95_jpy)

    def test_different_seed_different_results(self) -> None:
        """異なる seed → 異なる結果 (統計結果は近いが厳密不一致)."""
        recs = _make_records_typical(8, 2)
        cfg1 = MonteCarloConfig(n_simulations=1000, random_seed=1)
        cfg2 = MonteCarloConfig(n_simulations=1000, random_seed=999)

        result1 = PnLMonteCarloSimulator(recs, cfg1).run()
        result2 = PnLMonteCarloSimulator(recs, cfg2).run()

        # 平均は近いが厳密一致はしない（小さい差はあり得る）
        # raw_monthly_pnls が異なることで確認
        assert not np.array_equal(result1.raw_monthly_pnls, result2.raw_monthly_pnls)

    def test_var_cvar_relationship(self) -> None:
        """CVaR ≤ VaR (CVaR は tail mean で VaR 以下)."""
        recs = _make_records_typical(8, 2)
        cfg = MonteCarloConfig(n_simulations=1000, random_seed=42)
        sim = PnLMonteCarloSimulator(recs, cfg)
        result = sim.run()

        assert result.cvar_95_jpy <= result.var_95_jpy + 1e-6  # tolerance


# =====================================================================
# G1.1 criteria
# =====================================================================

class TestG11Criteria:
    """G1.1 pass/fail 判定テスト."""

    def test_g11_pass_all_criteria_met(self) -> None:
        """全指標クリア → G1.1 PASS."""
        # fill_rate ≥ 90%, cancel ≤ 30%, queue ≤ 60s, pnl ≥ 0, AS ≤ 20%
        filled = [
            _make_record(f"fill_{i}", filled=True, pnl_bps=0.5,
                         adverse=False, wait=3.0)
            for i in range(9)
        ]
        cancelled = [
            _make_record("cancel_0", filled=False, cancelled=True)
        ]
        recs = filled + cancelled
        # 9/10 = 90%, cancel 1/10 = 10%, AS = 0%
        cfg = MonteCarloConfig(n_simulations=100)
        sim = PnLMonteCarloSimulator(recs, cfg)
        result = sim.run()

        assert result.g11_fill_rate >= 0.90
        assert result.g11_cancel_ratio <= 0.30
        assert result.g11_queue_wait_median <= 60.0
        assert result.g11_pnl_mean_bps >= 0.0
        assert result.g11_as_ratio <= 0.20
        assert result.g11_pass is True

    def test_g11_fail_low_fill_rate(self) -> None:
        """fill_rate < 90% → FAIL."""
        filled = [_make_record(f"f_{i}", pnl_bps=1.0, adverse=False) for i in range(5)]
        cancelled = [_make_record(f"c_{i}", filled=False, cancelled=True) for i in range(5)]
        recs = filled + cancelled
        # fill_rate = 50% < 90%
        cfg = MonteCarloConfig(n_simulations=50)
        result = PnLMonteCarloSimulator(recs, cfg).run()
        assert result.g11_pass is False

    def test_g11_fail_high_as_ratio(self) -> None:
        """AS > 20% → FAIL."""
        recs = [
            _make_record(f"fill_{i}", pnl_bps=0.5, adverse=(i < 3), wait=3.0)
            for i in range(10)
        ]
        # AS = 3/10 = 30% > 20%, but fill_rate = 100%!
        cfg = MonteCarloConfig(n_simulations=50)
        result = PnLMonteCarloSimulator(recs, cfg).run()
        assert result.g11_as_ratio > 0.20
        assert result.g11_pass is False


# =====================================================================
# Sensitivity analysis
# =====================================================================

class TestSensitivityAnalysis:
    """感度分析テスト."""

    def test_returns_grid(self) -> None:
        recs = _make_records_typical(8, 2)
        cfg = MonteCarloConfig(n_simulations=50, random_seed=42)
        sim = PnLMonteCarloSimulator(recs, cfg)

        results = sim.sensitivity_analysis(
            fill_rates=[0.5, 0.9],
            pnl_adjustments_bps=[-1.0, 0.0, 1.0],
        )

        assert len(results) == 6  # 2 × 3
        for r in results:
            assert "fill_rate" in r
            assert "pnl_adj_bps" in r
            assert "mean_jpy" in r
            assert "var_95_jpy" in r
            assert "prob_loss" in r

    def test_higher_fill_rate_more_pnl(self) -> None:
        """fill_rate 高い方が PnL mean 高い (PnL mean > 0 のとき)."""
        recs = [_make_record(f"pos_{i}", pnl_bps=1.0) for i in range(10)]
        cfg = MonteCarloConfig(n_simulations=200, random_seed=42)
        sim = PnLMonteCarloSimulator(recs, cfg)

        results = sim.sensitivity_analysis(
            fill_rates=[0.5, 1.0],
            pnl_adjustments_bps=[0.0],
        )
        low_fr = next(r for r in results if r["fill_rate"] == 0.5)
        high_fr = next(r for r in results if r["fill_rate"] == 1.0)
        assert high_fr["mean_jpy"] > low_fr["mean_jpy"]

    def test_positive_adjustment_increases_pnl(self) -> None:
        """pnl_adj > 0 → PnL 増加."""
        recs = _make_records_typical(8, 2)
        cfg = MonteCarloConfig(n_simulations=200, random_seed=42)
        sim = PnLMonteCarloSimulator(recs, cfg)

        results = sim.sensitivity_analysis(
            fill_rates=[0.8],
            pnl_adjustments_bps=[-1.0, 1.0],
        )
        neg_adj = next(r for r in results if r["pnl_adj_bps"] == -1.0)
        pos_adj = next(r for r in results if r["pnl_adj_bps"] == 1.0)
        assert pos_adj["mean_jpy"] > neg_adj["mean_jpy"]

    def test_default_grid_size(self) -> None:
        recs = _make_records_typical(8, 2)
        cfg = MonteCarloConfig(n_simulations=50)
        sim = PnLMonteCarloSimulator(recs, cfg)

        results = sim.sensitivity_analysis()
        # Default: 7 fill_rates × 5 pnl_adjustments = 35
        assert len(results) == 35


# =====================================================================
# Report & serialization
# =====================================================================

class TestReportAndSerialization:
    """print_report / to_dict テスト."""

    def test_print_report_returns_string(self) -> None:
        recs = _make_records_typical(8, 2)
        cfg = MonteCarloConfig(n_simulations=50)
        sim = PnLMonteCarloSimulator(recs, cfg)
        result = sim.run()

        report = sim.print_report(result)
        assert isinstance(report, str)
        assert "PnL Monte Carlo Report" in report
        assert "G1.1" in report

    def test_report_contains_key_sections(self) -> None:
        recs = _make_records_typical(8, 2)
        cfg = MonteCarloConfig(n_simulations=50)
        sim = PnLMonteCarloSimulator(recs, cfg)
        result = sim.run()

        report = sim.print_report(result)
        assert "Observed Data" in report
        assert "Monthly Simulation" in report
        assert "Risk Metrics" in report
        assert "G1.1 Criteria" in report

    def test_to_dict_serializable(self) -> None:
        recs = _make_records_typical(8, 2)
        cfg = MonteCarloConfig(n_simulations=50)
        sim = PnLMonteCarloSimulator(recs, cfg)
        result = sim.run()

        d = result.to_dict()
        # Must be JSON serializable
        s = json.dumps(d)
        assert isinstance(s, str)

        # Key fields present
        assert "n_records" in d
        assert "pnl_mean_jpy" in d
        assert "var_95_jpy" in d
        assert "g11_pass" in d
        assert "pnl_percentiles_jpy" in d

    def test_to_dict_values_match_result(self) -> None:
        recs = _make_records_typical(8, 2)
        cfg = MonteCarloConfig(n_simulations=50)
        sim = PnLMonteCarloSimulator(recs, cfg)
        result = sim.run()

        d = result.to_dict()
        assert d["n_records"] == result.n_records
        assert d["n_filled"] == result.n_filled
        assert d["g11_pass"] == result.g11_pass
        assert d["pnl_mean_jpy"] == pytest.approx(result.pnl_mean_jpy)


# =====================================================================
# Integration — end-to-end with JSONL file
# =====================================================================

class TestEndToEnd:
    """JSONL ファイル → シミュレーション → レポート E2E テスト."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        recs = _make_records_typical(10, 2)
        f = tmp_path / "fill_records_test.jsonl"
        _write_jsonl(recs, f)

        # Load
        loaded = PnLMonteCarloSimulator.load_fill_records(f)
        assert len(loaded) == 12

        # Simulate
        cfg = MonteCarloConfig(n_simulations=100, random_seed=42)
        sim = PnLMonteCarloSimulator(loaded, cfg)
        result = sim.run()

        # Report
        report = sim.print_report(result)
        assert len(report) > 100

        # Serialize
        d = result.to_dict()
        out = tmp_path / "mc_result.json"
        out.write_text(json.dumps(d, indent=2), encoding="utf-8")
        reloaded = json.loads(out.read_text(encoding="utf-8"))
        assert reloaded["n_records"] == 12

    def test_multiple_files_merge(self, tmp_path: Path) -> None:
        """複数ファイルのマージ → 正しくシミュレート."""
        _write_jsonl(_make_records_typical(5, 1), tmp_path / "fill_records_d1.jsonl")
        # 101# §5: cycle_id をユニークにする (dedup 対応)
        recs2_raw = _make_records_typical(5, 1)
        recs2: list[FillRecord] = []
        for r in recs2_raw:
            d = r.to_dict()
            d["cycle_id"] = f"d2_{d['cycle_id']}"
            recs2.append(FillRecord.from_dict(d))
        _write_jsonl(recs2, tmp_path / "fill_records_d2.jsonl")

        loaded = PnLMonteCarloSimulator.load_fill_records(tmp_path)
        assert len(loaded) == 12

        cfg = MonteCarloConfig(n_simulations=50)
        sim = PnLMonteCarloSimulator(loaded, cfg)
        result = sim.run()
        assert result.n_records == 12
