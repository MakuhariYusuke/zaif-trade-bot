"""158# P0-4: Oracle テスト単体テスト.

oracle_test.py の core 計算ロジックを直接テスト。
データローダーをモックし、純粋な DataFrame ベースで検証。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.v460.analysis.oracle_test import run_oracle_test


def _make_fill_df(
    pnl30_values: list[float],
    sides: list[str] | None = None,
    adverse: list[bool] | None = None,
    pnl120_values: list[float] | None = None,
) -> pd.DataFrame:
    """テスト用 fill_records DataFrame を構築."""
    n = len(pnl30_values)
    df = pd.DataFrame({
        "filled": [True] * n,
        "post_fill_30s_pnl": pnl30_values,
        "side": sides or (["buy"] * (n // 2) + ["sell"] * (n - n // 2)),
        "adverse_selected": adverse or ([False] * n),
    })
    if pnl120_values is not None:
        df["post_fill_120s_pnl"] = pnl120_values
    return df


class TestOracleComputation:
    """Oracle 計算ロジックの正確性テスト."""

    @patch("scripts.v460.analysis.oracle_test.enrich_fill_records")
    @patch("scripts.v460.analysis.oracle_test.load_fill_records")
    def test_all_positive_baseline(self, mock_load: object, mock_enrich: object) -> None:
        """全トレード黒字 → baseline = oracle_skip."""
        df = _make_fill_df([1.0, 2.0, 3.0, 4.0])
        mock_load.return_value = df  # type: ignore[union-attr]
        mock_enrich.return_value = df  # type: ignore[union-attr]

        result = run_oracle_test(results_dir="dummy")

        assert result["status"] == "completed"
        pnl30 = result["oracle"]["pnl30"]
        assert pnl30["baseline_mean_bps"] == pytest.approx(2.5, abs=0.001)
        assert pnl30["oracle_skip_mean_bps"] == pytest.approx(2.5, abs=0.001)
        assert pnl30["profitable_rate"] == pytest.approx(1.0, abs=0.001)

    @patch("scripts.v460.analysis.oracle_test.enrich_fill_records")
    @patch("scripts.v460.analysis.oracle_test.load_fill_records")
    def test_mixed_pnl_oracle_skip(self, mock_load: object, mock_enrich: object) -> None:
        """正負混在 → Oracle Skip は正のみ平均."""
        pnls = [3.0, -1.0, 5.0, -2.0]  # positive=[3,5], negative=[-1,-2]
        df = _make_fill_df(pnls)
        mock_load.return_value = df  # type: ignore[union-attr]
        mock_enrich.return_value = df  # type: ignore[union-attr]

        result = run_oracle_test(results_dir="dummy")
        pnl30 = result["oracle"]["pnl30"]

        # Baseline = (3-1+5-2)/4 = 1.25
        assert pnl30["baseline_mean_bps"] == pytest.approx(1.25, abs=0.001)
        # Oracle Skip = (3+5)/2 = 4.0
        assert pnl30["oracle_skip_mean_bps"] == pytest.approx(4.0, abs=0.001)
        # Oracle Flip = mean(|3|,|1|,|5|,|2|) = 2.75
        assert pnl30["oracle_flip_mean_bps"] == pytest.approx(2.75, abs=0.001)
        # Profitable rate = 2/4 = 0.5
        assert pnl30["profitable_rate"] == pytest.approx(0.5, abs=0.001)

    @patch("scripts.v460.analysis.oracle_test.enrich_fill_records")
    @patch("scripts.v460.analysis.oracle_test.load_fill_records")
    def test_all_negative_oracle_zero(self, mock_load: object, mock_enrich: object) -> None:
        """全トレード赤字 → Oracle Skip mean = 0."""
        df = _make_fill_df([-1.0, -2.0, -3.0])
        mock_load.return_value = df  # type: ignore[union-attr]
        mock_enrich.return_value = df  # type: ignore[union-attr]

        result = run_oracle_test(results_dir="dummy")
        pnl30 = result["oracle"]["pnl30"]
        assert pnl30["oracle_skip_mean_bps"] == pytest.approx(0.0, abs=0.001)
        assert pnl30["oracle_skip_rate"] == pytest.approx(1.0, abs=0.001)

    @patch("scripts.v460.analysis.oracle_test.enrich_fill_records")
    @patch("scripts.v460.analysis.oracle_test.load_fill_records")
    def test_kill_switch_pass(self, mock_load: object, mock_enrich: object) -> None:
        """Oracle > 1.0 bps → PASS."""
        pnls = [5.0, -1.0, 4.0, -0.5, 3.0, -2.0]
        df = _make_fill_df(pnls)
        mock_load.return_value = df  # type: ignore[union-attr]
        mock_enrich.return_value = df  # type: ignore[union-attr]

        result = run_oracle_test(results_dir="dummy")
        assert result["kill_switch"]["pnl30"] == "PASS"
        assert result["kill_switch"]["oracle_pnl30_bps"] > 1.0

    @patch("scripts.v460.analysis.oracle_test.enrich_fill_records")
    @patch("scripts.v460.analysis.oracle_test.load_fill_records")
    def test_kill_switch_fail(self, mock_load: object, mock_enrich: object) -> None:
        """Oracle全赤字 → FAIL."""
        df = _make_fill_df([-3.0, -2.0, -1.0, -4.0])
        mock_load.return_value = df  # type: ignore[union-attr]
        mock_enrich.return_value = df  # type: ignore[union-attr]

        result = run_oracle_test(results_dir="dummy")
        assert result["kill_switch"]["pnl30"] == "FAIL"


class TestASCostAnalysis:
    """AS コスト分析 (158# P0-4 追加分) テスト."""

    @patch("scripts.v460.analysis.oracle_test.enrich_fill_records")
    @patch("scripts.v460.analysis.oracle_test.load_fill_records")
    def test_as_cost_computation(self, mock_load: object, mock_enrich: object) -> None:
        """AS cost = AS_ratio × |avg_AS_loss|."""
        # 10 records: 3 AS (pnl=-6,-4,-2 → avg=-4), 7 non-AS (pnl=+2 each)
        pnls = [-6.0, -4.0, -2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        adverse = [True, True, True, False, False, False, False, False, False, False]
        df = _make_fill_df(pnls, adverse=adverse)
        mock_load.return_value = df  # type: ignore[union-attr]
        mock_enrich.return_value = df  # type: ignore[union-attr]

        result = run_oracle_test(results_dir="dummy")
        as_cost = result["as_cost"]

        assert as_cost["n_as"] == 3
        assert as_cost["n_non_as"] == 7
        assert as_cost["as_ratio"] == pytest.approx(0.3, abs=0.001)
        assert as_cost["as_avg_pnl30_bps"] == pytest.approx(-4.0, abs=0.001)
        assert as_cost["non_as_avg_pnl30_bps"] == pytest.approx(2.0, abs=0.001)
        # AS cost = 0.3 × 4.0 = 1.2
        assert as_cost["as_cost_bps"] == pytest.approx(1.2, abs=0.001)

    @patch("scripts.v460.analysis.oracle_test.enrich_fill_records")
    @patch("scripts.v460.analysis.oracle_test.load_fill_records")
    def test_oracle_net_positive(self, mock_load: object, mock_enrich: object) -> None:
        """Oracle net = Oracle Flip - AS cost > 0 → PASS."""
        pnls = [5.0, -1.0, 4.0, -0.5]
        adverse = [False, True, False, True]
        df = _make_fill_df(pnls, adverse=adverse)
        mock_load.return_value = df  # type: ignore[union-attr]
        mock_enrich.return_value = df  # type: ignore[union-attr]

        result = run_oracle_test(results_dir="dummy")
        as_cost = result["as_cost"]

        # Oracle Flip = mean(|5|,|1|,|4|,|0.5|) = 2.625
        # AS: n=2, avg=(-1-0.5)/2 = -0.75, ratio=0.5
        # AS cost = 0.5 × 0.75 = 0.375
        # Oracle net = 2.625 - 0.375 = 2.25
        assert as_cost["oracle_net_of_as_bps"] == pytest.approx(2.25, abs=0.001)
        assert as_cost["oracle_net_of_as_bps"] > 0  # PASS

    @patch("scripts.v460.analysis.oracle_test.enrich_fill_records")
    @patch("scripts.v460.analysis.oracle_test.load_fill_records")
    def test_no_as_records(self, mock_load: object, mock_enrich: object) -> None:
        """AS レコードなし → as_cost は空."""
        df = _make_fill_df([1.0, 2.0])
        # adverse_selected 全て False
        mock_load.return_value = df  # type: ignore[union-attr]
        mock_enrich.return_value = df  # type: ignore[union-attr]

        result = run_oracle_test(results_dir="dummy")
        # n_as==0 なので as_cost は空 dict
        as_cost = result.get("as_cost", {})
        assert as_cost.get("n_as") is None or as_cost.get("n_as") == 0

    @patch("scripts.v460.analysis.oracle_test.enrich_fill_records")
    @patch("scripts.v460.analysis.oracle_test.load_fill_records")
    def test_side_analysis_present(self, mock_load: object, mock_enrich: object) -> None:
        """Side 別分析が buy/sell 両方含まれる."""
        pnls = [3.0, -1.0, 5.0, -2.0]
        sides = ["buy", "buy", "sell", "sell"]
        df = _make_fill_df(pnls, sides=sides)
        mock_load.return_value = df  # type: ignore[union-attr]
        mock_enrich.return_value = df  # type: ignore[union-attr]

        result = run_oracle_test(results_dir="dummy")
        side_analysis = result["oracle"]["pnl30"]["side_analysis"]

        assert "buy" in side_analysis
        assert "sell" in side_analysis
        # buy: [3, -1] → mean=1.0, profitable=50%
        assert side_analysis["buy"]["mean_bps"] == pytest.approx(1.0, abs=0.001)
        assert side_analysis["buy"]["profitable_rate"] == pytest.approx(0.5, abs=0.001)
        # sell: [5, -2] → mean=1.5, profitable=50%
        assert side_analysis["sell"]["mean_bps"] == pytest.approx(1.5, abs=0.001)


class TestPnl120Horizon:
    """120s horizon テスト."""

    @patch("scripts.v460.analysis.oracle_test.enrich_fill_records")
    @patch("scripts.v460.analysis.oracle_test.load_fill_records")
    def test_pnl120_computed(self, mock_load: object, mock_enrich: object) -> None:
        """pnl120 カラムがあれば 120s horizon も計算."""
        pnl30 = [2.0, -1.0, 3.0, -0.5]
        pnl120 = [5.0, -2.0, 6.0, -1.0]
        df = _make_fill_df(pnl30, pnl120_values=pnl120)
        mock_load.return_value = df  # type: ignore[union-attr]
        mock_enrich.return_value = df  # type: ignore[union-attr]

        result = run_oracle_test(results_dir="dummy")

        assert "pnl120" in result["oracle"]
        pnl120_result = result["oracle"]["pnl120"]
        # Oracle Skip 120s: (5+6)/2 = 5.5
        assert pnl120_result["oracle_skip_mean_bps"] == pytest.approx(5.5, abs=0.001)
        assert result["kill_switch"]["pnl120"] == "PASS"


class TestNonFiniteHandling:
    """非有限値の入力を安全に無視すること."""

    @patch("scripts.v460.analysis.oracle_test.enrich_fill_records")
    @patch("scripts.v460.analysis.oracle_test.load_fill_records")
    def test_non_finite_pnl_ignored(self, mock_load: object, mock_enrich: object) -> None:
        df = _make_fill_df([1.0, float("nan"), float("inf"), -2.0])
        mock_load.return_value = df  # type: ignore[union-attr]
        mock_enrich.return_value = df  # type: ignore[union-attr]

        result = run_oracle_test(results_dir="dummy")
        pnl30 = result["oracle"]["pnl30"]

        # finite values are [1.0, -2.0]
        assert pnl30["n"] == 2
        assert pnl30["baseline_mean_bps"] == pytest.approx(-0.5, abs=0.001)
        assert pnl30["oracle_skip_mean_bps"] == pytest.approx(1.0, abs=0.001)
