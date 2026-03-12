"""396# G3 パイプラインのテスト.

_compute_g3_metrics (sac_common.py) および
evaluate_g3_checks (gate_judgment_core.py) を検証。

389# P0-3: reward-PnL alignment (reward_profit_corr)
389# P1-1: G3 gate 接続 (seed_metrics → evaluate_g3_checks)
399# 統合: _evaluate_g3_from_results → evaluate_g3_checks
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scripts.v460.lib.sac_common import _compute_g3_metrics
from scripts.v460.lib.gate_judgment_core import evaluate_g3_checks


# =====================================================================
# _compute_g3_metrics
# =====================================================================


class TestComputeG3Metrics:
    """_compute_g3_metrics の単体テスト."""

    def test_empty_inputs(self) -> None:
        """空入力ではデフォルト値 (0.0) を返す."""
        result = _compute_g3_metrics([], [], [], 0)
        assert result["pf"] == 0.0
        assert result["max_drawdown"] == 0.0
        assert result["sharpe_annual"] == 0.0
        assert result["avg_gross_per_trade"] == 0.0
        assert result["avg_fee_per_trade"] == 0.0
        assert result["reward_profit_corr"] == 0.0

    def test_profit_factor_basic(self) -> None:
        """正PnL / |負PnL| で Profit Factor を計算."""
        # +10, +20, -5 → PF = 30 / 5 = 6.0
        pnl = [10.0, 20.0, -5.0]
        result = _compute_g3_metrics([100.0, 110.0, 130.0], pnl, [0.0] * 3, 3)
        assert result["pf"] == pytest.approx(6.0)

    def test_profit_factor_no_losses(self) -> None:
        """負PnL がない場合は inf."""
        pnl = [10.0, 20.0]
        result = _compute_g3_metrics([100.0, 110.0], pnl, [0.0] * 2, 2)
        assert result["pf"] == float("inf")

    def test_profit_factor_no_gains(self) -> None:
        """正PnL がない場合は 0."""
        pnl = [-10.0, -20.0]
        result = _compute_g3_metrics([100.0, 90.0], pnl, [0.0] * 2, 2)
        assert result["pf"] == 0.0

    def test_max_drawdown(self) -> None:
        """Max DD: peak-to-trough 比率を計算."""
        # 100 → 120 → 90 → 110
        # peak=120, trough=90, DD=(120-90)/120=0.25
        pv = [100.0, 120.0, 90.0, 110.0]
        result = _compute_g3_metrics(pv, [0.0] * 4, [0.0] * 4, 0)
        assert result["max_drawdown"] == pytest.approx(0.25)

    def test_max_drawdown_no_drawdown(self) -> None:
        """単調増加なら DD=0."""
        pv = [100.0, 110.0, 120.0, 130.0]
        result = _compute_g3_metrics(pv, [0.0] * 4, [0.0] * 4, 0)
        assert result["max_drawdown"] == pytest.approx(0.0)

    def test_sharpe_annual_basic(self) -> None:
        """1440*2 steps 以上で年率 Sharpe を計算."""
        # 2日分 (2880 steps) + 1 の PV を作る
        n = 1440 * 3  # 3日分
        rng = np.random.RandomState(42)
        # 日次リターン ~1% ± 0.5% → Sharpe ≈ 1%/0.5% * sqrt(365) ≈ 38
        daily_growth = 1.01
        pv = [100.0]
        for i in range(1, n):
            if i % 1440 == 0:
                pv.append(pv[-1] * (daily_growth + rng.normal(0, 0.001)))
            else:
                pv.append(pv[-1] * (1 + rng.normal(0, 0.0001)))

        result = _compute_g3_metrics(pv, [0.0] * n, [0.0] * n, 0)
        # Sharpe should be positive and finite
        assert result["sharpe_annual"] > 0.0
        assert math.isfinite(result["sharpe_annual"])

    def test_sharpe_insufficient_data(self) -> None:
        """1440 steps 未満ではSharpe=0."""
        pv = list(range(100, 200))  # 100 steps
        result = _compute_g3_metrics(pv, [0.0] * 100, [0.0] * 100, 0)
        assert result["sharpe_annual"] == 0.0

    def test_per_trade_averages(self) -> None:
        """取引あたり平均: 非ゼロ PnL の |p| 平均."""
        # PnL: +10, 0, -5, +3, 0 → trade_pnls = [10, -5, 3]
        # avg_gross = (10+5+3)/3 = 6.0
        pnl = [10.0, 0.0, -5.0, 3.0, 0.0]
        result = _compute_g3_metrics([100.0] * 5, pnl, [0.0] * 5, 3)
        assert result["avg_gross_per_trade"] == pytest.approx(6.0)
        assert result["avg_fee_per_trade"] == 0.0

    def test_per_trade_no_trades(self) -> None:
        """取引なしでは 0."""
        result = _compute_g3_metrics([100.0] * 5, [0.0] * 5, [0.0] * 5, 0)
        assert result["avg_gross_per_trade"] == 0.0

    def test_reward_profit_corr_positive(self) -> None:
        """reward と PnL が同方向 → 正相関."""
        n = 1000
        pnl = list(np.linspace(0, 1, n))
        reward = list(np.linspace(0, 1, n))  # 完全相関
        result = _compute_g3_metrics([100.0] * n, pnl, reward, 0)
        assert result["reward_profit_corr"] > 0.9

    def test_reward_profit_corr_negative(self) -> None:
        """reward と PnL が逆方向 → 負相関 (392# P0-3 ケース)."""
        n = 1000
        # PnL: 正 → 累積PnL 単調増加
        pnl = list(np.linspace(0.01, 0.02, n))
        # reward: 負 → 累積reward 単調減少 → PnL累積と逆方向
        reward = list(np.linspace(-0.01, -0.02, n))
        result = _compute_g3_metrics([100.0] * n, pnl, reward, 0)
        assert result["reward_profit_corr"] < -0.9

    def test_reward_profit_corr_insufficient_data(self) -> None:
        """100 steps 未満では 0."""
        result = _compute_g3_metrics([100.0] * 50, [1.0] * 50, [1.0] * 50, 0)
        assert result["reward_profit_corr"] == 0.0

    def test_reward_profit_corr_length_mismatch(self) -> None:
        """reward と PnL の長さが異なれば 0."""
        result = _compute_g3_metrics(
            [100.0] * 200, [1.0] * 200, [1.0] * 150, 0
        )
        assert result["reward_profit_corr"] == 0.0


# =====================================================================
# _evaluate_g3_from_results
# =====================================================================


def _make_seed_metrics(
    n_seeds: int = 4,
    pf: float = 1.3,
    sharpe: float = 1.5,
    max_dd: float = 0.08,
    gross: float = 0.005,
    fee: float = 0.002,
) -> list[dict]:
    """seed_metrics テストデータ生成."""
    rng = np.random.RandomState(42)
    return [
        {
            "seed": i,
            "pf": rng.normal(pf, 0.05),
            "sharpe_annual": rng.normal(sharpe, 0.1),
            "max_drawdown": rng.uniform(max_dd * 0.5, max_dd),
            "avg_gross_per_trade": gross,
            "avg_fee_per_trade": fee,
        }
        for i in range(n_seeds)
    ]


def _g3_thresholds() -> dict:
    return {
        "min_pf_median": 1.05,
        "min_pf_worst": 0.95,
        "gross_gt_fee": True,
        "max_drawdown": 0.15,
        "min_sharpe_annual": 0.8,
    }


class TestEvaluateG3FromResults:
    """evaluate_g3_checks の単体テスト."""

    def test_pass(self) -> None:
        """全チェック PASS のケース."""
        metrics = _make_seed_metrics()
        judgment = evaluate_g3_checks(metrics, _g3_thresholds())
        assert judgment["gate"] == "G3-pnl"
        assert judgment["gate_result"] == "PASS"
        assert judgment["n_seeds"] == 4
        assert all(c["pass"] for c in judgment["checks"].values())

    def test_no_data(self) -> None:
        """seed_metrics なし → NO_DATA."""
        judgment = evaluate_g3_checks([], _g3_thresholds())
        assert judgment["gate_result"] == "NO_DATA"

    def test_low_pf_median(self) -> None:
        """PF median < 1.05 → FAIL."""
        metrics = _make_seed_metrics(pf=0.8)
        judgment = evaluate_g3_checks(metrics, _g3_thresholds())
        assert judgment["gate_result"] == "FAIL"
        assert not judgment["checks"]["pf_median"]["pass"]

    def test_low_pf_worst(self) -> None:
        """worst-seed PF < 0.95 → FAIL."""
        metrics = _make_seed_metrics(pf=1.2)
        # 1 seed の PF を極端に下げる
        metrics[0]["pf"] = 0.5
        judgment = evaluate_g3_checks(metrics, _g3_thresholds())
        assert judgment["gate_result"] == "FAIL"
        assert not judgment["checks"]["pf_worst"]["pass"]

    def test_high_drawdown(self) -> None:
        """MaxDD > 15% → FAIL."""
        metrics = _make_seed_metrics(max_dd=0.20)
        judgment = evaluate_g3_checks(metrics, _g3_thresholds())
        assert judgment["gate_result"] == "FAIL"
        assert not judgment["checks"]["max_drawdown"]["pass"]

    def test_low_sharpe(self) -> None:
        """Sharpe median < 0.8 → FAIL."""
        metrics = _make_seed_metrics(sharpe=0.3)
        judgment = evaluate_g3_checks(metrics, _g3_thresholds())
        assert judgment["gate_result"] == "FAIL"
        assert not judgment["checks"]["sharpe_annual"]["pass"]

    def test_gross_lt_fee(self) -> None:
        """gross < fee → FAIL."""
        metrics = _make_seed_metrics(gross=0.001, fee=0.005)
        judgment = evaluate_g3_checks(metrics, _g3_thresholds())
        assert judgment["gate_result"] == "FAIL"
        assert not judgment["checks"]["gross_gt_fee"]["pass"]

    def test_default_thresholds(self) -> None:
        """thresholds={} のときデフォルト値で動作."""
        metrics = _make_seed_metrics()
        judgment = evaluate_g3_checks(metrics, {})
        assert judgment["gate"] == "G3-pnl"
        assert judgment["gate_result"] in ("PASS", "FAIL")  # 動作すれば OK

    def test_checks_structure(self) -> None:
        """checks の構造を検証."""
        metrics = _make_seed_metrics()
        judgment = evaluate_g3_checks(metrics, _g3_thresholds())
        expected_checks = {
            "pf_median", "pf_worst", "gross_gt_fee",
            "max_drawdown", "sharpe_annual",
        }
        assert set(judgment["checks"].keys()) == expected_checks
        for check in judgment["checks"].values():
            assert "value" in check
            assert "threshold" in check
            assert "pass" in check
