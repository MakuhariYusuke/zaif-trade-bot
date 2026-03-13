"""v460 train_sg_v3 の回帰テスト."""

from __future__ import annotations

import numpy as np

from scripts.v460.ml.train_sg_v3 import _dual_horizon_skip_sim


def test_dual_horizon_skip_sim_handles_all_nan_pnl120() -> None:
    """PnL120 が全欠損でも NaN を返さず処理継続できる."""
    probs = np.linspace(0.0, 1.0, 100, dtype=float)
    pnl30 = np.linspace(-1.0, 1.0, 100, dtype=float)
    pnl120 = np.full(100, np.nan, dtype=float)

    result = _dual_horizon_skip_sim(probs, pnl30, pnl120)

    assert result["baseline_pnl120"] == 0.0
    assert all(np.isfinite(v) for v in result.values())


def test_dual_horizon_skip_sim_returns_finite_with_partial_nan_pnl120() -> None:
    """PnL120 が部分欠損でも各改善値が有限値で返る."""
    probs = np.linspace(0.0, 1.0, 120, dtype=float)
    pnl30 = np.sin(np.linspace(0.0, 3.0, 120)).astype(float)
    pnl120 = np.cos(np.linspace(0.0, 3.0, 120)).astype(float)
    pnl120[::3] = np.nan  # 1/3 を欠損

    result = _dual_horizon_skip_sim(probs, pnl30, pnl120)

    assert all(np.isfinite(v) for v in result.values())
