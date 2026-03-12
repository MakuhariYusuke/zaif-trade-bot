import json
import tempfile
from pathlib import Path

import pandas as pd

from ztb.analysis.action_confidence_diagnostics import (
    extract_trades_from_step_logs,
    compute_trade_metrics,
    bin_and_aggregate,
)


def make_synthetic_trade_sequence():
    # Create synthetic trades: two trades
    trades = []
    # Trade 1: entry at step 0, duration 3 steps, small action
    trades.append({"step": 0, "action": 0.008, "price": 100, "position": 1, "pnl": 0.0})
    trades.append({"step": 1, "action": 0.008, "price": 101, "position": 1, "pnl": 1.0})
    trades.append({"step": 2, "action": 0.008, "price": 102, "position": 0, "pnl": 1.0})
    # Trade 2: entry at step 3, duration 2 steps, large action
    trades.append({"step": 3, "action": 0.02, "price": 200, "position": 1, "pnl": 0.0})
    trades.append({"step": 4, "action": 0.02, "price": 199, "position": 0, "pnl": -1.0})
    return trades


def test_extract_and_compute_metrics():
    trades = make_synthetic_trade_sequence()
    windows = extract_trades_from_step_logs(trades)
    assert len(windows) == 2

    m1 = compute_trade_metrics(windows[0])
    assert m1["realized_pnl"] == 2.0
    assert m1["mae"] == 0.0

    m2 = compute_trade_metrics(windows[1])
    assert m2["realized_pnl"] == -1.0
    assert m2["mae"] == 1.0


def test_bin_and_aggregate(tmp_path: Path):
    trades = make_synthetic_trade_sequence()
    windows = extract_trades_from_step_logs(trades)
    metrics = [compute_trade_metrics(w) for w in windows]
    summary = bin_and_aggregate(metrics, [0, 0.01, 0.03, 1.0])
    assert not summary.empty
    # Expect 2 bins: one for small action, one for large
    assert summary["trade_count"].sum() == 2
