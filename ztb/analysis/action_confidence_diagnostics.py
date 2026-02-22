"""Action confidence diagnostics

Analyze trade logs (from SAC backtests) and decompose metrics by absolute action bins.
Produces CSV/JSON summary and prints per-bin statistics including:
- trade_count
- mean_realized_pnl
- median_realized_pnl
- win_rate
- mean_mae
- mean_mfe
- mean_duration_steps

Usage:
    python -m ztb.analysis.action_confidence_diagnostics --trade-log path/to/trades.json --price-csv path/to/price.csv

If only `--trade-log` is provided, it will analyze trades using step-level `pnl` that must be present in each trade entry.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ztb.io.json_io import read_json
from ztb.utils.file_utils import save_csv_data


@dataclass
class TradeWindow:
    entry_idx: int
    exit_idx: int
    entry_action: float
    steps: List[Dict[str, Any]]


def load_trade_log(path: str) -> List[Dict[str, Any]]:
    return read_json(path)


def extract_trades_from_step_logs(trades: List[Dict[str, Any]]) -> List[TradeWindow]:
    """Group step-level trade events into trade windows.

    This function assumes each trade event dict contains at least:
    - 'step': int
    - 'action': float (continuous action value)
    - 'price': float
    - 'position': float (position AFTER the step)
    - 'pnl': float (step pnl)

    We consider a trade entry when position changes from 0 to non-zero, and exit when position returns to 0.
    """
    windows: List[TradeWindow] = []
    current_window_steps: List[Dict[str, Any]] = []
    entry_idx = -1
    entry_action = 0.0

    prev_position = 0.0
    for idx, s in enumerate(trades):
        pos = float(s.get("position", 0.0))
        action = float(s.get("action", 0.0))

        # Detect entry: prev_position == 0 and pos != 0
        if prev_position == 0.0 and abs(pos) > 0.0:
            # Start new window
            current_window_steps = [s]
            entry_idx = idx
            entry_action = action
        elif prev_position != 0.0 and abs(pos) > 0.0:
            # Check for flip (sign change)
            if np.sign(prev_position) != np.sign(pos):
                # Flip detected: Close current window, Start new window
                if current_window_steps:
                    current_window_steps.append(s)
                    win = TradeWindow(
                        entry_idx=entry_idx,
                        exit_idx=idx,
                        entry_action=entry_action,
                        steps=current_window_steps.copy(),
                    )
                    windows.append(win)
                
                # Start new window
                # Note: The PnL in this step 's' is dominated by the closing of the previous position.
                # Ideally we would separate the entry fee for the new position, but we don't have that granularity.
                # We start the new window with this step to capture entry metadata, but we might want to zero the PnL
                # to avoid attributing the previous trade's result to this one.
                # For now, we'll include it but be aware of the artifact.
                # A better approach might be to set pnl=0 for the first step of a flip in the new window.
                s_new = s.copy()
                s_new['pnl'] = 0.0 # Zero out PnL for the entry step of the new trade to avoid contamination
                current_window_steps = [s_new]
                entry_idx = idx
                entry_action = action
            else:
                # Continue trade
                if current_window_steps is not None:
                    current_window_steps.append(s)
        elif prev_position != 0.0 and abs(pos) == 0.0:
            # Exit occurred; close window
            if current_window_steps:
                # include current step where pos==0 (if it contains pnl for close)
                current_window_steps.append(s)
                win = TradeWindow(
                    entry_idx=entry_idx,
                    exit_idx=idx,
                    entry_action=entry_action,
                    steps=current_window_steps.copy(),
                )
                windows.append(win)
            current_window_steps = []
            entry_idx = -1
            entry_action = 0.0
        prev_position = pos

    return windows


def compute_trade_metrics(window: TradeWindow) -> Dict[str, Any]:
    # cumulative pnl across steps
    # Prefer 'step_pnl' if available, otherwise fall back to 'pnl' (which might be total, so this is risky)
    pnls = [float(s.get("step_pnl", s.get("pnl", 0.0))) for s in window.steps]
    if not pnls:
        return {}
    cum = np.cumsum(pnls)
    realized_pnl = float(cum[-1])
    mae = float(abs(np.min(cum)))
    mfe = float(np.max(cum))
    duration = len(window.steps)
    entry_step_pnl = float(window.steps[0].get("pnl", 0.0))
    return {
        "entry_action": float(window.entry_action),
        "realized_pnl": realized_pnl,
        "mae": mae,
        "mfe": mfe,
        "duration": duration,
        "entry_step_pnl": entry_step_pnl,
    }


def bin_and_aggregate(trade_metrics: List[Dict[str, Any]], bins: List[float]) -> pd.DataFrame:
    df = pd.DataFrame(trade_metrics)
    if df.empty:
        return pd.DataFrame()

    df["abs_action"] = df["entry_action"].abs()
    labels = []
    # Create labels from bins
    for i in range(len(bins) - 1):
        labels.append(f"{bins[i]}-{bins[i+1]}")
    df["action_bin"] = pd.cut(df["abs_action"], bins=bins, labels=labels, include_lowest=True)

    agg = df.groupby("action_bin").agg(
        trade_count=("realized_pnl", "count"),
        mean_realized_pnl=("realized_pnl", "mean"),
        median_realized_pnl=("realized_pnl", "median"),
        win_rate=("realized_pnl", lambda x: np.mean(np.array(x) > 0) if len(x) > 0 else np.nan),
        mean_mae=("mae", "mean"),
        mean_mfe=("mfe", "mean"),
        mean_duration=("duration", "mean"),
        mean_entry_step_pnl=("entry_step_pnl", "mean"),
    )

    return agg.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-log", required=True, help="Path to trade log JSON")
    parser.add_argument(
        "--bins",
        nargs="*",
        type=float,
        help="Bin edges for abs(action). Example: 0 0.005 0.01 0.015 0.03 1.0",
    )
    parser.add_argument("--out-csv", default="action_confidence_summary.csv")
    args = parser.parse_args()

    trades = load_trade_log(args.trade_log)

    windows = extract_trades_from_step_logs(trades)
    metrics = [compute_trade_metrics(w) for w in windows if compute_trade_metrics(w)]

    if args.bins:
        bins = args.bins
    else:
        bins = [0.0, 0.005, 0.01, 0.015, 0.03, 1.0]

    summary = bin_and_aggregate(metrics, bins)
    save_csv_data(summary, args.out_csv, index=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
