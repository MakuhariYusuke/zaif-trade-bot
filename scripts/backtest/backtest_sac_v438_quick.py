"""Minimal shim for `backtest_sac_v438_quick` used by tests.

This file contains minimal implementations of the functions the tests patch.
They are intentionally lightweight and suitable for unit tests that patch internals.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd


def calculate_backtest_summary(results_df: pd.DataFrame, portfolio_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
    if results_df.empty:
        return {}
    return {
        "total_episodes": len(results_df),
        "avg_total_reward": float(results_df["total_reward"].mean()),
        "total_trades_all_episodes": int(trades_df.shape[0]),
        "best_episode_reward": float(results_df["total_reward"].max()),
        "worst_episode_reward": float(results_df["total_reward"].min()),
    }


def backtest_sac_v438_quick(model_path: str, data_path: str, n_episodes: int = 10) -> Optional[Dict[str, Any]]:
    # A highly simplified flow: ensure files exist, otherwise return None
    if not Path(model_path).exists():
        return None
    if not Path(data_path).exists():
        return None
    # Simulate a trivial successful result
    return {"total_episodes": n_episodes, "avg_total_reward": 0.0}


def run_quick_backtest(*args, **kwargs):
    return backtest_sac_v438_quick(*args, **kwargs)


__all__ = ["calculate_backtest_summary", "backtest_sac_v438_quick", "run_quick_backtest"]
