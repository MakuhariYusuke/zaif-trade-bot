#!/usr/bin/env python3

"""Compute risk metrics (Sharpe ratio, max drawdown) for reward function variants."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, cast


def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Compute annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0:
        return 0.0

    sharpe = cast(float, excess_returns.mean() / excess_returns.std() * np.sqrt(252))
    return sharpe


def compute_max_drawdown(portfolio_values: pd.Series) -> float:
    """Compute maximum drawdown."""
    if len(portfolio_values) < 2:
        return 0.0

    peak = portfolio_values.expanding().max()
    drawdown = (portfolio_values - peak) / peak
    max_dd = drawdown.min()
    return cast(float, abs(max_dd))


def analyze_risk_metrics() -> None:
    """Analyze risk metrics from action distribution data."""
    print("🔍 Analyzing risk metrics from action distribution data...")

    # Load action distribution results
    action_dist_path = Path("action_distribution_summary.md")
    if not action_dist_path.exists():
        print("❌ Action distribution data not found. Run analyze_action_distribution.py first.")
        return

    # Load detailed results
    results_path = Path("action_distribution_comparison.json")
    if not results_path.exists():
        print("❌ Detailed results not found.")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    print("\n📊 Risk Metrics Analysis:")
    print("=" * 50)

    for config_name, config_data in results.items():
        if "backtest_results" not in config_data:
            continue

        backtest_data = config_data["backtest_results"]
        if "returns" not in backtest_data or not backtest_data["returns"]:
            continue

        returns = pd.Series(backtest_data["returns"])
        portfolio_values = pd.Series(backtest_data.get("portfolio_values", []))

        if len(returns) < 2:
            continue

        sharpe = compute_sharpe_ratio(returns)
        max_dd = compute_max_drawdown(portfolio_values) if len(portfolio_values) > 0 else 0.0

        print(f"\n🔹 {config_name}:")
        print(f"   Sharpe Ratio: {sharpe:.3f}")
        print(f"   Max Drawdown: {max_dd:.3f}")
        print(f"   Total Returns: {returns.sum():.3f}")
        print(f"   Volatility: {returns.std():.3f}")


def main() -> None:
    """Main entry point."""
    analyze_risk_metrics()


if __name__ == "__main__":
    main()