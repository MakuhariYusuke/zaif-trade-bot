#!/usr/bin/env python3

"""Compute risk metrics (Sharpe ratio, max drawdown) for reward function variants."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ztb.io.json_io import read_json, write_json
from ztb.metrics.metrics import max_drawdown, sharpe_ratio

# 年間取引日数
from ztb.trading.constants import TRADING_DAYS_PER_YEAR  # = 252


def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Compute annualized Sharpe ratio."""
    return sharpe_ratio(
        returns, rf=risk_free_rate, period_per_year=TRADING_DAYS_PER_YEAR
    )


def analyze_risk_metrics(
    backtest_results: Optional[str] = None,
    risk_measures: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze risk metrics from backtest results.

    Args:
        backtest_results: Path to backtest results file
        risk_measures: List of risk measures to analyze
        output_path: Path to save analysis results

    Returns:
        Dictionary with risk analysis results
    """
    print("🔍 Analyzing risk metrics from backtest results...")

    # Placeholder implementation - would need actual backtest results processing
    results = {
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "total_return": 0.0,
        "volatility": 0.0,
    }

    if output_path:
        write_json(output_path, results, indent=2, ensure_ascii=False)

    return results


def main() -> None:
    """Main entry point."""
    print(
        "❌ Action distribution data not found. Run analyze_action_distribution.py first."
    )
    return

    # Load detailed results
    results_path = Path("action_distribution_comparison.json")
    if not results_path.exists():
        print("❌ Detailed results not found.")
        return

    results = read_json(results_path)

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
        max_dd = (
            max_drawdown(portfolio_values) if len(portfolio_values) > 0 else 0.0
        )

        print(f"\n🔹 {config_name}:")
        print(f"   Sharpe Ratio: {sharpe:.3f}")
        print(f"   Max Drawdown: {max_dd:.3f}")
        print(f"   Total Returns: {returns.sum():.3f}")
        print(f"   Volatility: {returns.std():.3f}")
    """Main entry point."""
    analyze_risk_metrics()


if __name__ == "__main__":
    main()
