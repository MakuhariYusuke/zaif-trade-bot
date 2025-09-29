#!/usr/bin/env python3
"""
Run market regime evaluation.

Compares RL agent performance against baselines across different market regimes.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ztb.evaluation.baseline_comparison import get_baseline_comparison_engine
from ztb.evaluation.regime_eval import RegimeEvaluator


def load_trade_data(trade_log_path: str) -> list:
    """Load trade log from JSON file."""
    with open(trade_log_path, "r") as f:
        return json.load(f)


def load_price_data(price_data_path: str) -> pd.DataFrame:
    """Load price data from CSV file."""
    df = pd.read_csv(price_data_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    return df


def run_regime_evaluation(
    price_data_path: str, trade_log_path: str, output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Run regime evaluation and generate reports.

    Args:
        price_data_path: Path to price data CSV
        trade_log_path: Path to trade log JSON
        output_dir: Output directory for reports

    Returns:
        Evaluation results
    """
    # Load data
    price_data = load_price_data(price_data_path)
    trade_log = load_trade_data(trade_log_path)

    # Initialize evaluators
    regime_evaluator = RegimeEvaluator()
    baseline_engine = get_baseline_comparison_engine()

    # Evaluate RL agent performance across regimes
    regime_results = regime_evaluator.evaluate_performance(price_data, trade_log)

    # Generate baseline comparisons for each regime
    baseline_strategies = {}

    # For each regime, run baseline strategies on that segment
    for regime_name in ["bull", "bear", "sideways"]:
        if regime_name not in regime_results:
            continue

        # Get price data segments for this regime
        # This is a simplified version - in practice you'd need to extract regime segments
        regime_price_data = price_data  # Simplified: use all data

        # Run baseline strategies
        buy_hold_result = baseline_engine.strategies["buy_hold"].evaluate(
            regime_price_data
        )
        sma_result = baseline_engine.strategies["sma_crossover"].evaluate(
            regime_price_data
        )

        baseline_strategies[regime_name] = {
            "buy_hold": {
                "total_return": buy_hold_result.total_return,
                "sharpe_ratio": buy_hold_result.sharpe_ratio,
                "win_rate": buy_hold_result.win_rate,
            },
            "sma_crossover": {
                "total_return": sma_result.total_return,
                "sharpe_ratio": sma_result.sharpe_ratio,
                "win_rate": sma_result.win_rate,
            },
        }

    # Add baseline comparison to results
    regime_results["baseline_comparison"] = regime_evaluator._compare_baselines(
        regime_results, baseline_strategies
    )

    # Generate reports
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # JSON report
    json_path = output_path / "regime_report.json"
    with open(json_path, "w") as f:
        # Convert enum keys to strings for JSON serialization
        serializable_results = {}
        for key, value in regime_results.items():
            if isinstance(value, dict) and "metrics" in value:
                serializable_results[key] = value.copy()
                serializable_results[key]["metrics"] = value["metrics"].__dict__
            else:
                serializable_results[key] = value
        json.dump(serializable_results, f, indent=2, default=str)

    # Markdown report
    md_path = output_path / "regime_report.md"
    md_report = regime_evaluator.generate_report(regime_results, str(md_path))

    print(f"Regime evaluation completed!")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")

    return regime_results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run market regime evaluation")
    parser.add_argument(
        "--price-data", required=True, help="Path to price data CSV file"
    )
    parser.add_argument(
        "--trade-log", required=True, help="Path to trade log JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for reports (default: reports)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.price_data).exists():
        print(f"Error: Price data file not found: {args.price_data}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.trade_log).exists():
        print(f"Error: Trade log file not found: {args.trade_log}", file=sys.stderr)
        sys.exit(1)

    try:
        results = run_regime_evaluation(
            args.price_data, args.trade_log, args.output_dir
        )

        # Print summary
        print("\nRegime Summary:")
        for regime_name, data in results.items():
            if regime_name == "baseline_comparison":
                continue
            if "metrics" in data:
                metrics = data["metrics"]
                print(
                    f"{regime_name.title()}: Return={metrics.total_return:.4f}, "
                    f"Sharpe={metrics.sharpe_ratio:.4f}, WinRate={metrics.win_rate:.4f}"
                )

    except Exception as e:
        print(f"Regime evaluation failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
