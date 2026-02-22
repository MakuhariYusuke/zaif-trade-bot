#!/usr/bin/env python3
"""
Run market regime evaluation.

Compares RL agent performance against baselines across different market regimes.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ztb.io.json_io import read_json, write_json
from ztb.evaluation.baseline_comparison import get_baseline_comparison_engine
from ztb.analysis.regime.regime_eval import RegimeEvaluator
from ztb.evaluation.unified_evaluation import EvaluationType, UnifiedEvaluator
from ztb.io.data_loader import DataLoader


def load_trade_data(trade_log_path: Optional[str]) -> list[Any]:
    """Load trade log from JSON file."""
    if trade_log_path is None:
        return []
    return read_json(trade_log_path)  # type: ignore


def load_price_data(price_data_path: str) -> pd.DataFrame:
    """Load price data from CSV file."""
    df = DataLoader.load_csv_strict(price_data_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    return df


def run_regime_evaluation(
    price_data_path: str, trade_log_path: Optional[str], output_dir: str = "reports"
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

    unified = UnifiedEvaluator(
        config={
            "regime_trade_log": trade_log,
            "regime_price_data_path": price_data_path,
        }
    )
    evaluation = unified.evaluate_model(
        model_path="regime",
        data_path=price_data_path,
        evaluation_type=EvaluationType.REGIME,
    )
    regime_results_raw = evaluation.market_regime_analysis.get("overall_raw", {})
    regime_results = evaluation.market_regime_analysis.get("overall", {})

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
    regime_results_raw["baseline_comparison"] = regime_evaluator._compare_baselines(
        regime_results_raw, baseline_strategies
    )
    regime_results["baseline_comparison"] = regime_results_raw["baseline_comparison"]

    # Generate reports
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # JSON report
    json_path = output_path / "regime_report.json"
    # Convert enum keys to strings for JSON serialization
    serializable_results = {}
    for key, value in regime_results.items():
        if isinstance(value, dict) and "metrics" in value:
            serializable_results[key] = value.copy()
            metrics_value = value["metrics"]
            serializable_results[key]["metrics"] = (
                metrics_value.__dict__
                if hasattr(metrics_value, "__dict__")
                else metrics_value
            )
        else:
            serializable_results[key] = value
    write_json(json_path, serializable_results, indent=2, default=str)

    # Markdown report
    md_path = output_path / "regime_report.md"
    md_report = regime_evaluator.generate_report(regime_results_raw, str(md_path))

    print("Regime evaluation completed!")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")

    return regime_results


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run market regime evaluation")
    parser.add_argument(
        "--price-data", required=True, help="Path to price data CSV file"
    )
    parser.add_argument(
        "--trade-log", help="Path to trade log JSON file"
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

    if args.trade_log and not Path(args.trade_log).exists():
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
