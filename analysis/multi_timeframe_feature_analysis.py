#!/usr/bin/env python3
"""
Analysis script for multi-timeframe feature comparison

Analyzes training results and existing backtest data to compare
multi-timeframe enabled vs disabled configurations.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ztb.utils.analysis_formatters import create_result_summary


def load_training_results():
    """Load training comparison results."""
    training_file = "multi_timeframe_comparison_10k_results.json"
    if not os.path.exists(training_file):
        print(f"❌ Training results not found: {training_file}")
        return None

    with open(training_file, "r") as f:
        return json.load(f)


def load_existing_backtests():
    """Load existing backtest results."""
    backtests = {}

    # Load results from results/v435/ directory
    results_dir = Path("results/v435")
    if results_dir.exists():
        for json_file in results_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    model_name = json_file.stem
                    backtests[model_name] = data
                    print(f"✅ Loaded backtest: {model_name}")
            except Exception as e:
                print(f"❌ Failed to load {json_file}: {e}")

    # Load from backtest_results/ directory
    backtest_dir = Path("backtest_results")
    if backtest_dir.exists():
        for json_file in backtest_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    model_name = f"backtest_{json_file.stem}"
                    backtests[model_name] = data
                    print(f"✅ Loaded backtest: {model_name}")
            except Exception as e:
                print(f"❌ Failed to load {json_file}: {e}")

    return backtests


def analyze_training_performance(training_results):
    """Analyze training performance."""
    if not training_results:
        return {}

    analysis = {}

    for config_name, result in training_results.items():
        if result.get("success", False):
            analysis[config_name] = {
                "training_success": True,
                "timesteps": result.get("timesteps", 0),
                "completion_time": "N/A",  # Not available in current results
                "status": "completed",
            }
        else:
            analysis[config_name] = {
                "training_success": False,
                "error": result.get("error", "Unknown error"),
                "status": "failed",
            }

    return analysis


def analyze_backtest_performance(backtests):
    """Analyze backtest performance metrics."""
    analysis = {}

    for model_name, data in backtests.items():
        # Extract metrics
        total_return = data.get("total_return_pct", 0)
        win_rate = data.get("win_rate_pct", 0)
        total_trades = data.get("total_trades", 0)
        final_balance = data.get("final_balance", 10000)

        analysis[model_name] = {
            "total_return_pct": total_return,
            "win_rate_pct": win_rate,
            "total_trades": total_trades,
            "final_balance": final_balance,
            "profit_loss": final_balance - 10000,
        }

    return analysis


def compare_configurations(training_analysis, backtest_analysis):
    """Compare multi-timeframe enabled vs disabled configurations."""
    comparison = {
        "training_comparison": training_analysis,
        "backtest_comparison": backtest_analysis,
        "insights": [],
    }

    # Training insights
    successful_configs = [
        k for k, v in training_analysis.items() if v.get("training_success", False)
    ]
    comparison["insights"].append(
        f"Training completed successfully for {len(successful_configs)} configurations"
    )

    # Backtest insights
    if backtest_analysis:
        returns = [v.get("total_return_pct", 0) for v in backtest_analysis.values()]
        win_rates = [v.get("win_rate_pct", 0) for v in backtest_analysis.values()]

        if returns:
            avg_return = np.mean(returns)
            max_return = max(returns)
            min_return = min(returns)

            comparison["insights"].extend(
                [
                    f"Average total return across models: {avg_return:.2f}%",
                    f"Best performing model return: {max_return:.2f}%",
                    f"Worst performing model return: {min_return:.2f}%",
                ]
            )

        if win_rates:
            avg_win_rate = np.mean(win_rates)
            comparison["insights"].append(
                f"Average win rate across models: {avg_win_rate:.2f}%"
            )

    return comparison


def generate_report(comparison):
    """Generate comprehensive analysis report."""
    report = {
        "title": "Multi-Timeframe Feature Impact Analysis",
        "timestamp": pd.Timestamp.now().isoformat(),
        "summary": {
            "objective": "Compare SAC v435 performance with and without multi-timeframe features",
            "methodology": "Training comparison (10k steps) + existing backtest analysis",
            "configurations_tested": list(
                comparison.get("training_comparison", {}).keys()
            ),
        },
        "results": comparison,
        "conclusions": [],
    }

    # Generate conclusions
    training_comp = comparison.get("training_comparison", {})
    backtest_comp = comparison.get("backtest_comparison", {})

    # Training conclusions
    successful_training = sum(
        1 for v in training_comp.values() if v.get("training_success", False)
    )
    total_training = len(training_comp)

    if successful_training == total_training:
        report["conclusions"].append(
            "✅ All configurations trained successfully for 10,000 steps"
        )
    else:
        report["conclusions"].append(
            f"⚠️ {successful_training}/{total_training} configurations trained successfully"
        )

    # Backtest conclusions
    if backtest_comp:
        returns = [v.get("total_return_pct", 0) for v in backtest_comp.values()]
        if returns:
            best_return = max(returns)
            report["conclusions"].append(f"📈 Best backtest return: {best_return:.2f}%")

            profitable_models = sum(1 for r in returns if r > 0)
            report["conclusions"].append(
                f"💰 {profitable_models} out of {len(returns)} models showed positive returns"
            )

    # Multi-timeframe impact assessment
    report["conclusions"].append(
        "🔍 Multi-timeframe feature impact requires further analysis with proper backtesting"
    )

    return report


def main():
    print("🔍 Analyzing multi-timeframe feature impact...")

    # Load data
    training_results = load_training_results()
    backtests = load_existing_backtests()

    # Analyze performance
    training_analysis = analyze_training_performance(training_results)
    backtest_analysis = analyze_backtest_performance(backtests)

    # Compare configurations
    comparison = compare_configurations(training_analysis, backtest_analysis)

    # Generate report
    report = generate_report(comparison)

    # Save report
    output_file = "multi_timeframe_feature_analysis_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"✅ Analysis complete. Report saved to {output_file}")

    # Print summary with structured format
    print("\n📊 Analysis Summary:")
    for insight in comparison.get("insights", []):
        print(f"  • {insight}")

    # 主要指標の構造化表示
    backtest_comp = comparison.get("backtest_comparison", {})
    if backtest_comp:
        returns = [v.get("total_return_pct", 0) for v in backtest_comp.values()]
        if returns:
            summary_metrics = {
                "best_return_pct": max(returns),
                "profitable_models": sum(1 for r in returns if r > 0),
                "total_models": len(returns),
                "avg_return_pct": sum(returns) / len(returns)
            }
            print(f"\n📈 Key Metrics:\n{create_result_summary(summary_metrics)}")

    print("\n📋 Conclusions:")
    for conclusion in report.get("conclusions", []):
        print(f"  • {conclusion}")


if __name__ == "__main__":
    main()
