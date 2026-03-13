#!/usr/bin/env python3
"""
Comparison script for SAC v435.5 and v435.6 models
v435.5: Micro frequency penalty scalping
v435.6: Ensemble majority voting system
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_backtest_results(file_path: str) -> dict:
    """Load backtest results from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Results file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"⚠️  Invalid JSON in {file_path}: {e}")
        return {}


def main():
    print("🔍 SAC v435.5 & v435.6 Models Comparison")
    print("=" * 60)

    # Define result file paths
    results_files = {
        "v435.5": "results/backtest_v435_5_results.json",
        "v435.6": "results/backtest_v435_6_results.json",
    }

    # Load results
    results = {}
    for model, file_path in results_files.items():
        results[model] = load_backtest_results(file_path)

    # Display comparison
    print("\n📊 Model Comparison:")
    print("-" * 40)

    for model in ["v435.5", "v435.6"]:
        data = results[model]
        if data:
            print(f"\n{model.upper()}:")
            print(f"  Total Return: {data.get('total_return', 'N/A')}")
            print(f"  Total Trades: {data.get('total_trades', 'N/A')}")
            print(f"  Win Rate: {data.get('win_rate', 'N/A')}")
            print(f"  Max Drawdown: {data.get('max_drawdown', 'N/A')}")
            print(f"  Sharpe Ratio: {data.get('sharpe_ratio', 'N/A')}")
            if model == "v435.6":
                print(
                    f"  Ensemble Consensus Rate: {data.get('ensemble_consensus_rate', 'N/A')}"
                )
        else:
            print(f"\n{model.upper()}: No results available")

    # Determine best model
    best_model = None
    best_return = float("-inf")

    for model, data in results.items():
        if data and "total_return" in data:
            return_val = data["total_return"]
            if return_val > best_return:
                best_return = return_val
                best_model = model

    print("\n" + "=" * 60)
    if best_model:
        print(f"🏆 BEST PERFORMING MODEL: {best_model.upper()}")
        print(f"   Best Total Return: {best_return}")
    else:
        print("❌ No valid results to compare")

    print("\n💡 Model Descriptions:")
    print("v435.5: Micro frequency penalty scalping (penalty = 0.001)")
    print("v435.6: Ensemble majority voting system (combines v435.3, v435.4, v435.5)")

    print("\n🚀 Recommendations:")
    if best_model == "v435.5":
        print("- Deploy v435.5 for controlled scalping with profitability")
        print("- Consider v435.6 for more robust decision making")
    elif best_model == "v435.6":
        print("- Deploy v435.6 for ensemble-based trading decisions")
        print("- Use v435.5 as fallback when ensemble consensus is low")
    else:
        print("- Run backtests first to generate comparison data")

    print("=" * 60)


if __name__ == "__main__":
    main()
