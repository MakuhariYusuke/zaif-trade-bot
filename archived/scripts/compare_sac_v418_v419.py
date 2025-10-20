#!/usr/bin/env python3
"""
Statistical comparison of SAC v418 vs SAC v419 models.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_results(file_path: str) -> Dict[str, Any]:
    """Load paper trading results from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate additional metrics from results."""
    initial_portfolio = results["initial_portfolio"]
    final_portfolio = results["final_portfolio"]
    total_trades = results["total_trades"]
    action_dist = results["action_distribution"]

    # Basic metrics
    total_return = results["total_return_pct"]
    total_pnl = results["total_pnl"]

    # Trading intensity
    trades_per_step = total_trades / results["total_steps"]

    # Action balance metrics
    total_actions = sum(action_dist.values())
    buy_ratio = action_dist.get(1, 0) / total_actions if total_actions > 0 else 0
    sell_ratio = action_dist.get(2, 0) / total_actions if total_actions > 0 else 0
    hold_ratio = action_dist.get(0, 0) / total_actions if total_actions > 0 else 0

    # Action balance score (closer to 0.5 is better balance)
    action_balance = abs(buy_ratio - sell_ratio)

    return {
        "total_return_pct": total_return,
        "total_pnl": total_pnl,
        "trades_per_step": trades_per_step,
        "buy_ratio": buy_ratio,
        "sell_ratio": sell_ratio,
        "hold_ratio": hold_ratio,
        "action_balance_score": action_balance,
        "total_trades": total_trades,
    }


def compare_models(
    v418_results: Dict[str, Any], v419_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare two models statistically."""
    v418_metrics = calculate_metrics(v418_results)
    v419_metrics = calculate_metrics(v419_results)

    # Performance comparison
    return_diff = v418_metrics["total_return_pct"] - v419_metrics["total_return_pct"]
    pnl_diff = v418_metrics["total_pnl"] - v419_metrics["total_pnl"]

    # Trading behavior comparison
    trade_intensity_diff = (
        v418_metrics["trades_per_step"] - v419_metrics["trades_per_step"]
    )
    balance_diff = (
        v418_metrics["action_balance_score"] - v419_metrics["action_balance_score"]
    )

    return {
        "performance_comparison": {
            "v418_return": v418_metrics["total_return_pct"],
            "v419_return": v419_metrics["total_return_pct"],
            "return_difference": return_diff,
            "return_improvement_pct": (
                return_diff / abs(v419_metrics["total_return_pct"])
            )
            * 100
            if v419_metrics["total_return_pct"] != 0
            else float("inf"),
            "v418_pnl": v418_metrics["total_pnl"],
            "v419_pnl": v419_metrics["total_pnl"],
            "pnl_difference": pnl_diff,
        },
        "trading_behavior": {
            "v418_trades_per_step": v418_metrics["trades_per_step"],
            "v419_trades_per_step": v419_metrics["trades_per_step"],
            "trade_intensity_difference": trade_intensity_diff,
            "v418_buy_ratio": v418_metrics["buy_ratio"],
            "v419_buy_ratio": v419_metrics["buy_ratio"],
            "v418_sell_ratio": v418_metrics["sell_ratio"],
            "v419_sell_ratio": v419_metrics["sell_ratio"],
            "v418_hold_ratio": v418_metrics["hold_ratio"],
            "v419_hold_ratio": v419_metrics["hold_ratio"],
            "v418_balance_score": v418_metrics["action_balance_score"],
            "v419_balance_score": v419_metrics["action_balance_score"],
            "balance_improvement": balance_diff,
        },
        "recommendation": "v418"
        if v418_metrics["total_return_pct"] > v419_metrics["total_return_pct"]
        else "v419",
    }


def main():
    """Main comparison function."""
    # Load results
    v418_path = "results/paper_trade_v418_balanced.json"
    v419_path = "results/paper_trade_v419_equalized.json"

    try:
        v418_results = load_results(v418_path)
        v419_results = load_results(v419_path)

        # Calculate comparison
        comparison = compare_models(v418_results, v419_results)

        # Print results
        print("=" * 80)
        print("STATISTICAL COMPARISON: SAC v418 vs SAC v419")
        print("=" * 80)

        print("\n📊 PERFORMANCE COMPARISON:")
        print(
            f"  SAC v418 Return: {comparison['performance_comparison']['v418_return']:.2f}%"
        )
        print(
            f"  SAC v419 Return: {comparison['performance_comparison']['v419_return']:.2f}%"
        )
        print(
            f"  Return Difference: {comparison['performance_comparison']['return_difference']:.2f}%"
        )
        if comparison["performance_comparison"]["return_improvement_pct"] != float(
            "inf"
        ):
            print(
                f"  Return Improvement: {comparison['performance_comparison']['return_improvement_pct']:.1f}%"
            )
        print(f"  SAC v418 PnL: {comparison['performance_comparison']['v418_pnl']:.2f}")
        print(f"  SAC v419 PnL: {comparison['performance_comparison']['v419_pnl']:.2f}")
        print(
            f"  PnL Difference: {comparison['performance_comparison']['pnl_difference']:.2f}"
        )
        print("\n🎯 TRADING BEHAVIOR:")
        print(
            f"  SAC v418 Trades/Step: {comparison['trading_behavior']['v418_trades_per_step']:.4f}"
        )
        print(
            f"  SAC v419 Trades/Step: {comparison['trading_behavior']['v419_trades_per_step']:.4f}"
        )
        print(
            f"  Trade Intensity Difference: {comparison['trading_behavior']['trade_intensity_difference']:.4f}"
        )
        print(
            f"  SAC v418 Buy Ratio: {comparison['trading_behavior']['v418_buy_ratio']:.3f}"
        )
        print(
            f"  SAC v419 Buy Ratio: {comparison['trading_behavior']['v419_buy_ratio']:.3f}"
        )
        print(
            f"  SAC v418 Sell Ratio: {comparison['trading_behavior']['v418_sell_ratio']:.3f}"
        )
        print(
            f"  SAC v419 Sell Ratio: {comparison['trading_behavior']['v419_sell_ratio']:.3f}"
        )
        print(
            f"  SAC v418 Hold Ratio: {comparison['trading_behavior']['v418_hold_ratio']:.3f}"
        )
        print(
            f"  SAC v419 Hold Ratio: {comparison['trading_behavior']['v419_hold_ratio']:.3f}"
        )
        print(
            f"  SAC v418 Balance Score: {comparison['trading_behavior']['v418_balance_score']:.4f}"
        )
        print(
            f"  SAC v419 Balance Score: {comparison['trading_behavior']['v419_balance_score']:.4f}"
        )
        print(
            f"  Balance Improvement: {comparison['trading_behavior']['balance_improvement']:.4f}"
        )
        print("\n🏆 RECOMMENDATION:")
        print(
            f"Based on paper trading results, {comparison['recommendation']} shows superior performance"
        )

        # Save comparison results
        output_path = "results/sac_v418_v419_comparison.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        print(f"\nComparison results saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: Results file not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during comparison: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
