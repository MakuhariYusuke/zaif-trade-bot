#!/usr/bin/env python3
"""
V442.8 Enhanced Balance Training Results Analysis
Analyzes the training results from v442.8 with enhanced balance mechanisms
"""

import json
from pathlib import Path

import numpy as np


def load_training_results():
    """Load the latest training results from v442.8"""
    results_dir = Path("backtest_results")

    # Look for v442.8 results
    v442_8_files = list(results_dir.glob("*v442_8*"))
    if not v442_8_files:
        print("No v442.8 results found")
        return None

    # Get the most recent file
    latest_file = max(v442_8_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_file}")

    with open(latest_file, "r") as f:
        return json.load(f)


def analyze_action_distribution(results):
    """Analyze the action distribution from training results"""
    if not results or "action_distribution" not in results:
        print("No action distribution data found")
        return None

    action_dist = results["action_distribution"]
    print("\n=== V442.8 Action Distribution Analysis ===")
    print(f"HOLD: {action_dist.get('HOLD', 0)*100:.1f}%")
    print(f"BUY:  {action_dist.get('BUY', 0)*100:.1f}%")
    print(f"SELL: {action_dist.get('SELL', 0)*100:.1f}%")

    # Calculate balance metrics
    buy_pct = action_dist.get("BUY", 0)
    sell_pct = action_dist.get("SELL", 0)
    hold_pct = action_dist.get("HOLD", 0)

    # Balance ratio (closer to 1.0 is better)
    balance_ratio = (
        min(buy_pct, sell_pct) / max(buy_pct, sell_pct)
        if max(buy_pct, sell_pct) > 0
        else 0
    )

    # Entropy (higher is better for balance)
    entropy = -sum(
        p * np.log(p + 1e-10) for p in [buy_pct, sell_pct, hold_pct] if p > 0
    )

    print(f"Balance ratio: {balance_ratio:.3f}")
    print(f"Entropy: {entropy:.3f}")
    # Compare with v442.6
    print("\n=== Comparison with v442.6 ===")
    print("v442.6: BUY 86.7%, SELL 7.1%, HOLD 6.2% (severe BUY bias)")
    print(
        f"v442.8: BUY {buy_pct*100:.1f}%, SELL {sell_pct*100:.1f}%, HOLD {hold_pct*100:.1f}%"
    )

    improvement = abs(buy_pct - sell_pct) < 0.8  # BUY bias was 0.8 difference
    print(f"Balance improved: {'Yes' if improvement else 'No'}")

    return {
        "balance_ratio": balance_ratio,
        "entropy": entropy,
        "action_distribution": action_dist,
        "improvement": improvement,
    }


def analyze_training_metrics(results):
    """Analyze training performance metrics"""
    if not results:
        return None

    print("\n=== Training Performance ===")
    print(f"Total timesteps: {results.get('total_timesteps', 'N/A')}")
    print(f"Training time: {results.get('training_time', 'N/A'):.1f}s")
    print(f"Steps per second: {results.get('steps_per_second', 'N/A'):.1f}")
    print(f"Final reward: {results.get('final_reward', 'N/A')}")


def generate_recommendations(analysis):
    """Generate recommendations based on analysis"""
    if not analysis:
        return

    print("\n=== Recommendations ===")

    balance_ratio = analysis["balance_ratio"]
    entropy = analysis["entropy"]
    action_dist = analysis["action_distribution"]

    buy_pct = action_dist.get("BUY", 0)
    sell_pct = action_dist.get("SELL", 0)

    if balance_ratio < 0.1:  # Very imbalanced
        if buy_pct > sell_pct:
            print(
                "• BUY bias still present - consider increasing entropy_regularization or adjusting action_balance_target"
            )
        else:
            print(
                "• SELL bias detected - consider decreasing consistency_penalty or entropy_regularization"
            )
            print("• Current parameters may be too aggressive for SELL actions")

    elif balance_ratio < 0.5:  # Moderately imbalanced
        print("• Moderate imbalance - fine-tune behavior_optimization parameters")
        print("• Consider adjusting action_smoothing to reduce action consistency")

    else:  # Well balanced
        print("• Good balance achieved - validate with longer training")

    if entropy < 0.8:
        print(
            "• Low entropy indicates predictable actions - consider increasing exploration"
        )

    print(
        "• Next step: Run extended training (10k-50k timesteps) to validate stability"
    )
    print("• Consider A/B testing different parameter combinations")


def main():
    """Main analysis function"""
    print("V442.8 Enhanced Balance Training Results Analysis")
    print("=" * 50)

    # Load results
    results = load_training_results()
    if not results:
        print("No results to analyze")
        return

    # Analyze action distribution
    analysis = analyze_action_distribution(results)

    # Analyze training metrics
    analyze_training_metrics(results)

    # Generate recommendations
    generate_recommendations(analysis)

    print("\n" + "=" * 50)
    print("Analysis complete")


if __name__ == "__main__":
    main()
