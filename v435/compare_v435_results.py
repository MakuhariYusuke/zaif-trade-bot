#!/usr/bin/env python3
"""
Compare SAC v435.5 and v435.6 backtest results
"""

import json
from pathlib import Path


def main():
    print("📊 SAC v435.5 vs v435.6 Comparison")
    print("=" * 40)

    # Load results
    results_dir = Path("results")

    v435_5_results = results_dir / "sac_v435_5_backtest.json"
    v435_6_results = results_dir / "sac_v435_6_backtest.json"

    if not v435_5_results.exists() or not v435_6_results.exists():
        print("❌ Results files not found")
        return

    with open(v435_5_results, "r") as f:
        v435_5 = json.load(f)

    with open(v435_6_results, "r") as f:
        v435_6 = json.load(f)

    print("\n🎯 Performance Comparison:")
    print(f"{'Metric':<20} {'v435.5':<10} {'v435.6':<10} {'Difference':<12}")
    print("-" * 55)

    metrics = [
        ("Total Return %", "total_return_pct", ".2f"),
        ("Total Trades", "total_trades", "d"),
        ("Win Rate %", "win_rate_pct", ".1f"),
        ("Final Balance", "final_balance", ".2f"),
    ]

    for metric_name, key, fmt in metrics:
        v5_val = v435_5.get(key, 0)
        v6_val = v435_6.get(key, 0)
        diff = v5_val - v6_val if key != "final_balance" else v5_val - v6_val

        print(f"{metric_name:<20} {v5_val:<10{fmt}} {v6_val:<10{fmt}} {diff:<12{fmt}}")

    print("\n📋 Analysis:")
    print(
        f"• v435.5 (Micro penalty): {v435_5['total_trades']} trades, {v435_5['total_return_pct']:.2f}% return"
    )
    print(
        f"• v435.6 (Ensemble): {v435_6['total_trades']} trades, {v435_6['total_return_pct']:.2f}% return"
    )

    if v435_5["total_trades"] > v435_6["total_trades"]:
        print("• v435.5 shows more trading activity than v435.6")
    elif v435_6["total_trades"] > v435_5["total_trades"]:
        print("• v435.6 shows more trading activity than v435.5")
    else:
        print("• Both models show similar trading activity")

    print("\n💡 Recommendations:")
    if v435_5["total_trades"] > 0 and v435_6["total_trades"] == 0:
        print("• v435.5 shows promise - consider extending training for v435.6")
        print("• v435.6 may need longer training or reward function adjustments")
    elif v435_5["total_return_pct"] > v435_6["total_return_pct"]:
        print("• v435.5 outperforms v435.6 - micro penalty approach more effective")
    else:
        print("• Both models need further optimization")


if __name__ == "__main__":
    main()
