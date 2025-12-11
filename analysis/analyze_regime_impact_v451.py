import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def analyze_regime_impact():
    results_dir = os.path.join(project_root, "backtest_results", "v451")
    csv_path = os.path.join(results_dir, "backtest_results.csv")

    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        return

    print(f"Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Clean up regime names (remove "RegimeType.")
    df["regime"] = df["regime"].apply(
        lambda x: x.replace("RegimeType.", "") if isinstance(x, str) else x
    )

    print("\n=== Regime Performance Analysis ===")

    # Group by regime
    regime_stats = df.groupby("regime").agg(
        {
            "pnl": ["count", "sum", "mean", "std"],
            "action_type": lambda x: x.value_counts().index[0]
            if len(x) > 0
            else "None",
        }
    )

    # Flatten columns
    regime_stats.columns = [
        "count",
        "total_pnl",
        "avg_pnl",
        "std_pnl",
        "dominant_action",
    ]
    regime_stats["win_rate"] = df.groupby("regime")["pnl"].apply(
        lambda x: (x > 0).mean()
    )

    # Sort by Total PnL
    regime_stats = regime_stats.sort_values("total_pnl", ascending=False)

    print(regime_stats)

    # Save stats
    regime_stats.to_csv(os.path.join(results_dir, "regime_performance.csv"))

    # Plotting
    plt.figure(figsize=(14, 8))

    # 1. PnL by Regime
    plt.subplot(2, 1, 1)
    sns.barplot(x=regime_stats.index, y=regime_stats["total_pnl"], palette="viridis")
    plt.title("Total PnL by Market Regime")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total PnL (JPY)")
    plt.grid(True, alpha=0.3)

    # 2. Win Rate by Regime
    plt.subplot(2, 1, 2)
    sns.barplot(x=regime_stats.index, y=regime_stats["win_rate"], palette="coolwarm")
    plt.title("Win Rate by Market Regime")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Win Rate")
    plt.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "regime_performance.png"))
    print(f"\nPlots saved to {results_dir}")

    # Action distribution by regime
    print("\n=== Action Distribution by Regime ===")
    action_dist = pd.crosstab(df["regime"], df["action_type"], normalize="index")
    print(action_dist)

    # Plot Action Distribution
    plt.figure(figsize=(14, 6))
    action_dist.plot(kind="bar", stacked=True, colormap="coolwarm", figsize=(14, 6))
    plt.title("Action Distribution by Regime")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Proportion")
    plt.legend(title="Action")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "regime_action_dist.png"))


if __name__ == "__main__":
    analyze_regime_impact()
