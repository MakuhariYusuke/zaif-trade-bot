import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_phase6():
    project_root = Path(__file__).resolve().parents[1]
    results_path = project_root / "backtest_results" / "phase6_hft_backtest.json"

    with open(results_path, "r") as f:
        data = json.load(f)

    actions = data["actions"]
    portfolio = data["portfolio_history"]
    trade_pnls = data["trade_pnls"]

    # Action distribution
    action_counts = pd.Series(actions).value_counts()
    print("Action Distribution:")
    print(action_counts)
    print(action_counts / len(actions) * 100)

    # Trade analysis
    n_trades = len(trade_pnls)
    print(f"\nNumber of Realized Trades (PnL events): {n_trades}")

    if n_trades > 0:
        pnls = np.array(trade_pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        win_rate = len(wins) / n_trades * 100
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win: {avg_win:.2f}")
        print(f"Average Loss: {avg_loss:.2f}")
        print(f"Total PnL: {np.sum(pnls):.2f}")
    else:
        print(
            "No trades realized during backtest (except maybe final close if recorded)."
        )

    # Plot portfolio
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio)
    plt.title("Phase 6 HFT Portfolio Value")
    plt.xlabel("Step")
    plt.ylabel("Value (JPY)")
    plt.grid(True)
    plt.savefig(project_root / "analysis_results" / "phase6_hft_portfolio.png")
    print("Portfolio plot saved to analysis_results/phase6_hft_portfolio.png")


if __name__ == "__main__":
    analyze_phase6()
