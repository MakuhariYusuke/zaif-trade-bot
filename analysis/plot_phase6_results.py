import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_results():
    results_path = "backtest_results/phase6_hft_backtest.json"
    if not os.path.exists(results_path):
        print(f"File not found: {results_path}")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    portfolio_history = data["portfolio_history"]
    price_history = data["price_history"]
    timestamps = data.get("timestamps", [])

    # Ensure lengths match
    min_len = min(len(portfolio_history), len(price_history))
    portfolio_history = portfolio_history[:min_len]
    price_history = price_history[:min_len]

    # Create DataFrame
    df = pd.DataFrame({"Portfolio": portfolio_history, "Price": price_history})

    if len(timestamps) >= min_len:
        df.index = pd.to_datetime(timestamps[:min_len])

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = "tab:blue"
    ax1.set_xlabel("Time/Step")
    ax1.set_ylabel("Portfolio Value (JPY)", color=color)
    ax1.plot(
        df.index if len(timestamps) >= min_len else df.index,
        df["Portfolio"],
        color=color,
        label="Portfolio",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:red"
    ax2.set_ylabel(
        "BTC Price (JPY)", color=color
    )  # we already handled the x-label with ax1
    ax2.plot(
        df.index if len(timestamps) >= min_len else df.index,
        df["Price"],
        color=color,
        linestyle="--",
        label="Price",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Phase 6 Backtest: Portfolio vs Price")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    output_path = "backtest_results/phase6_hft_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_results()
