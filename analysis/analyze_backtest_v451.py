import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_v451_results():
    # Paths
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "backtest_results" / "v451"
    csv_path = results_dir / "backtest_results.csv"
    json_path = results_dir / "backtest_results.json"

    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    print(f"Loaded {len(df)} steps of backtest data.")
    print(f"First Portfolio Value: {df['portfolio_value'].iloc[0]}")
    print(df.head())
    print(df.tail())

    # 1. Performance Metrics
    initial_balance = (
        df["portfolio_value"].iloc[0] - df["pnl"].iloc[0]
    )  # Approximate if not stored
    # Better to load from JSON if available
    if json_path.exists():
        with open(json_path, "r") as f:
            summary = json.load(f)
            initial_balance = summary.get("initial_balance", 10000.0)

    final_balance = df["portfolio_value"].iloc[-1]
    total_return = (final_balance - initial_balance) / initial_balance * 100

    # Drawdown
    peak = df["portfolio_value"].cummax()
    drawdown = (df["portfolio_value"] - peak) / peak
    max_drawdown = drawdown.min() * 100

    # Sharpe Ratio (assuming 1m data)
    returns = df["portfolio_value"].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 60)  # Annualized

    print("\n=== Performance Metrics ===")
    print(f"Initial Balance: {initial_balance:,.2f}")
    print(f"Final Balance:   {final_balance:,.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Max Drawdown:    {max_drawdown:.2f}%")
    print(f"Sharpe Ratio:    {sharpe:.2f}")

    # 2. Action Analysis
    print("\n=== Action Analysis ===")
    action_counts = df["action_type"].value_counts()
    print(action_counts)

    # 3. Regime Analysis
    if "regime" in df.columns:
        print("\n=== Regime Analysis ===")
        regime_counts = df["regime"].value_counts()
        print(regime_counts)

        # PnL by Regime
        regime_pnl = df.groupby("regime")["pnl"].sum()
        print("\nPnL by Regime:")
        print(regime_pnl)

    # 4. Plots
    plt.figure(figsize=(15, 10))

    # Portfolio Value
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df["portfolio_value"], label="Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.grid(True)
    plt.legend()

    # Price vs Actions
    plt.subplot(2, 2, 2)
    plt.plot(df.index, df["price"], label="Price", color="gray", alpha=0.5)

    # Buy points
    buys = df[df["action_type"] == "BUY"]
    plt.scatter(
        buys.index, buys["price"], marker="^", color="green", label="Buy", alpha=0.6
    )

    # Sell points
    sells = df[df["action_type"] == "SELL"]
    plt.scatter(
        sells.index, sells["price"], marker="v", color="red", label="Sell", alpha=0.6
    )

    plt.title("Price & Actions")
    plt.grid(True)
    plt.legend()

    # Drawdown
    plt.subplot(2, 2, 3)
    plt.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
    plt.title("Drawdown")
    plt.grid(True)

    # Cumulative PnL
    plt.subplot(2, 2, 4)
    plt.plot(df.index, df["pnl"].cumsum(), label="Cumulative PnL")
    plt.title("Cumulative PnL")
    plt.grid(True)

    plt.tight_layout()
    plot_path = results_dir / "backtest_analysis.png"
    plt.savefig(plot_path)
    print(f"\nAnalysis plot saved to {plot_path}")


if __name__ == "__main__":
    analyze_v451_results()
