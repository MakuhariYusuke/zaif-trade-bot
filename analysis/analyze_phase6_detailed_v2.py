import json

import numpy as np
import pandas as pd


def analyze_detailed(file_path):
    print(f"Analyzing: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)

    # Convert to DataFrame
    # Ensure all arrays are the same length by trimming to the minimum length
    min_len = min(
        len(data["timestamps"]),
        len(data["portfolio_history"]),
        len(data["price_history"]),
        len(data["actions"]),
    )

    df = pd.DataFrame(
        {
            "timestamp": data["timestamps"][:min_len],
            "portfolio": data["portfolio_history"][:min_len],
            "price": data["price_history"][:min_len],
            "action": data["actions"][:min_len],
        }
    )

    # Convert timestamp to datetime if possible
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        has_time = True
    except:
        print(
            "Warning: Could not parse timestamps. Time-based analysis will be skipped."
        )
        has_time = False

    # --- 1. Drawdown Analysis ---
    df["peak"] = df["portfolio"].cummax()
    df["drawdown"] = (df["portfolio"] - df["peak"]) / df["peak"]

    max_dd = df["drawdown"].min()
    max_dd_idx = df["drawdown"].idxmin()

    print("\n=== Drawdown Analysis ===")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    if has_time:
        print(f"Max Drawdown Date: {df.iloc[max_dd_idx]['timestamp']}")

    # Find top 5 drawdown periods
    # Simple approach: Find local minima in drawdown

    # --- 2. Trade Analysis ---
    trade_pnls = np.array(data["trade_pnls"])
    n_trades = len(trade_pnls)

    if n_trades > 0:
        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls <= 0]

        win_rate = len(wins) / n_trades * 100
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        profit_factor = (
            abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else float("inf")
        )

        # Consecutive streaks
        streaks = []
        current_streak = 0
        for pnl in trade_pnls:
            if pnl > 0:
                if current_streak < 0:
                    streaks.append(current_streak)
                    current_streak = 1
                else:
                    current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                    current_streak = -1
                else:
                    current_streak -= 1
        streaks.append(current_streak)

        max_win_streak = max([s for s in streaks if s > 0], default=0)
        max_loss_streak = min([s for s in streaks if s < 0], default=0)

        print("\n=== Trade Statistics ===")
        print(f"Total Trades: {n_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Avg Win: {avg_win:.2f}")
        print(f"Avg Loss: {avg_loss:.2f}")
        print(f"Max Consecutive Wins: {max_win_streak}")
        print(f"Max Consecutive Losses: {abs(max_loss_streak)}")

    # --- 3. Volatility Analysis ---
    # Calculate price volatility (rolling std dev of returns)
    df["price_return"] = df["price"].pct_change()
    df["volatility"] = df["price_return"].rolling(window=60).std() * np.sqrt(
        60
    )  # 1-hour volatility approx

    # Calculate portfolio return
    df["portfolio_return"] = df["portfolio"].pct_change()

    # Correlation between portfolio return and volatility
    # Do we lose money when volatility is high?

    # Bin volatility into quartiles
    try:
        df["vol_quartile"] = pd.qcut(
            df["volatility"], 4, labels=["Low", "Med-Low", "Med-High", "High"]
        )

        print("\n=== Performance by Volatility Regime ===")
        vol_perf = (
            df.groupby("vol_quartile")["portfolio_return"].mean() * 100 * 60 * 24
        )  # Approx daily return
        print("Estimated Daily Return by Volatility:")
        print(vol_perf)
    except Exception as e:
        print(f"Could not calculate volatility analysis: {e}")

    # --- 4. Hourly Analysis (if time exists) ---
    if has_time:
        df["hour"] = df["timestamp"].dt.hour
        hourly_perf = (
            df.groupby("hour")["portfolio_return"].mean() * 100 * 60
        )  # Hourly return

        print("\n=== Hourly Performance (Avg Return %) ===")
        # Print top 3 best and worst hours
        print("Best 3 Hours:")
        print(hourly_perf.nlargest(3))
        print("Worst 3 Hours:")
        print(hourly_perf.nsmallest(3))


if __name__ == "__main__":
    import sys

    file_path = "backtest_results/phase6_hft_backtest.json"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    analyze_detailed(file_path)
