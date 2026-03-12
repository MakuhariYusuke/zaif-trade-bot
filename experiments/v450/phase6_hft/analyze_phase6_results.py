import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_results():
    file_path = Path("backtest_results/phase6_hft_backtest.json")
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    print("=== Phase 6 HFT Backtest Analysis ===")
    total_steps = data.get("total_steps", len(data.get("portfolio_history", [])))
    print(f"Total Steps: {total_steps}")

    initial_balance = data.get("initial_balance", 0)
    final_balance = data.get("final_balance", 0)
    print(f"Initial Balance: {initial_balance:,.2f}")
    print(f"Final Balance: {final_balance:,.2f}")

    portfolio_history = data.get("portfolio_history", [])
    if portfolio_history:
        initial_pv = portfolio_history[0]
        final_pv = portfolio_history[-1]
        total_return = (final_pv - initial_pv) / initial_pv * 100
        print(f"Total Return: {total_return:.2f}%")

        # Drawdown
        ph = np.array(portfolio_history)
        peak = np.maximum.accumulate(ph)
        drawdown = (ph - peak) / peak
        max_drawdown = drawdown.min() * 100
        print(f"Max Drawdown: {max_drawdown:.2f}%")

    print("\n--- Action Distribution ---")
    actions = data.get("actions", [])
    if actions:
        total_actions = len(actions)
        counts = Counter(actions)
        for action, count in counts.items():
            print(f"Action {action}: {count} ({count/total_actions*100:.1f}%)")

    # Trade Analysis
    trade_pnls = data.get("trade_pnls", [])
    if trade_pnls:
        df_trades = pd.DataFrame({"pnl": trade_pnls})
        print(f"\n--- Trade Analysis (Total {len(trade_pnls)}) ---")

        profitable_trades = df_trades[df_trades["pnl"] > 0]
        losing_trades = df_trades[df_trades["pnl"] <= 0]

        print(
            f"Profitable Trades: {len(profitable_trades)} ({len(profitable_trades)/len(trade_pnls)*100:.1f}%)"
        )
        print(
            f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trade_pnls)*100:.1f}%)"
        )

        if not profitable_trades.empty:
            print(f"Avg Profit: {profitable_trades['pnl'].mean():.2f}")
        if not losing_trades.empty:
            print(f"Avg Loss: {losing_trades['pnl'].mean():.2f}")

        total_pnl = sum(trade_pnls)
        print(f"Total Trade PnL: {total_pnl:,.2f}")

        if len(losing_trades) > 0:
            win_rate = len(profitable_trades) / len(trade_pnls)
            avg_win = (
                profitable_trades["pnl"].mean() if not profitable_trades.empty else 0
            )
            avg_loss = (
                abs(losing_trades["pnl"].mean()) if not losing_trades.empty else 0
            )
            profit_factor = (
                (avg_win * len(profitable_trades)) / (avg_loss * len(losing_trades))
                if avg_loss > 0
                else float("inf")
            )
            print(f"Win Rate: {win_rate*100:.2f}%")
            print(f"Profit Factor: {profit_factor:.2f}")


if __name__ == "__main__":
    analyze_results()
