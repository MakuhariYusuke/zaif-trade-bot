import json
from pathlib import Path

import numpy as np


def analyze_backtest(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    portfolio_history = np.array(data["portfolio_history"])
    price_history = np.array(data["price_history"])
    actions_history = np.array(data["actions"])
    trade_pnls = np.array(data["trade_pnls"])

    initial_value = portfolio_history[0]
    final_value = portfolio_history[-1]
    total_return = (final_value - initial_value) / initial_value * 100

    # Drawdown
    peak = np.maximum.accumulate(portfolio_history)
    drawdown = (portfolio_history - peak) / peak
    max_drawdown = np.min(drawdown) * 100

    # Trade Analysis
    n_trades = len(trade_pnls)
    if n_trades > 0:
        winning_trades = trade_pnls[trade_pnls > 0]
        losing_trades = trade_pnls[trade_pnls <= 0]
        win_rate = len(winning_trades) / n_trades * 100
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        profit_factor = (
            abs(np.sum(winning_trades) / np.sum(losing_trades))
            if np.sum(losing_trades) != 0
            else float("inf")
        )
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    # Action Analysis
    # Assuming actions are continuous [-1, 1] or discrete mapped
    # Let's just count non-zero actions if possible, or just distribution
    # actions_history might be floats

    print("=" * 50)
    print(f"Backtest Analysis: {file_path.name}")
    print("=" * 50)
    print(f"Initial Portfolio: {initial_value:,.2f}")
    print(f"Final Portfolio:   {final_value:,.2f}")
    print(f"Total Return:      {total_return:.2f}%")
    print(f"Max Drawdown:      {max_drawdown:.2f}%")
    print("-" * 30)
    print(f"Total Trades:      {n_trades}")
    print(f"Win Rate:          {win_rate:.2f}%")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"Avg Win:           {avg_win:,.2f}")
    print(f"Avg Loss:          {avg_loss:,.2f}")
    print("=" * 50)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    results_path = project_root / "backtest_results" / "phase6_hft_backtest.json"
    analyze_backtest(results_path)
