import json
import sys

import numpy as np

file_path = "backtest_results/phase6_hft_backtest.json"

try:
    with open(file_path, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    sys.exit(1)

trade_pnls = data.get("trade_pnls", [])
if not trade_pnls:
    print("No trades found.")
    sys.exit(0)

pnls = np.array(trade_pnls)
total_trades = len(pnls)
win_trades = pnls[pnls > 0]
loss_trades = pnls[pnls <= 0]

win_rate = len(win_trades) / total_trades * 100
total_pnl = np.sum(pnls)
gross_profit = np.sum(win_trades)
gross_loss = abs(np.sum(loss_trades))
profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Total PnL: {total_pnl:.2f}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Avg Trade PnL: {np.mean(pnls):.2f}")
