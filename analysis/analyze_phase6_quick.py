import json
import sys

import pandas as pd

file_path = "backtest_results/phase6_hft_backtest.json"

try:
    with open(file_path, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    sys.exit(1)

trades = data.get("trades", [])
if not trades:
    print("No trades found.")
    sys.exit(0)

df = pd.DataFrame(trades)
df["pnl"] = df["pnl"].astype(float)
df["pnl_pct"] = df["pnl_pct"].astype(float)

total_trades = len(df)
win_trades = df[df["pnl"] > 0]
loss_trades = df[df["pnl"] <= 0]

win_rate = len(win_trades) / total_trades * 100
total_pnl = df["pnl"].sum()
gross_profit = win_trades["pnl"].sum()
gross_loss = abs(loss_trades["pnl"].sum())
profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Total PnL: {total_pnl:.2f}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Avg Trade PnL: {df['pnl'].mean():.2f}")
