#!/usr/bin/env python3
"""Check backtest data time period and calculations"""

from datetime import datetime

import pandas as pd

# Load data
df = pd.read_csv("data/btc_jpy_real_dataset.csv")
print(f"データ行数: {len(df)}")
print(f'開始日時: {df.iloc[0]["timestamp"]}')
print(f'終了日時: {df.iloc[len(df)-1]["timestamp"]}')
print("タイムフレーム: 1時間足（1時間間隔）")

# 期間計算
start = datetime.strptime(df.iloc[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
end = datetime.strptime(df.iloc[len(df) - 1]["timestamp"], "%Y-%m-%d %H:%M:%S")
duration = end - start
print(f"データ期間: {duration.days}日 {duration.seconds//3600}時間")

# バックテストで使用したステップ数
backtest_steps = 4139
print(f"バックテストステップ数: {backtest_steps}")
print(
    f"相当時間: {backtest_steps}時間 = {backtest_steps//24}日 {backtest_steps%24}時間"
)

# 計算確認
initial_portfolio = 10000.0
final_portfolio = 1247365.9552089658
total_profit = final_portfolio - initial_portfolio
total_return_pct = (total_profit / initial_portfolio) * 100

print("\n計算確認:")
print(f"初期ポートフォリオ: ¥{initial_portfolio:,.2f}")
print(f"最終ポートフォリオ: ¥{final_portfolio:,.2f}")
print(f"総利益: ¥{total_profit:,.2f}")
print(f"総リターン: {total_return_pct:.2f}%")

# 手動計算確認
manual_profit = 1247365.9552089658 - 10000
manual_return = (manual_profit / 10000) * 100
print("\n手動計算確認:")
print(f"総利益: ¥{manual_profit:,.2f}")
print(f"総リターン: {manual_return:.2f}%")
