import json


# v441学習結果 vs バックテスト結果の比較
with open("reports/training_report_unknown_unknown_20251029_192512.json", "r") as f:
    v441_train = json.load(f)

with open("backtest_results/backtest_results.json", "r") as f:
    backtest = json.load(f)

print("=== v441 学習結果 vs バックテスト結果の比較 ===")
print(f'学習最終報酬: {v441_train["training_stats"]["final_reward"]}')
print(f'バックテスト総報酬: {backtest["total_reward"]:.2f}')
print(f'バックテストポートフォリオリターン: {backtest["portfolio_return_pct"]:.2f}%')
print(f'バックテスト勝率: {backtest["win_rate"]:.1%}')
print()
print("アクション分布比較:")
train_actions = v441_train["training_stats"]["action_distribution"]
print(
    f'学習時 - HOLD: {train_actions["HOLD"]:.1%}, BUY: {train_actions["BUY"]:.1%}, SELL: {train_actions["SELL"]:.1%}'
)
