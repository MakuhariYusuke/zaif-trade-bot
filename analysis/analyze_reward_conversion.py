import json

import numpy as np

# バックテスト結果から報酬統計を確認
with open(
    "backtest_results/backtest_results_sac_v427_hybrid_20251026_063723.json", "r"
) as f:
    data = json.load(f)

print("Reward stats from backtest:")
print(f'Mean: {data["reward_stats"]["mean"]:.2f}')
print(f'Std: {data["reward_stats"]["std"]:.2f}')
print(f'Min: {data["reward_stats"]["min"]:.2f}')
print(f'Max: {data["reward_stats"]["max"]:.2f}')

# 現在の変換ロジックの効果を確認
reward_range = np.linspace(data["reward_stats"]["min"], data["reward_stats"]["max"], 10)
portfolio_changes = [(r + 10) * 10 for r in reward_range]

print("\nCurrent conversion (reward + 10) * 10:")
for r, pc in zip(reward_range, portfolio_changes):
    print(f"Reward {r:.2f} -> Portfolio change {pc:.2f}")

# より適切な変換ロジックの提案
print("\nProposed conversion (reward + 16) * 5:")  # より小さなスケーリング
portfolio_changes_new = [(r + 16) * 5 for r in reward_range]
for r, pc in zip(reward_range, portfolio_changes_new):
    print(f"Reward {r:.2f} -> Portfolio change {pc:.2f}")

# 正規化ベースの変換
reward_mean = data["reward_stats"]["mean"]
reward_std = data["reward_stats"]["std"]
print("\nNormalized conversion (reward - mean) / std * 10:")
portfolio_changes_norm = [(r - reward_mean) / reward_std * 10 for r in reward_range]
for r, pc in zip(reward_range, portfolio_changes_norm):
    print(f"Reward {r:.2f} -> Portfolio change {pc:.2f}")
