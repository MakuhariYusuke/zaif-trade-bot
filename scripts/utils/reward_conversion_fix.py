"""
報酬変換ロジックの修正
現在の問題: portfolio_change = (reward[0] + 10) * 10
- 相関係数が低い (0.16)
- 勝率計算の不正確さ

新しいロジック:
1. 報酬を直接ポートフォリオ変化として使用（スケーリング最小化）
2. または、報酬の範囲に適した小さなスケーリング
"""


def convert_reward_to_portfolio_change(reward, method="direct"):
    """
    報酬をポートフォリオ変化に変換

    Args:
        reward: エピソード報酬（スカラーまたは配列）
        method: 変換方法
            - 'direct': 報酬を直接使用
            - 'scaled': 小さなスケーリング (reward * 0.1)
            - 'normalized': 平均0、標準偏差1に正規化後スケーリング

    Returns:
        portfolio_change: ポートフォリオ変化
    """
    if isinstance(reward, (list, tuple)):
        reward_val = reward[0] if reward else 0
    else:
        reward_val = reward

    if method == "direct":
        # 報酬を直接ポートフォリオ変化として使用
        return reward_val

    elif method == "scaled":
        # 小さなスケーリング（現在の1/10）
        return reward_val * 0.1

    elif method == "normalized":
        # 経験的な平均と標準偏差に基づく正規化
        # Pendulum環境の典型的な報酬範囲を考慮
        reward_mean = 0.0  # 制御コストの期待値
        reward_std = 8.0  # 経験的な標準偏差
        normalized = (reward_val - reward_mean) / reward_std
        return normalized * 10  # 適度なスケーリング

    else:
        raise ValueError(f"Unknown conversion method: {method}")


# テスト
if __name__ == "__main__":
    # テスト報酬範囲
    test_rewards = [-16, -8, 0, 8, 16]

    print("Reward -> Portfolio Change Conversion Test")
    print("=" * 50)

    for method in ["direct", "scaled", "normalized"]:
        print(f"\nMethod: {method}")
        for r in test_rewards:
            pc = convert_reward_to_portfolio_change(r, method)
            print(f"  {r:4.0f} -> {pc:6.2f}")
