"""
報酬関数の詳細分析スクリプト
現在の報酬構造を分析し、HOLD率増加の原因を特定
"""


def analyze_reward_structure() -> None:
    """現在の報酬設定を分析"""

    print("=" * 100)
    print("報酬関数分析レポート")
    print("=" * 100)

    # 現在の報酬設定例 (v371/v376/v377 共通)
    reward_settings = {
        "hold_penalty_weight": 0.005,
        "consecutive_hold_penalty": 0.002,
        "trading_frequency_bonus": 0.05,
        "profit_reward_multiplier": 1.5,
        "action_diversity_bonus": 0.01,
    }

    print("\n【現在の報酬設定】")
    print("-" * 100)
    for key, value in reward_settings.items():
        print(f"  {key:30s}: {value}")

    print("\n【問題分析】")
    print("-" * 100)

    # 1. スケール問題
    print("\n■ 1. 報酬スケールの不均衡")
    print("問題:")
    print("  - HOLDペナルティ: -0.005 (固定値)")
    print("  - PnL報酬: ATR正規化された小さな値 (通常 -0.1 ~ +0.1)")
    print("  - ペナルティとPnLが同じスケール → HOLDを避けるインセンティブが弱い")
    print("\n解決策:")
    print("  ✅ HOLDペナルティを動的にスケーリング (ATRやボラティリティに応じて)")
    print("  ✅ PnL報酬を増幅 (現在1.5倍だが、さらに強化)")

    # 2. 連続HOLD問題
    print("\n■ 2. 連続HOLDペナルティの効果不足")
    print("問題:")
    print("  - consecutive_hold_penalty: 0.002")
    print("  - 10回連続HOLDでも -0.02 (小さすぎる)")
    print("\n解決策:")
    print("  ✅ 指数的ペナルティ: penalty × (1.1 ^ consecutive_holds)")
    print("  ✅ 閾値設定: 5回以上連続でペナルティ急増")

    # 3. ボーナスの効果不足
    print("\n■ 3. トレーディングボーナスの効果不足")
    print("問題:")
    print("  - trading_frequency_bonus: 0.05 (取引時のみ)")
    print("  - action_diversity_bonus: 0.01 (多様性時のみ)")
    print("  - 合計でも +0.06程度 → 取引コストや小さなPnL損失を上回らない")
    print("\n解決策:")
    print("  ✅ ボーナスを大幅強化 (0.05 → 0.2+)")
    print("  ✅ 成功取引にはさらなるボーナス")

    # 4. リスク・リターン比の問題
    print("\n■ 4. リスク・リターン比の不均衡")
    print("問題:")
    print("  - HOLD: 安定的に -0.005 のペナルティ")
    print("  - BUY/SELL: 不確実なPnL + 取引コスト")
    print("  - 期待値が HOLD有利 になっている可能性")
    print("\n解決策:")
    print("  ✅ PnLの期待値を上げる (成功取引の報酬増幅)")
    print("  ✅ 取引コストを下げる (現在0.0001, さらに削減?)")
    print("  ✅ HOLDのペナルティを上げる")

    # 5. 時間依存性の欠如
    print("\n■ 5. 市場状況に応じた報酬調整の欠如")
    print("問題:")
    print("  - ボラティリティが高い時もHOLDペナルティ一定")
    print("  - トレンドが明確な時もボーナス一定")
    print("  - 市場状況を無視した画一的な報酬")
    print("\n解決策:")
    print("  ✅ ボラティリティ連動ペナルティ (高ボラ時はHOLDペナルティ増)")
    print("  ✅ トレンド連動ボーナス (明確なトレンド時は取引ボーナス増)")
    print("  ✅ レンジ相場ではHOLD許容")

    print("\n" + "=" * 100)
    print("\n【推奨される報酬関数の改善案】")
    print("-" * 100)

    print("\n■ 改善案1: スケール調整版")
    improved_v1 = {
        "hold_penalty_weight": 0.02,  # 4倍に強化
        "consecutive_hold_penalty": 0.01,  # 5倍に強化
        "consecutive_hold_exponential": True,  # NEW: 指数的増加
        "consecutive_hold_threshold": 5,  # NEW: 閾値
        "trading_frequency_bonus": 0.15,  # 3倍に強化
        "profit_reward_multiplier": 3.0,  # 2倍に強化 (1.5→3.0)
        "action_diversity_bonus": 0.05,  # 5倍に強化
        "successful_trade_bonus": 0.5,  # NEW: 利益取引にボーナス
    }

    for key, value in improved_v1.items():
        new_marker = " # NEW" if key not in reward_settings else ""
        print(f"  {key:35s}: {value}{new_marker}")

    print("\n■ 改善案2: 動的スケール版")
    improved_v2 = {
        **improved_v1,
        "volatility_adjusted_penalty": True,  # NEW: ボラティリティ連動
        "trend_adjusted_bonus": True,  # NEW: トレンド連動
        "range_market_hold_tolerance": 0.5,  # NEW: レンジ相場でHOLD許容
    }

    print("  (改善案1に加えて)")
    for key in [
        "volatility_adjusted_penalty",
        "trend_adjusted_bonus",
        "range_market_hold_tolerance",
    ]:
        print(f"  {key:35s}: {improved_v2[key]} # NEW")

    print("\n■ 改善案3: アグレッシブ版")
    improved_v3 = {
        "hold_penalty_weight": 0.05,  # 10倍に強化
        "consecutive_hold_penalty": 0.03,  # 15倍に強化
        "consecutive_hold_exponential": True,
        "consecutive_hold_threshold": 3,  # より厳しく
        "trading_frequency_bonus": 0.3,  # 6倍に強化
        "profit_reward_multiplier": 5.0,  # 超強化
        "action_diversity_bonus": 0.1,  # 10倍に強化
        "successful_trade_bonus": 1.0,  # 大幅ボーナス
        "hold_opportunity_cost": 0.02,  # NEW: 機会損失ペナルティ
    }

    for key, value in improved_v3.items():
        print(f"  {key:35s}: {value}")

    print("\n" + "=" * 100)
    print("\n【期待される効果】")
    print("-" * 100)

    comparison = [
        ["指標", "現在", "改善案1", "改善案2", "改善案3"],
        ["HOLDペナルティ", "-0.005", "-0.02 (4x)", "-0.02 (動的)", "-0.05 (10x)"],
        [
            "連続HOLDペナルティ",
            "-0.002/回",
            "-0.01/回 (指数)",
            "同左",
            "-0.03/回 (指数)",
        ],
        ["取引ボーナス", "+0.05", "+0.15 (3x)", "+0.15 (動的)", "+0.3 (6x)"],
        ["利益増幅", "1.5x", "3.0x", "3.0x", "5.0x"],
        ["期待HOLD率", "60-65%", "45-55%", "40-50%", "30-40%"],
        ["リスク", "低", "中", "中高", "高"],
    ]

    col_widths = [25, 15, 15, 15, 15]
    for i, row in enumerate(comparison):
        if i == 0:
            print(
                "  "
                + " | ".join(f"{cell:^{col_widths[j]}}" for j, cell in enumerate(row))
            )
            print("  " + "-" * (sum(col_widths) + 3 * len(col_widths)))
        else:
            print(
                "  "
                + " | ".join(f"{cell:^{col_widths[j]}}" for j, cell in enumerate(row))
            )

    print("\n" + "=" * 100)
    print("\n【実装推奨順序】")
    print("-" * 100)
    print("  1. ✅ 改善案1 (スケール調整) - リスク低、効果中")
    print("     → まず基本的なスケール調整で効果確認")
    print("  2. ⏳ 改善案2 (動的スケール) - リスク中、効果高")
    print("     → 市場状況に応じた適応的報酬")
    print("  3. ⏳ 改善案3 (アグレッシブ) - リスク高、効果超高")
    print("     → HOLD率を徹底的に下げたい場合")

    print("\n" + "=" * 100)
    print("分析完了")
    print("=" * 100)


if __name__ == "__main__":
    analyze_reward_structure()
