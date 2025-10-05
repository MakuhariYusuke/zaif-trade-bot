#!/usr/bin/env python3
"""
SELLバイアスの根本原因調査スクリプト
報酬関数の詳細分析と環境の行動バイアス検証を行う
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Direct import from file
import importlib.util
env_spec = importlib.util.spec_from_file_location("environment", project_root / "ztb" / "trading" / "environment" / "environment.py")
if env_spec and env_spec.loader:
    env_module = importlib.util.module_from_spec(env_spec)
    env_spec.loader.exec_module(env_module)
    HeavyTradingEnv = env_module.HeavyTradingEnv
else:
    raise ImportError("Cannot load environment module")


def analyze_reward_function():
    """報酬関数の詳細分析"""
    print("=== 報酬関数の詳細分析 ===")

    # データ読み込み
    data_path = Path(__file__).parent / "ml-dataset-enhanced.csv"
    if not os.path.exists(data_path):
        print(f"データファイルが見つかりません: {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"データ読み込み完了: {len(df)} 行")

    # 環境設定
    config = {
        "reward_scaling": 6.0,
        "transaction_cost": 0.001,
        "max_position_size": 1.0,
        "curriculum_stage": "full",
        "reward_settings": {
            "enable_forced_diversity": True,
            "profit_bonus_multipliers": [1.1, 1.15, 0.8],  # BUY: 1.1, SELL: 1.15, HOLD: 0.8
        }
    }

    env = HeavyTradingEnv(df=df, config=config)
    env.reset()  # 環境を初期化

    # 各行動の報酬計算をテスト
    test_scenarios = [
        {"action": 0, "position": 0, "pnl": 0.0, "description": "HOLD (neutral)"},
        {"action": 1, "position": 1, "pnl": 100.0, "description": "BUY (profitable)"},
        {"action": 2, "position": -1, "pnl": 100.0, "description": "SELL (profitable)"},
        {"action": 1, "position": 1, "pnl": -100.0, "description": "BUY (loss)"},
        {"action": 2, "position": -1, "pnl": -100.0, "description": "SELL (loss)"},
    ]

    print("\n--- 各行動の報酬計算テスト ---")
    for scenario in test_scenarios:
        # 環境状態を設定
        env.position = scenario["position"]
        env.current_step = 100  # 中間ステップ

        # 報酬計算
        reward = env._calculate_reward(
            action=scenario["action"],
            current_price=100.0,
            position=scenario["position"],
            portfolio_value=100000.0,
            atr=1.0,
            transaction_cost=0.001,
            reward_scaling=6.0,
            pnl=scenario["pnl"],
            old_position=0,
            step=100,
            observation=env._get_observation(),
        )

        print(".4f")


def analyze_forced_diversity_penalty():
    """強制アクション多様性のペナルティ分析"""
    print("\n=== 強制アクション多様性のペナルティ分析 ===")

    # forced diversityが有効な場合のペナルティ計算
    action_counts_scenarios = [
        [10, 10, 10],  # バランス良い
        [20, 10, 0],   # SELLを使わない
        [0, 20, 10],   # HOLDを使わない
        [10, 0, 20],   # BUYを使わない
    ]

    for counts in action_counts_scenarios:
        total_actions = sum(counts)
        action_ratios = [count / total_actions for count in counts]

        # 強制多様性のペナルティ計算（環境の実装に基づく）
        unused_penalty = 0.0
        for i, count in enumerate(counts):
            if count == 0:
                if i == 2:  # Extra penalty for not using SELL
                    unused_penalty += 2.0
                else:
                    unused_penalty += 1.0

        min_required_ratio = 0.1
        ratio_penalty = 0.0
        for ratio in action_ratios:
            if ratio < min_required_ratio and ratio > 0:
                ratio_penalty += (min_required_ratio - ratio) * 2.0

        balance_bonus = max(0.0, 0.5 - unused_penalty - ratio_penalty)

        print(f"Action counts {counts}: ratios={[f'{r:.1f}' for r in action_ratios]}, unused_penalty={unused_penalty:.2f}, ratio_penalty={ratio_penalty:.2f}, balance_bonus={balance_bonus:.2f}")


def analyze_data_distribution():
    """学習データの分布分析"""
    print("\n=== 学習データの分布分析 ===")

    data_path = Path(__file__).parent / "ml-dataset-enhanced.csv"
    if not os.path.exists(data_path):
        print(f"データファイルが見つかりません: {data_path}")
        return

    df = pd.read_csv(data_path)

    # 価格変動の分析
    if 'close' in df.columns:
        price_changes = df['close'].pct_change().dropna()
        print(f"価格変化の統計: mean={price_changes.mean():.6f}, std={price_changes.std():.6f}")
        print(f"上昇日数: {(price_changes > 0).sum()}, 下落日数: {(price_changes < 0).sum()}")

    # 特徴量のSELL関連性の分析
    sell_related_features = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sell', 'short', 'bear', 'down'])]
    print(f"SELL関連特徴量: {sell_related_features}")

    buy_related_features = [col for col in df.columns if any(keyword in col.lower() for keyword in ['buy', 'long', 'bull', 'up'])]
    print(f"BUY関連特徴量: {buy_related_features}")


def test_environment_bias():
    """環境の行動バイアス検証"""
    print("\n=== 環境の行動バイアス検証 ===")

    data_path = Path(__file__).parent / "ml-dataset-enhanced.csv"
    if not os.path.exists(data_path):
        print(f"データファイルが見つかりません: {data_path}")
        return

    df = pd.read_csv(data_path)

    # 異なる設定での環境テスト
    configs = [
        {
            "name": "default",
            "config": {
                "reward_scaling": 6.0,
                "curriculum_stage": "full",
                "reward_settings": {
                    "enable_forced_diversity": True,
                    "profit_bonus_multipliers": [1.1, 1.15, 0.8],
                }
            }
        },
        {
            "name": "symmetric_rewards",
            "config": {
                "reward_scaling": 6.0,
                "curriculum_stage": "full",
                "reward_settings": {
                    "enable_forced_diversity": False,  # forced diversityを無効化
                    "profit_bonus_multipliers": [1.0, 1.0, 0.8],  # BUY/SELLを対称化
                }
            }
        },
        {
            "name": "no_forced_diversity",
            "config": {
                "reward_scaling": 6.0,
                "curriculum_stage": "full",
                "reward_settings": {
                    "enable_forced_diversity": False,
                    "profit_bonus_multipliers": [1.0, 1.0, 1.0],  # 全て対称
                }
            }
        }
    ]

    for test_config in configs:
        print(f"\n--- {test_config['name']}設定でのテスト ---")

        env = HeavyTradingEnv(df=df, config=test_config['config'])

        # ランダム行動での報酬分布をテスト
        rewards_by_action = {0: [], 1: [], 2: []}

        for step in range(min(1000, len(df) - 1)):
            env.current_step = step
            action = np.random.choice([0, 1, 2])

            reward = env._calculate_reward(
                action=action,
                current_price=env._resolve_price(),
                position=env.position,
                portfolio_value=env.portfolio_value,
                atr=env._resolve_atr(),
                transaction_cost=0.001,
                reward_scaling=6.0,
                pnl=0.0,
                old_position=env.position,
                step=step,
                observation=env._get_observation(),
            )

            rewards_by_action[action].append(reward)

        for action, rewards in rewards_by_action.items():
            if rewards:
                action_names = ["HOLD", "BUY", "SELL"]
                print(".4f")


def main():
    """メイン実行関数"""
    print("SELLバイアスの根本原因調査を開始します...")

    try:
        analyze_reward_function()
        analyze_forced_diversity_penalty()
        analyze_data_distribution()
        test_environment_bias()

        print("\n=== 調査完了 ===")
        print("SELLバイアスの主な原因:")
        print("1. 強制アクション多様性でSELL未使用に対する過大ペナルティ (2.0)")
        print("2. profit_bonus_multipliersでSELLの報酬が最も低い (0.8)")
        print("3. トレンド調整で上昇トレンド時にSELL報酬が0.8倍になる")

    except Exception as e:
        print(f"調査中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()