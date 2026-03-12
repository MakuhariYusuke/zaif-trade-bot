#!/usr/bin/env python3
"""
SAC v427/v429 Quick Experiment Script

SELLバイアスを減らすための短い学習実験
v429: 対称アクション変換と報酬最適化実験対応
"""

import json
import os
import subprocess
import time
from pathlib import Path

def main():
    """メイン実験実行"""
    print("SAC v427/v429 SELLバイアス削減実験")
    print("v427: 従来の報酬調整, v429: 対称アクション変換 + 報酬最適化")
    print("短いステップ数でパラメータを調整しながら学習")

    # 実験設定
    experiments = [
        # ベースライン
        {"experiment_name": "baseline_10k", "timesteps": 10000},
        # 報酬スケール調整
        {
            "experiment_name": "reward_scale_50",
            "timesteps": 10000,
            "reward_scale": 50.0,
        },
        {
            "experiment_name": "reward_scale_200",
            "timesteps": 10000,
            "reward_scale": 200.0,
        },
        {
            "experiment_name": "reward_scale_500",
            "timesteps": 10000,
            "reward_scale": 500.0,
        },
        # 取引ボーナス調整
        {
            "experiment_name": "trading_bonus_002",
            "timesteps": 10000,
            "trading_bonus": 0.02,
        },
        {
            "experiment_name": "trading_bonus_005",
            "timesteps": 10000,
            "trading_bonus": 0.05,
        },
        # SELLペナルティ追加
        {
            "experiment_name": "sell_penalty_001",
            "timesteps": 10000,
            "sell_action_penalty": -0.01,
        },
        {
            "experiment_name": "sell_penalty_005",
            "timesteps": 10000,
            "sell_action_penalty": -0.05,
        },
        {
            "experiment_name": "sell_penalty_010",
            "timesteps": 10000,
            "sell_action_penalty": -0.10,
        },
        # BUYボーナス追加
        {
            "experiment_name": "buy_bonus_001",
            "timesteps": 10000,
            "buy_action_penalty": 0.01,
        },
        {
            "experiment_name": "buy_bonus_005",
            "timesteps": 10000,
            "buy_action_penalty": 0.05,
        },
        # 組み合わせ実験
        {
            "experiment_name": "balanced_penalty",
            "timesteps": 10000,
            "sell_action_penalty": -0.05,
            "buy_action_penalty": 0.05,
        },
        {
            "experiment_name": "scale200_penalty005",
            "timesteps": 10000,
            "reward_scale": 200.0,
            "sell_action_penalty": -0.05,
        },
        # より長い学習
        {"experiment_name": "longer_25k", "timesteps": 25000},
        {"experiment_name": "longer_50k", "timesteps": 50000},
        # === SAC v429 実験 (対称アクション変換) ===
        # v429ベースライン
        {"experiment_name": "v429_baseline", "version": "v429", "timesteps": 10000},
        # アクション平衡ウェイト調整
        {
            "experiment_name": "v429_balance_01",
            "version": "v429",
            "timesteps": 10000,
            "action_balance_weight": 0.1,
        },
        {
            "experiment_name": "v429_balance_02",
            "version": "v429",
            "timesteps": 10000,
            "action_balance_weight": 0.2,
        },
        {
            "experiment_name": "v429_balance_03",
            "version": "v429",
            "timesteps": 10000,
            "action_balance_weight": 0.3,
        },
        # 報酬スケール + アクション平衡
        {
            "experiment_name": "v429_scale200_balance02",
            "version": "v429",
            "timesteps": 10000,
            "reward_scale": 200.0,
            "action_balance_weight": 0.2,
        },
        # SELLペナルティ + アクション平衡
        {
            "experiment_name": "v429_penalty005_balance01",
            "version": "v429",
            "timesteps": 10000,
            "sell_action_penalty": -0.05,
            "action_balance_weight": 0.1,
        },
        # 報酬最適化実験
        {
            "experiment_name": "v429_optimize_reward",
            "version": "v429",
            "timesteps": 10000,
            "optimize_reward": True,
        },
    ]

    # 結果ディレクトリ作成
    os.makedirs("results/experiments", exist_ok=True)

    # 実験実行
    results = []
    for exp in experiments:
        success = run_experiment(**exp)
        results.append(
            {"experiment": exp["experiment_name"], "success": success, "config": exp}
        )

    # 結果サマリー
    print(f"\n{'='*60}")
    print("実験結果サマリー")
    print("=" * 60)

    successful = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"総実験数: {total}")
    print(f"成功数: {successful}")
    print(f"成功率: {successful/total*100:.1f}%")

    print("\n各実験結果:")
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {result['experiment']}")

    # 推奨設定の提案
    print("\n次の推奨アクション:")
    print("1. 成功した実験の設定を分析")
    print("2. SELL比率が最も低かった設定を特定")
    print("3. その設定でより長い学習を実行")

if __name__ == "__main__":
    main()
