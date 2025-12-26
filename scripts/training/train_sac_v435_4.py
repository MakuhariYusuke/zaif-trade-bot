#!/usr/bin/env python3
"""
SAC v435.4 Training - Advanced scalping with enhanced profit bonuses
1万ステップ学習
"""
import json
from pathlib import Path

from ztb.training.unified_trainer.algorithms import create_algorithm_trainer
from utils.config_utils import load_config_from_json, merge_training_configs


def main():
    print("🚀 SAC v435.4 Training - Advanced Scalping")
    print("=" * 60)

    # 設定ファイルのパス
    config_dir = Path("backtest_experiments/v435.4")
    config_path = config_dir / "sac_v435_config.json"
    env_config_path = config_dir / "sac_v435_environment_config.json"
    reward_config_path = config_dir / "sac_v435_reward_config.json"

    # メイン設定ファイル読み込み
    config = load_config_from_json(config_path)

    # 環境設定と報酬設定を統合
    config = merge_training_configs(
        config,
        env_config_path=env_config_path,
        reward_config_path=reward_config_path
    )

    print("📋 Configuration loaded:")
    print(f"  - Model: {config['model_name']}")
    print(f"  - Timesteps: {config['training']['total_timesteps']:,}")
    print(f"  - Transaction cost: {config['environment']['transaction_cost']}")
    print(f"  - Max position size: {config['environment']['max_position_size']}")
    print(
        f"  - Frequency penalty: {config['reward_function']['action_frequency_penalty']}"
    )
    if "profit_bonus_multiplier" in config["reward_function"]:
        print(
            f"  - Profit bonus multiplier: {config['reward_function']['profit_bonus_multiplier']}"
        )

    print("\n🚀 Starting training...")
    trainer = create_algorithm_trainer("sac", config)
    result = trainer.train()

    print("\n" + "=" * 60)
    if result:
        print("✅ Training completed successfully!")
        stats = trainer.get_training_stats()
        print(f"Model saved to: {stats.get('model_path', 'N/A')}")
        print(f"Final reward: {stats.get('final_reward', 'N/A')}")
        print(f"Training time: {stats.get('training_time', 'N/A')}")
    else:
        print("❌ Training failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
