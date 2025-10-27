#!/usr/bin/env python3
"""
SAC v435.5 Training - Micro frequency penalty scalping
微少な高頻度ペナルティを課すスケルピングモデル
"""
import json
from pathlib import Path

from ztb.training.unified_trainer.algorithms import create_algorithm_trainer


def main():
    print("🚀 SAC v435.5 Training - Micro frequency penalty scalping")
    print("=" * 60)

    # 設定ファイルのパス
    config_dir = Path("backtest_experiments/v435.5")
    config_path = config_dir / "sac_v435_config.json"
    env_config_path = config_dir / "sac_v435_environment_config.json"
    reward_config_path = config_dir / "sac_v435_reward_config.json"

    # メイン設定ファイル読み込み
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 環境設定を統合
    if env_config_path.exists():
        with open(env_config_path, "r", encoding="utf-8") as f:
            env_config = json.load(f)
        if "environment" not in config:
            config["environment"] = {}
        config["environment"].update(env_config)

    # 報酬設定を統合
    if reward_config_path.exists():
        with open(reward_config_path, "r", encoding="utf-8") as f:
            reward_config = json.load(f)
        if "reward_function" not in config:
            config["reward_function"] = {}
        config["reward_function"].update(reward_config)

    print("📋 Configuration loaded:")
    print(f"  - Model: {config['model_name']}")
    print(f"  - Timesteps: {config['training']['total_timesteps']:,}")
    print(f"  - Transaction cost: {config['environment']['transaction_cost']}")
    print(f"  - Max position size: {config['environment']['max_position_size']}")
    print(
        f"  - Frequency penalty: {config['reward_function']['action_frequency_penalty']}"
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
