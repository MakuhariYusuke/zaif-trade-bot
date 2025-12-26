#!/usr/bin/env python3
"""
SAC v435.6 Training - Ensemble majority voting system
多数決システムによる判断モデル
"""
import json
from pathlib import Path

from ztb.training.unified_trainer.algorithms import create_algorithm_trainer
from utils.config_utils import load_config_from_json, merge_training_configs


def main():
    print("🚀 SAC v435.6 Training - Ensemble majority voting system")
    print("=" * 60)

    # 設定ファイルのパス
    config_dir = Path("backtest_experiments/v435.6")
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
    print(f"  - Ensemble models: {config['environment'].get('ensemble_models', [])}")
    print(
        f"  - Voting mechanism: {config['environment'].get('voting_mechanism', 'N/A')}"
    )

    print("\n🚀 Starting ensemble training...")
    trainer = create_algorithm_trainer("sac", config)
    result = trainer.train()

    print("\n" + "=" * 60)
    if result:
        print("✅ Ensemble training completed successfully!")
        stats = trainer.get_training_stats()
        print(f"Model saved to: {stats.get('model_path', 'N/A')}")
        print(f"Final reward: {stats.get('final_reward', 'N/A')}")
        print(f"Training time: {stats.get('training_time', 'N/A')}")
        print(f"Ensemble consensus rate: {stats.get('ensemble_consensus_rate', 'N/A')}")
    else:
        print("❌ Ensemble training failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
