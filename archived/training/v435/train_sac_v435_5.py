#!/usr/bin/env python3
"""
SAC v435.5 Training - Micro frequency penalty scalping
微少な高頻度ペナルティを課すスケルピングモデル
"""
import json
import time
from pathlib import Path

from utils.config_utils import load_config_from_json, merge_training_configs
from ztb.training.unified_trainer.algorithms import create_algorithm_trainer
from ztb.utils.training_utils import display_training_complete


def main():
    print("🚀 SAC v435.5 Training - Micro frequency penalty scalping")
    print("=" * 60)

    start_time = time.time()

    # 設定ファイルのパス
    config_dir = Path("backtest_experiments/v435.5")
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

    print("\n🚀 Starting training...")
    trainer = create_algorithm_trainer("sac", config)
    result = trainer.train()

    training_time = time.time() - start_time
    if result:
        stats = trainer.get_training_stats()
        final_metrics = {
            "model_path": stats.get('model_path', 'N/A'),
            "final_reward": stats.get('final_reward', 'N/A'),
            "training_success": True,
        }
    else:
        final_metrics = {
            "training_success": False,
        }
    display_training_complete(final_metrics, training_time)
    else:
        print("❌ Training failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
