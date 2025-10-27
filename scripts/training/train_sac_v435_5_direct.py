#!/usr/bin/env python3
"""
SAC v435.5 Training Script - Micro frequency penalty scalping
僅かに高頻度ペナルティを課すスケルピングモデル
"""

import json
import logging
from pathlib import Path

import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment.schema_env_factory import create_env_from_schema

logger = logging.getLogger(__name__)


def main():
    print("🚀 SAC v435.5 Training - Micro frequency penalty scalping")
    print("=" * 60)

    # Load configuration
    config_dir = Path("backtest_experiments/v435.5")
    config_path = config_dir / "sac_v435_config.json"
    env_config_path = config_dir / "sac_v435_environment_config.json"
    reward_config_path = config_dir / "sac_v435_reward_config.json"

    # Load main config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load environment config
    if env_config_path.exists():
        with open(env_config_path, "r", encoding="utf-8") as f:
            env_config = json.load(f)
        if "environment" not in config:
            config["environment"] = {}
        config["environment"].update(env_config)

    # Load reward config
    if reward_config_path.exists():
        with open(reward_config_path, "r", encoding="utf-8") as f:
            reward_config = json.load(f)
        if "reward_function" not in config:
            config["reward_function"] = {}
        config["reward_function"].update(reward_config)

    print("📋 Configuration loaded:")
    print(f"  - Model: {config['model_name']}")
    print(
        f"  - Frequency penalty: {config['reward_function']['action_frequency_penalty']}"
    )
    print(f"  - Max position size: {config['environment']['max_position_size']}")

    # Load data
    print("📊 Loading data...")
    data_path = config["training"]["data_config"]["data_path"]
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} rows of data from {data_path}")

    # Create environment
    print("\n🏗️  Creating environment...")
    # Extract model name as string
    model_name = config.get("model_name", "sac_v435.5")
    env = create_env_from_schema(model_name, df)
    env = DummyVecEnv([lambda: env])

    # Create model
    print("🤖 Creating SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=config["training"]["sac_hyperparameters"]["learning_rate"],
        batch_size=config["training"]["sac_hyperparameters"]["batch_size"],
        buffer_size=config["training"]["sac_hyperparameters"]["buffer_size"],
        learning_starts=config["training"]["sac_hyperparameters"]["learning_starts"],
        tau=config["training"]["sac_hyperparameters"]["tau"],
        gamma=config["training"]["sac_hyperparameters"]["gamma"],
        ent_coef=config["training"]["sac_hyperparameters"]["ent_coef"],
        target_entropy=config["training"]["sac_hyperparameters"]["target_entropy"],
        verbose=1,
        tensorboard_log=f"./tensorboard/{config['model_name']}",
    )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path="./checkpoints", name_prefix=config["model_name"]
    )

    # Train model
    print("🚀 Starting training...")
    total_timesteps = config["training"]["total_timesteps"]
    model.learn(
        total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=True
    )

    # Save final model
    model_path = f"models/{config['model_name']}.zip"
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")

    print("\n" + "=" * 60)
    print("✅ SAC v435.5 training completed successfully!")
    print(f"Model: {config['model_name']}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Frequency penalty: {config['reward_function']['action_frequency_penalty']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
