#!/usr/bin/env python3
"""
Direct SAC Training for v445.3 Strong Selling Optimized
1万ステップの学習を実行し、統計情報を収集
"""

import json
import os

# Add project root to path
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_environment(config):
    """Create HeavyTradingEnv from config"""
    # Load data
    data_path = config["training"]["data_config"]["data_path"]
    print(f"Loading data from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Data loaded: {len(df)} rows")

    # Create environment config
    from ztb.trading.environment.utils.config import EnvironmentConfig

    env_config_dict = config["training"]["environment"].copy()
    # Ensure continuous actions
    env_config_dict["use_continuous_actions"] = True

    # Extract reward_scaling from reward_settings if nested
    if "reward_settings" in env_config_dict and isinstance(
        env_config_dict["reward_settings"], dict
    ):
        if "reward_scaling" in env_config_dict["reward_settings"]:
            env_config_dict["reward_scaling"] = float(
                env_config_dict["reward_settings"]["reward_scaling"]
            )
        elif "reward_scale" in env_config_dict["reward_settings"]:
            env_config_dict["reward_scaling"] = float(
                env_config_dict["reward_settings"]["reward_scale"]
            )

    # Remove reward_settings to avoid conflicts
    if "reward_settings" in env_config_dict:
        del env_config_dict["reward_settings"]

    # Convert initial_balance to initial_portfolio_value if needed
    if "initial_balance" in env_config_dict:
        env_config_dict["initial_portfolio_value"] = env_config_dict.pop(
            "initial_balance"
        )

    # Remove fields that don't exist in EnvironmentConfig
    fields_to_remove = [
        "feature_engineering",
        "market_regime_detection",
        "risk_management",
        "multi_timeframe_integration",
        "behavior_optimization",
    ]
    for field in fields_to_remove:
        env_config_dict.pop(field, None)

    env_config = EnvironmentConfig(**env_config_dict)

    print(
        f"Environment config created with reward_scaling: {env_config_dict.get('reward_scaling')}"
    )
    print(f"EnvironmentConfig.reward_scaling: {env_config.reward_scaling}")
    print(
        f"Type of EnvironmentConfig.reward_scaling: {type(env_config.reward_scaling)}"
    )
    env = HeavyTradingEnv(df=df, config=env_config, use_continuous_actions=True)
    print(f"Environment action space: {env.action_space}")
    print(f"Environment observation space: {env.observation_space}")
    env = Monitor(env)
    return env


def create_callbacks(config):
    """Create training callbacks"""
    callbacks = []

    # Checkpoint callback
    if (
        "checkpoint" in config["training"]
        and config["training"]["checkpoint"]["enabled"]
    ):
        checkpoint_callback = CheckpointCallback(
            save_freq=config["training"]["checkpoint"]["save_freq"],
            save_path=config["training"]["checkpoint"]["save_path"],
            name_prefix=config["model_name"],
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)

    # Evaluation callback
    if (
        "evaluation" in config["training"]
        and config["training"]["evaluation"]["enabled"]
    ):
        # Create eval environment
        eval_env = create_environment(config)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=config["training"]["checkpoint"]["save_path"],
            log_path="./logs/",
            eval_freq=config["training"]["evaluation"]["eval_freq"],
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

    return callbacks


def main():
    print("🚀 SAC v445.3 Strong Selling Optimized - 10k Training")
    print("=" * 60)

    # Load configuration
    config_path = "config/v445/sac_v445.3_strong_selling_optimized.json"
    config = load_config(config_path)
    print(f"✅ Configuration loaded: {config['model_name']}")

    # Create environment
    print("🏗️ Creating environment...")
    env = create_environment(config)

    # Create PPO model (SAC doesn't support discrete actions)
    print("🤖 Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=config["training"]["logging"]["tensorboard_log"]
        if "logging" in config["training"]
        else None,
    )

    # Create callbacks
    callbacks = create_callbacks(config)

    # Train the model
    print(
        f"🎯 Starting training for {config['training']['total_timesteps']} timesteps..."
    )
    start_time = datetime.now()

    try:
        model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            callback=callbacks,
            log_interval=config["training"]["logging"]["log_interval"]
            if "logging" in config["training"]
            else 100,
        )

        end_time = datetime.now()
        training_duration = end_time - start_time

        print("\n✅ Training completed!")
        print(f"Duration: {training_duration}")
        print("Model saved at checkpoints")

        # Save final model
        final_model_path = f"models/{config['model_name']}_final.zip"
        model.save(final_model_path)
        print(f"Final model saved: {final_model_path}")

        # Collect training statistics
        print("\n📊 Collecting training statistics...")

        # Get environment stats if available
        if hasattr(env, "get_stats"):
            stats = env.get_stats()
            print("Environment Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        # Save training summary
        summary = {
            "model_name": config["model_name"],
            "total_timesteps": config["training"]["total_timesteps"],
            "training_duration": str(training_duration),
            "final_model_path": final_model_path,
            "config_path": config_path,
            "timestamp": datetime.now().isoformat(),
        }

        summary_path = f"results/{config['model_name']}_training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Training summary saved: {summary_path}")

        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
