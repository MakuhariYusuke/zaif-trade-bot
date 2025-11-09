#!/usr/bin/env python3
"""
SAC v445.4 Ultra Aggressive Selling - Training Script
Enhanced SELL action incentives with maximum profit-taking optimization
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_environment(config: dict) -> HeavyTradingEnv:
    """Create and configure the trading environment."""
    print("🏗️ Creating environment...")

    # Load data
    data_path = config["training"]["data_config"]["data_path"]
    print(f"Loading data from: {data_path}")
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
    env = HeavyTradingEnv(df=df, config=env_config, use_continuous_actions=True)
    print(f"Environment action space: {env.action_space}")
    print(f"Environment observation space: {env.observation_space}")
    env = Monitor(env)
    return env


def create_callbacks(config: dict):
    """Create training callbacks."""
    callbacks = []

    # Checkpoint callback
    checkpoint_config = config["training"]["checkpoint"]
    if checkpoint_config["enabled"]:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_config["save_freq"],
            save_path=checkpoint_config["save_path"],
            name_prefix=config["model_name"],
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_config = config["training"]["evaluation"]
    if eval_config["enabled"]:
        # Create eval environment
        eval_env = create_environment(config)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=checkpoint_config["save_path"],
            log_path="./logs/",
            eval_freq=eval_config["eval_freq"],
            n_eval_episodes=eval_config["n_eval_episodes"],
            deterministic=eval_config["deterministic"],
            render=eval_config["render"],
            verbose=eval_config["verbose"] if "verbose" in eval_config else 0,
        )
        callbacks.append(eval_callback)

    return callbacks


def main():
    """Main training function."""
    print("🚀 SAC v445.4 Ultra Aggressive Selling - 10k Training")
    print("=" * 60)

    # Load configuration
    config_path = "config/v445/sac_v445.4_ultra_aggressive_selling.json"
    print(f"✅ Configuration loaded: {Path(config_path).stem}")
    config = load_config(config_path)

    # Create environment
    env = create_environment(config)

    # Create PPO model (SAC -> PPO for discrete actions)
    print("🤖 Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
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
        tensorboard_log=config["training"]["logging"]["tensorboard_log"],
    )

    # Create callbacks
    callbacks = create_callbacks(config)

    # Training
    total_timesteps = config["training"]["total_timesteps"]
    print(f"🎯 Starting training for {total_timesteps} timesteps...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        print("✅ Training completed!")

        # Save final model
        model_path = f"models/{config['model_name']}_final.zip"
        model.save(model_path)
        print(f"💾 Final model saved: {model_path}")

        # Save training summary
        summary = {
            "model_name": config["model_name"],
            "total_timesteps": total_timesteps,
            "training_duration": "completed",
            "final_model_path": model_path,
            "config_path": config_path,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        summary_path = f"results/{config['model_name']}_training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📊 Training summary saved: {summary_path}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
