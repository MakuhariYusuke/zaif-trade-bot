#!/usr/bin/env python3
"""
Validation training with optimized parameters.

Combines the optimized PPO hyperparameters with Lagrange constraints
and runs a comprehensive 100k step training session with detailed logging.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.custom_ppo import CustomPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


class TrainingStatsCallback(BaseCallback):
    """Callback for tracking training statistics."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get("dones", [False])[0]:
            if "infos" in self.locals and len(self.locals["infos"]) > 0:
                info = self.locals["infos"][0]
                if "episode" in info:
                    self.episode_rewards.append(float(info["episode"]["r"]))
                    self.episode_lengths.append(int(info["episode"]["l"]))
                    self.episode_count += 1
        return True


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def create_environment(config: Dict[str, Any]) -> HeavyTradingEnv:
    """Create trading environment from config."""
    # Load dataset
    df = pd.read_csv(config["data"]["dataset_path"])
    
    # Split into train/test
    split_idx = int(len(df) * config["data"]["train_test_split"])
    train_df = df[:split_idx].copy()
    
    env_config = {
        "initial_portfolio_value": config["environment"]["initial_portfolio_value"],
        "max_position_size": config["environment"]["max_position_size"],
        "transaction_cost": config["environment"]["transaction_cost"],
        "reward_scaling": config["environment"]["reward_scaling"],
        "position_penalty_scale": config["environment"]["position_penalty_scale"],
        "inventory_penalty_scale": config["environment"]["inventory_penalty_scale"],
        "trade_frequency_penalty": config["environment"]["trade_frequency_penalty"],
        "fee_model": config["environment"]["fee_model"],
        "fee_rate": config["environment"]["fee_rate"],
        "curriculum_stage": config["training"]["curriculum_stage"],
    }
    
    env = HeavyTradingEnv(
        df=train_df,
        config=env_config,
        streaming_pipeline=None,
        stream_batch_size=1000,
        max_features=68,
    )
    
    return env


def main():
    """Run validation training."""
    print("=" * 80)
    print("🚀 Optimized Parameters Validation Training")
    print("=" * 80)
    
    # Load configuration
    config_path = "optimized_config_combined.json"
    config = load_config(config_path)
    
    print(f"\n📋 Configuration loaded from: {config_path}")
    print(f"Training timesteps: {config['training']['total_timesteps']:,}")
    print(f"\n🎯 PPO Hyperparameters (Optimized):")
    for key, value in config["ppo"].items():
        print(f"  - {key}: {value}")
    
    print(f"\n🔒 Lagrange Constraints (Optimized):")
    for key, value in config["lagrange"].items():
        print(f"  - {key}: {value}")
    
    # Create environment
    print("\n🏗️  Creating environment...")
    base_env = create_environment(config)
    base_env = Monitor(base_env)
    env = DummyVecEnv([lambda: base_env])
    
    # Create model with optimized parameters
    print("\n🤖 Creating CustomPPO model...")
    model = CustomPPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config["ppo"]["learning_rate"],
        gamma=config["ppo"]["gamma"],
        n_steps=config["ppo"]["n_steps"],
        batch_size=config["ppo"]["batch_size"],
        n_epochs=config["ppo"]["n_epochs"],
        gae_lambda=config["ppo"]["gae_lambda"],
        clip_range=config["ppo"]["clip_range"],
        clip_range_vf=config["ppo"]["clip_range_vf"],
        normalize_advantage=config["ppo"]["normalize_advantage"],
        ent_coef=config["ppo"]["ent_coef"],
        vf_coef=config["ppo"]["vf_coef"],
        max_grad_norm=config["ppo"]["max_grad_norm"],
        target_kl=config["ppo"]["target_kl"],
        verbose=config["ppo"]["verbose"],
        tensorboard_log="./tensorboard",
        # Lagrange parameters
        enable_lagrange=config["lagrange"]["enable_lagrange"],
        lagrange_warmup_steps=config["lagrange"]["warmup_steps"],
        lagrange_r_target=config["lagrange"]["r_target"],
        lagrange_tolerance=config["lagrange"]["tolerance"],
        lagrange_eta=config["lagrange"]["eta"],
        lagrange_lambda_max=config["lagrange"]["lambda_max"],
    )
    
    # Setup callbacks
    print("\n📊 Setting up callbacks...")
    
    # Stats callback
    stats_callback = TrainingStatsCallback()
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"]["save_interval"],
        save_path="./models/optimized_checkpoints/",
        name_prefix="optimized_ppo",
    )
    
    callbacks = [stats_callback, checkpoint_callback]
    
    # Start training
    print("\n🎓 Starting training...")
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")
    print("=" * 80)
    
    model.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
        use_masking=False,  # Disable action masking
    )
    
    # Save final model
    final_model_path = "./models/optimized_final.zip"
    model.save(final_model_path)
    print(f"\n✅ Training complete! Final model saved to: {final_model_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("📊 Training Summary")
    print("=" * 80)
    
    if stats_callback.episode_rewards:
        rewards = stats_callback.episode_rewards
        print(f"Total episodes: {len(rewards)}")
        print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Best episode: {np.max(rewards):.2f}")
        print(f"Worst episode: {np.min(rewards):.2f}")
    
    # Lagrange statistics
    if config["lagrange"]["enable_lagrange"]:
        print(f"\n🔒 Lagrange Constraint Statistics:")
        print(f"Target SELL ratio: {config['lagrange']['r_target']:.3f}")
        print(f"Tolerance: {config['lagrange']['tolerance']:.3f}")
    
    print("\n" + "=" * 80)
    print("🎉 Validation training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
