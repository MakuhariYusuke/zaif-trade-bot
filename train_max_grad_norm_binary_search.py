#!/usr/bin/env python3
"""
Binary search optimization for max_grad_norm parameter in PPO.
Tests different max_grad_norm values to find optimal gradient clipping.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment import HeavyTradingEnv


class TrainingCallback(BaseCallback):
    """Callback for logging training progress and action distribution."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.actions_taken = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if episode is done
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_info = info['episode']
                reward = episode_info['r']
                length = episode_info['l']

                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.episode_count += 1

                # Get actions from the environment
                if hasattr(self.training_env, 'get_last_actions'):
                    actions = self.training_env.get_last_actions()
                    if actions:
                        actions_list = [int(a) for a in actions] if hasattr(actions, '__iter__') else []
                        self.actions_taken.extend(actions_list)

                # Print episode summary every 10 episodes
                if self.episode_count % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                    print(f"Episode {self.episode_count}: Reward={reward:.4f}, Length={length}, Portfolio=N/A, Actions: H={self.actions_taken.count(0) if self.actions_taken else 0}, B={self.actions_taken.count(1) if self.actions_taken else 0}, S={self.actions_taken.count(2) if self.actions_taken else 0}")
                    print(f"Episode {self.episode_count}: Avg Reward = {avg_reward:.4f}")

        return True


def train_max_grad_norm_test(max_grad_norm: float = 0.5):
    """Train PPO model with specified max_grad_norm value."""

    print(f"\n=== Training with max_grad_norm={max_grad_norm} ===")

    # Load data
    data_path = Path(__file__).parent / "ml-dataset-enhanced.csv"
    df = pd.read_csv(data_path)

    # Create environment with simple reward
    env_config = {
        "reward_scaling": 6.0,
        "transaction_cost": 0.001,
        "max_position_size": 1.0,
        "risk_free_rate": 0.02,
        "feature_set": "full",
        "initial_portfolio_value": 1000000.0,
        "curriculum_stage": "simple_portfolio",  # Use simple reward
    }

    env = HeavyTradingEnv(
        df=df,
        config=env_config,
        streaming_pipeline=None,
        stream_batch_size=1000,
        max_features=68
    )

    # Wrap environment
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Create model with optimized parameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # optimized
        gamma=0.95,  # optimized
        gae_lambda=0.8,  # optimized
        clip_range=0.3,  # optimized
        vf_coef=0.5,  # optimized
        max_grad_norm=max_grad_norm,  # testing
        ent_coef=0.03,
        n_epochs=10,
        batch_size=64,
        n_steps=2048,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )

    # Create callback
    callback = TrainingCallback()

    # Train model
    total_timesteps = 100000
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # Calculate final statistics
    if callback.episode_rewards:
        avg_reward = np.mean(callback.episode_rewards)
        reward_std = np.std(callback.episode_rewards)
        best_reward = np.max(callback.episode_rewards)
        worst_reward = np.min(callback.episode_rewards)

        print("""
=== Training Results ===""")
        print(f"Total episodes: {len(callback.episode_rewards)}")
        print(f"Average episode reward: {avg_reward:.6f}")
        print(f"Reward std: {reward_std:.6f}")
        print(f"Best episode reward: {best_reward:.6f}")
        print(f"Worst episode reward: {worst_reward:.6f}")

        # Action distribution
        if callback.actions_taken:
            hold_count = callback.actions_taken.count(0)
            buy_count = callback.actions_taken.count(1)
            sell_count = callback.actions_taken.count(2)
            total_actions = len(callback.actions_taken)

            print("""
Action distribution:""")
            print(f"  HOLD: {hold_count} ({hold_count/total_actions*100:.1f}%)")
            print(f"  BUY: {buy_count} ({buy_count/total_actions*100:.1f}%)")
            print(f"  SELL: {sell_count} ({sell_count/total_actions*100:.1f}%)")

        # Save model
        max_grad_norm_str = f"{max_grad_norm:.4f}".replace('.', '')
        model_path = f"models/max_grad_norm_test_max_grad_norm_{max_grad_norm_str}.zip"
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        print(f"Model saved to: {model_path}")

        print(f"✅ Successfully trained: max_grad_norm_{max_grad_norm_str}")

        return avg_reward
    else:
        print("❌ No episodes completed")
        return 0.0


def main():
    parser = argparse.ArgumentParser(description='Train PPO with specific max_grad_norm value')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                       help='Max gradient norm for gradient clipping (default: 0.5)')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Train model
    train_max_grad_norm_test(args.max_grad_norm)


if __name__ == "__main__":
    main()