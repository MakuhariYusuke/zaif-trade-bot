#!/usr/bin/env python3
"""
Test script to check if higher entropy coefficient helps with SELL action learning.
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
                        # Count actions for this episode
                        h_count = actions_list.count(0)
                        b_count = actions_list.count(1)
                        s_count = actions_list.count(2)
                        self.actions_taken.extend(actions_list)

                # Print episode summary every 5 episodes
                if self.episode_count % 5 == 0:
                    avg_reward = np.mean(self.episode_rewards[-5:]) if self.episode_rewards else 0
                    print(f"Episode {self.episode_count}: Reward={reward:.4f}, Length={length}, Actions: H={h_count if 'h_count' in locals() else 0}, B={b_count if 'b_count' in locals() else 0}, S={s_count if 's_count' in locals() else 0}")
                    print(f"Episode {self.episode_count}: Avg Reward = {avg_reward:.4f}")

        return True


def test_high_entropy(ent_coef: float = 0.2):
    """Train PPO model with higher entropy coefficient to encourage exploration."""

    print(f"\n=== Training with ent_coef={ent_coef} (high entropy for exploration) ===")

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
        "reward_settings": {
            "enable_forced_diversity": False,  # Disable forced diversity
            "profit_bonus_multipliers": [1.0, 1.0, 1.0],
        },
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

    # Create model with high entropy for exploration
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # optimized
        gamma=0.95,  # optimized
        gae_lambda=0.8,  # optimized
        clip_range=0.3,  # optimized
        vf_coef=0.5,  # optimized
        max_grad_norm=1.0,  # optimized
        target_kl=0.005,  # optimized
        ent_coef=ent_coef,  # high entropy for exploration
        batch_size=64,  # optimized
        n_epochs=10,
        n_steps=2048,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )

    # Create callback
    callback = TrainingCallback()

    # Train model for shorter time to test
    total_timesteps = 25000  # Shorter training for testing
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # Calculate final statistics
    if callback.episode_rewards:
        avg_reward = np.mean(callback.episode_rewards)
        reward_std = np.std(callback.episode_rewards)

        print("""
=== Training Results ===""")
        print(f"Total episodes: {len(callback.episode_rewards)}")
        print(f"Average episode reward: {avg_reward:.6f}")
        print(f"Reward std: {reward_std:.6f}")

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
        ent_coef_str = f"{ent_coef:.3f}".replace(".", "_")
        model_path = f"models/test_high_entropy_ent_coef_{ent_coef_str}.zip"
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        print(f"Model saved to: {model_path}")

        print(f"✅ Successfully trained: high_entropy_ent_coef_{ent_coef_str}")

        return avg_reward, sell_count/total_actions if callback.actions_taken else 0
    else:
        print("❌ No episodes completed")
        return 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(description='Test PPO with high entropy coefficient')
    parser.add_argument('--ent_coef', type=float, default=0.2,
                       help='Entropy coefficient for exploration (default: 0.2)')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Test with high entropy
    avg_reward, sell_percentage = test_high_entropy(args.ent_coef)

    print("\n=== Summary ===")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"SELL action percentage: {sell_percentage:.1%}")


if __name__ == "__main__":
    main()