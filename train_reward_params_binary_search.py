#!/usr/bin/env python3
"""
Binary search optimization for reward function parameters.
Tests different reward parameter combinations to achieve balanced action distribution.
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
        self.current_episode_actions = []  # Track actions for current episode

    def _on_step(self) -> bool:
        # Get the action taken in this step
        if 'actions' in self.locals:
            action = self.locals['actions'][0]  # For vectorized env, take first action
            self.current_episode_actions.append(int(action))

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

                # Add current episode actions to total actions
                self.actions_taken.extend(self.current_episode_actions)

                # Count actions for this episode
                h_count = self.current_episode_actions.count(0)
                b_count = self.current_episode_actions.count(1)
                s_count = self.current_episode_actions.count(2)

                # Reset for next episode
                self.current_episode_actions = []

                # Print episode summary every 10 episodes
                if self.episode_count % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                    print(f"Episode {self.episode_count}: Reward={reward:.4f}, Length={length}, Actions: H={h_count}, B={b_count}, S={s_count}")
                    print(f"Episode {self.episode_count}: Avg Reward = {avg_reward:.4f}")

        return True


def train_reward_params_test(hold_penalty: float = -1.0, buy_penalty: float = -0.5, sell_reward: float = 2.0, ent_coef: float = 0.05):
    """Train PPO model with specified reward parameters."""

    print(f"\n=== Training with reward params: HOLD={hold_penalty}, BUY={buy_penalty}, SELL={sell_reward} ===")

    # Load data
    data_path = Path(__file__).parent / "ml-dataset-enhanced.csv"
    df = pd.read_csv(data_path)

    # Create environment with custom reward parameters
    env_config = {
        "reward_scaling": 6.0,
        "transaction_cost": 0.001,
        "max_position_size": 1.0,
        "risk_free_rate": 0.02,
        "feature_set": "full",
        "initial_portfolio_value": 1000000.0,
        "curriculum_stage": "simple_portfolio",  # Use simple reward
        "reward_settings": {
            "enable_forced_diversity": False,  # Disable forced diversity completely
            "profit_bonus_multipliers": [1.0, 1.0, 1.0],
            "custom_reward_params": {
                "hold_penalty": hold_penalty,
                "buy_penalty": buy_penalty,
                "sell_reward": sell_reward,
            }
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

    # Create model with optimized parameters
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
        ent_coef=ent_coef,  # parameter
        batch_size=64,  # optimized
        n_epochs=10,
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

            # Calculate balance score (lower is better balance)
            action_percentages = [
                hold_count/total_actions,
                buy_count/total_actions,
                sell_count/total_actions
            ]
            balance_score = np.std(action_percentages)  # Standard deviation of percentages

            print(f"Balance score (lower is better): {balance_score:.4f}")

        # Save model
        params_str = f"H{hold_penalty}_B{buy_penalty}_S{sell_reward}".replace(".", "").replace("-", "")
        model_path = f"models/reward_params_test_{params_str}.zip"
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        print(f"Model saved to: {model_path}")

        print(f"✅ Successfully trained: reward_params_{params_str}")

        return avg_reward, balance_score if 'balance_score' in locals() else 1.0
    else:
        print("❌ No episodes completed")
        return 0.0, 1.0


def main():
    parser = argparse.ArgumentParser(description='Train PPO with specific reward parameters')
    parser.add_argument('--hold_penalty', type=float, default=-1.0,
                       help='Penalty for HOLD action (default: -1.0)')
    parser.add_argument('--buy_penalty', type=float, default=-0.5,
                       help='Penalty for BUY action (default: -0.5)')
    parser.add_argument('--sell_reward', type=float, default=2.0,
                       help='Reward for SELL action (default: 2.0)')
    parser.add_argument('--ent_coef', type=float, default=0.05,
                       help='Entropy coefficient for exploration (default: 0.05)')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Train model
    avg_reward, balance_score = train_reward_params_test(
        args.hold_penalty, args.buy_penalty, args.sell_reward, args.ent_coef
    )

    print(f"\nFinal Results: Avg Reward={avg_reward:.4f}, Balance Score={balance_score:.4f}")


if __name__ == "__main__":
    main()