#!/usr/bin/env python3
"""
Binary search for clip_range parameter
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ztb.trading.environment import HeavyTradingEnv

class TrainingCallback(BaseCallback):
    """Callback for logging training progress"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_counts = []
        self.portfolio_values = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Log episode info
        episode_reward = sum(self.locals['rewards'])
        episode_length = len(self.locals['rewards'])
        actions = self.locals.get('actions', [])

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_count += 1

        # Count actions
        actions_list = [int(a) for a in actions] if hasattr(actions, '__iter__') else []
        action_count = Counter(actions_list)
        self.action_counts.append({
            'HOLD': action_count.get(0, 0),
            'BUY': action_count.get(1, 0),
            'SELL': action_count.get(2, 0)
        })

        # Get portfolio value from info if available
        if hasattr(self.locals, 'infos') and self.locals['infos']:
            info = self.locals['infos'][0] if isinstance(self.locals['infos'], list) else self.locals['infos']
            if isinstance(info, dict) and 'portfolio_value' in info:
                self.portfolio_values.append(info['portfolio_value'])

        # Log to TensorBoard every 10 episodes
        if self.episode_count % 10 == 0:
            self.logger.record("episode/reward", episode_reward)
            self.logger.record("episode/length", episode_length)

            if self.portfolio_values:
                self.logger.record("episode/portfolio_value", self.portfolio_values[-1])

            # Log action distribution
            total_actions = sum(action_count.values())
            if total_actions > 0:
                self.logger.record("actions/hold_ratio", action_count.get(0, 0) / total_actions)
                self.logger.record("actions/buy_ratio", action_count.get(1, 0) / total_actions)
                self.logger.record("actions/sell_ratio", action_count.get(2, 0) / total_actions)

            print(f"Episode {self.episode_count}: Reward={episode_reward:.4f}, "
                  f"Length={episode_length}, Portfolio={self.portfolio_values[-1] if self.portfolio_values else 'N/A'}, "
                  f"Actions: H={action_count.get(0, 0)}, B={action_count.get(1, 0)}, S={action_count.get(2, 0)}")

        if len(self.episode_rewards) % 10 == 0:
            avg_reward = np.mean(self.episode_rewards[-10:])
            print(f"Episode {len(self.episode_rewards)}: Avg Reward = {avg_reward:.4f}")

def train_clip_range_test(clip_range: float, reward_scaling: float = 6.0, entropy_coef: float = 0.03) -> str:
    """Train with clip_range test for 100k steps"""

    # Load data
    data_path = PROJECT_ROOT / "ml-dataset-enhanced.csv"
    df = pd.read_csv(data_path)

    # Create environment with simple reward
    env_config = {
        "reward_scaling": reward_scaling,
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

    # Create PPO model with TensorBoard logging
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # Fixed optimal learning rate
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,  # Fixed optimal gamma
        gae_lambda=0.8,  # Fixed optimal gae_lambda
        clip_range=clip_range,
        ent_coef=entropy_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tensorboard",
    )

    # Create callback
    callback = TrainingCallback()

    # Create config name for this test
    clip_range_str = f"{clip_range:.4f}".replace('.', '')
    config_name = f"clip_range_{clip_range_str}"

    print(f"Starting training with config: {config_name}")
    print(f"Reward scaling: {reward_scaling}, Entropy coef: {entropy_coef}, Clip range: {clip_range}")
    print("Training for 100,000 steps...")

    # Train for 100k steps
    model.learn(total_timesteps=100000, callback=callback)

    print("\n=== Training Results ===")
    print(f"Total episodes: {len(callback.episode_rewards)}")
    print(f"Average episode reward: {np.mean(callback.episode_rewards):.6f}")
    print(f"Reward std: {np.std(callback.episode_rewards):.6f}")
    print(f"Best episode reward: {np.max(callback.episode_rewards):.6f}")
    print(f"Worst episode reward: {np.min(callback.episode_rewards):.6f}")

    # Analyze action distribution
    if callback.action_counts:
        total_actions = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        for counts in callback.action_counts:
            for action, count in counts.items():
                total_actions[action] += count

        total_count = sum(total_actions.values())
        print("\nAction distribution:")
        for action, count in total_actions.items():
            percentage = count / total_count * 100 if total_count > 0 else 0
            print(f"  {action}: {count} ({percentage:.1f}%)")

    # Save model with config name
    model_path = PROJECT_ROOT / "models" / f"clip_range_test_{config_name}.zip"
    model_path.parent.mkdir(exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    env.close()
    return str(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO model with specified clip_range")
    parser.add_argument("--clip_range", type=float, default=0.25, help="Clip range parameter for PPO")
    parser.add_argument("--reward_scaling", type=float, default=6.0, help="Reward scaling factor")
    parser.add_argument("--entropy_coef", type=float, default=0.03, help="Entropy coefficient")

    args = parser.parse_args()

    # Create config name from clip_range
    clip_range_str = f"{args.clip_range:.4f}".replace('.', '')
    config_name = f"clip_range_{clip_range_str}"

    print(f"\n{'='*60}")
    print(f"Training with clip_range = {args.clip_range}")
    print(f"Config: {config_name}")
    print(f"{'='*60}")

    try:
        model_path = train_clip_range_test(
            clip_range=args.clip_range,
            reward_scaling=args.reward_scaling,
            entropy_coef=args.entropy_coef
        )
        print(f"✅ Successfully trained: {config_name}")

    except Exception as e:
        print(f"❌ Failed to train {config_name}: {e}")