#!/usr/bin/env python3
"""
Improved 5000-step SAC training with exploration fixes
探索不足を修正した改良版5000ステップSAC学習
"""

import json
import logging
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def create_improved_trading_env():
    """Create an improved trading environment with better reward function"""
    # Create simple price data
    np.random.seed(42)
    n_steps = 1000

    # Generate trending price data
    t = np.linspace(0, 4*np.pi, n_steps)
    trend = 0.1 * np.sin(t * 0.1)  # Long-term trend
    noise = np.random.normal(0, 0.005, n_steps)  # Short-term noise
    price_changes = trend + noise

    base_price = 5000000
    prices = base_price * (1 + np.cumsum(price_changes))

    # Create simple observation space (price, trend, position)
    class ImprovedTradingEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
            )
            self.current_step = 0
            self.balance = 200000.0
            self.position = 0.0
            self.prices = prices
            self.initial_balance = 200000.0
            self.prev_portfolio_value = self.initial_balance
            self.reset()

        def reset(self, seed=None, options=None):
            self.current_step = 0
            self.balance = self.initial_balance
            self.position = 0.0
            self.prev_portfolio_value = self.initial_balance
            return self._get_observation(), {}

        def step(self, action):
            # Simple trading logic
            action_value = float(action[0])

            # Execute trade
            price = self.prices[self.current_step]
            trade_executed = False

            if action_value > 0.05:  # Buy signal (lower threshold)
                if self.balance > price * 0.1:  # Can afford
                    self.position += 0.1
                    self.balance -= price * 0.1
                    trade_executed = True
            elif action_value < -0.05:  # Sell signal (lower threshold)
                if self.position > 0.1:
                    self.position -= 0.1
                    self.balance += price * 0.1
                    trade_executed = True

            # Calculate improved reward (portfolio return percentage)
            current_portfolio_value = self.balance + self.position * price
            portfolio_return = (current_portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value

            # Add small penalty for no action to encourage exploration
            action_penalty = 0.0
            if not trade_executed:
                action_penalty = -0.0001  # Small penalty for inaction

            reward = portfolio_return + action_penalty

            self.prev_portfolio_value = current_portfolio_value
            self.current_step += 1
            done = self.current_step >= len(self.prices) - 1

            return self._get_observation(), reward, done, False, {}

        def _get_observation(self):
            price = self.prices[self.current_step]
            # Simple features: price, trend, position, balance_ratio, step_ratio
            trend = (price - self.prices[max(0, self.current_step-10)]) / self.prices[max(0, self.current_step-10)]
            return np.array([
                price / 10000000,  # Normalized price
                trend,  # Price trend
                self.position,  # Current position
                self.balance / self.initial_balance,  # Balance ratio
                self.current_step / len(self.prices)  # Time progress
            ], dtype=np.float32)

    return ImprovedTradingEnv()


def main():
    """Execute improved 5000-step training"""
    logger.info("Starting improved 5000-step SAC training with exploration fixes...")

    # Create improved environment
    env = create_improved_trading_env()
    logger.info("Created improved trading environment")

    # Create SAC model with improved exploration parameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=50,  # Start learning earlier
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        ent_coef=0.5,  # Increased entropy coefficient for more exploration
        target_update_interval=1,
        verbose=1,
        # Exploration encouraging parameters
        policy_kwargs={
            'net_arch': [64, 64],
        }
    )

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path="models/checkpoints_5000step_improved/",
        name_prefix="sac_improved_5000step",
    )

    # Track training statistics
    training_stats = {
        "total_timesteps": 5000,
        "environment": "improved_trading_env",
        "model_config": {
            "learning_rate": 3e-4,
            "buffer_size": 10000,
            "batch_size": 64,
            "net_arch": [64, 64],
            "ent_coef": 0.5,
            "learning_starts": 50
        },
        "improvements": [
            "Lowered action thresholds (0.05 instead of 0.1)",
            "Added small penalty for inaction (-0.0001)",
            "Increased entropy coefficient (0.5) for more exploration",
            "Earlier learning start (50 steps)"
        ],
        "training_events": []
    }

    # Train for 5000 steps
    logger.info("Starting improved training for 5000 steps...")
    try:
        model.learn(
            total_timesteps=5000,
            callback=checkpoint_callback,
            progress_bar=True
        )
        logger.info("Training completed successfully")

        # Save final model
        model_path = "models/sac_improved_5000step_final.zip"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Update training stats
        training_stats.update({
            "training_completed": True,
            "model_path": model_path,
            "final_status": "success"
        })

    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_stats.update({
            "training_completed": False,
            "error": str(e),
            "final_status": "failed"
        })

    # Save training stats
    stats_path = "analysis/training_stats_5000step_improved.json"
    with open(stats_path, "w") as f:
        json.dump(training_stats, f, indent=2)
    logger.info(f"Training stats saved to {stats_path}")

    # Print summary
    print("\n" + "="*50)
    print("IMPROVED 5000-STEP TRAINING SUMMARY")
    print("="*50)
    print(f"Status: {training_stats['final_status']}")
    print(f"Timesteps: {training_stats['total_timesteps']}")
    print(f"Model saved: {training_stats.get('model_path', 'N/A')}")
    print("\nImprovements applied:")
    for improvement in training_stats['improvements']:
        print(f"  - {improvement}")
    print("="*50)


if __name__ == "__main__":
    main()