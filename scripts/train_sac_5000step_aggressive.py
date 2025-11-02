#!/usr/bin/env python3
"""
Aggressive exploration 5000-step SAC training
積極的な探索を促進した5000ステップSAC学習
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


def create_aggressive_trading_env():
    """Create a trading environment with aggressive exploration encouragement"""
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
    class AggressiveTradingEnv(gym.Env):
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

            # Execute trade with very low thresholds
            price = self.prices[self.current_step]
            trade_executed = False

            if action_value > 0.02:  # Very low buy threshold
                if self.balance > price * 0.1:  # Can afford
                    self.position += 0.1
                    self.balance -= price * 0.1
                    trade_executed = True
            elif action_value < -0.02:  # Very low sell threshold
                if self.position > 0.1:
                    self.position -= 0.1
                    self.balance += price * 0.1
                    trade_executed = True

            # Calculate reward with stronger exploration encouragement
            current_portfolio_value = self.balance + self.position * price
            portfolio_return = (current_portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value

            # Stronger penalty for no action to force exploration
            action_penalty = 0.0
            if not trade_executed:
                action_penalty = -0.001  # Stronger penalty for inaction

            # Add bonus for taking actions
            action_bonus = 0.0005 if trade_executed else 0.0

            reward = portfolio_return + action_penalty + action_bonus

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

    return AggressiveTradingEnv()


def main():
    """Execute aggressive exploration 5000-step training"""
    logger.info("Starting aggressive exploration 5000-step SAC training...")

    # Create aggressive environment
    env = create_aggressive_trading_env()
    logger.info("Created aggressive exploration trading environment")

    # Create SAC model with maximum exploration parameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # Higher learning rate
        buffer_size=10000,
        learning_starts=25,  # Very early learning start
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        ent_coef=1.0,  # Maximum entropy coefficient for exploration
        target_update_interval=1,
        verbose=1,
        # Maximum exploration parameters
        policy_kwargs={
            'net_arch': [64, 64],
        }
    )

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path="models/checkpoints_5000step_aggressive/",
        name_prefix="sac_aggressive_5000step",
    )

    # Track training statistics
    training_stats = {
        "total_timesteps": 5000,
        "environment": "aggressive_trading_env",
        "model_config": {
            "learning_rate": 5e-4,
            "buffer_size": 10000,
            "batch_size": 64,
            "net_arch": [64, 64],
            "ent_coef": 1.0,
            "learning_starts": 25
        },
        "aggressive_improvements": [
            "Very low action thresholds (±0.02)",
            "Stronger penalty for inaction (-0.001)",
            "Added bonus for taking actions (+0.0005)",
            "Maximum entropy coefficient (1.0)",
            "Higher learning rate (5e-4)",
            "Very early learning start (25 steps)"
        ],
        "training_events": []
    }

    # Train for 5000 steps
    logger.info("Starting aggressive exploration training for 5000 steps...")
    try:
        model.learn(
            total_timesteps=5000,
            callback=checkpoint_callback,
            progress_bar=True
        )
        logger.info("Training completed successfully")

        # Save final model
        model_path = "models/sac_aggressive_5000step_final.zip"
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
    stats_path = "analysis/training_stats_5000step_aggressive.json"
    with open(stats_path, "w") as f:
        json.dump(training_stats, f, indent=2)
    logger.info(f"Training stats saved to {stats_path}")

    # Print summary
    print("\n" + "="*50)
    print("AGGRESSIVE EXPLORATION 5000-STEP TRAINING SUMMARY")
    print("="*50)
    print(f"Status: {training_stats['final_status']}")
    print(f"Timesteps: {training_stats['total_timesteps']}")
    print(f"Model saved: {training_stats.get('model_path', 'N/A')}")
    print("\nAggressive improvements applied:")
    for improvement in training_stats['aggressive_improvements']:
        print(f"  - {improvement}")
    print("="*50)


if __name__ == "__main__":
    main()