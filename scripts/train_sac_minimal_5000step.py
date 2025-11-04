#!/usr/bin/env python3
"""
Minimal 5000-step SAC training for issue analysis
課題発見のための最小限の5000ステップ学習
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


def create_minimal_trading_env():
    """Create a minimal trading environment for testing"""
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
    class MinimalTradingEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
            )
            self.current_step = 0
            self.balance = 200000.0
            self.position = 0.0
            self.prices = prices
            self.reset()

        def reset(self, seed=None, options=None):
            self.current_step = 0
            self.balance = 200000.0
            self.position = 0.0
            return self._get_observation(), {}

        def step(self, action):
            # Simple trading logic
            action_value = float(action[0])

            # Execute trade
            price = self.prices[self.current_step]
            if action_value > 0.1:  # Buy signal
                if self.balance > price * 0.1:  # Can afford
                    self.position += 0.1
                    self.balance -= price * 0.1
            elif action_value < -0.1:  # Sell signal
                if self.position > 0.1:
                    self.position -= 0.1
                    self.balance += price * 0.1

            # Calculate reward (portfolio value change)
            portfolio_value = self.balance + self.position * price
            reward = portfolio_value - 200000.0  # Change from initial

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
                self.balance / 200000.0,  # Balance ratio
                self.current_step / len(self.prices)  # Time progress
            ], dtype=np.float32)

    return MinimalTradingEnv()


def main():
    """Execute minimal 5000-step training"""
    logger.info("Starting minimal 5000-step SAC training for issue analysis...")

    # Create minimal environment
    env = create_minimal_trading_env()
    logger.info("Created minimal trading environment")

    # Create SAC model with overfitting prevention parameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=10000,  # Smaller buffer for quick training
        learning_starts=100,
        batch_size=64,  # Smaller batch for quick training
        tau=0.005,
        gamma=0.99,
        ent_coef=0.1,
        target_update_interval=1,
        verbose=1,
        # Overfitting prevention parameters
        policy_kwargs={
            'net_arch': [64, 64],  # Smaller network
        }
    )

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path="models/checkpoints_5000step_minimal/",
        name_prefix="sac_minimal_5000step",
    )

    # Track training statistics
    training_stats = {
        "total_timesteps": 5000,
        "environment": "minimal_trading_env",
        "model_config": {
            "learning_rate": 3e-4,
            "buffer_size": 10000,
            "batch_size": 64,
            "net_arch": [64, 64]
        },
        "training_events": []
    }

    # Train for 5000 steps
    logger.info("Starting training for 5000 steps...")
    try:
        model.learn(
            total_timesteps=5000,
            callback=checkpoint_callback,
            progress_bar=True
        )
        logger.info("Training completed successfully")

        # Save final model
        model_path = "models/sac_minimal_5000step_final.zip"
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
    stats_path = "analysis/training_stats_5000step_minimal.json"
    with open(stats_path, "w") as f:
        json.dump(training_stats, f, indent=2)
    logger.info(f"Training stats saved to {stats_path}")

    # Print summary
    print("\n" + "="*50)
    print("MINIMAL 5000-STEP TRAINING SUMMARY")
    print("="*50)
    print(f"Status: {training_stats['final_status']}")
    print(f"Timesteps: {training_stats['total_timesteps']}")
    print(f"Model saved: {training_stats.get('model_path', 'N/A')}")
    print("="*50)


if __name__ == "__main__":
    main()