#!/usr/bin/env python3
"""
SAC Training Script with Continuous Actions

Trains SAC model with continuous action space to eliminate SELL bias.
Continuous actions range from -1 (SELL) to +1 (BUY), converted to discrete actions.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger


def continuous_to_discrete_action(
    continuous_action: float, buy_threshold: float = 0.1, sell_threshold: float = -0.1
) -> int:
    """
    Convert continuous action (-1 to 1) to discrete action (0=HOLD, 1=BUY, 2=SELL).

    Args:
        continuous_action: Action value from SAC (-1 to 1)
        buy_threshold: Threshold above which action is considered BUY
        sell_threshold: Threshold below which action is considered SELL

    Returns:
        Discrete action: 0=HOLD, 1=BUY, 2=SELL
    """
    if continuous_action >= buy_threshold:
        return 1  # BUY
    elif continuous_action <= sell_threshold:
        return 2  # SELL
    else:
        return 0  # HOLD


class TrainingProgressCallback(BaseCallback):
    """Callback for monitoring training progress and action distribution."""

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.continuous_actions = []  # Store continuous actions
        self.discrete_actions = []  # Store converted discrete actions
        self.reward_history = []
        self.episode_rewards = []
        self._logger = get_logger(__name__)

    def _on_step(self) -> bool:
        # Record continuous action taken
        if hasattr(self.locals, "actions"):
            continuous_action = self.locals["actions"][0]
            if isinstance(continuous_action, np.ndarray):
                continuous_action = continuous_action.item()
            self.continuous_actions.append(continuous_action)

            # Convert to discrete action for tracking
            discrete_action = continuous_to_discrete_action(continuous_action)
            self.discrete_actions.append(discrete_action)

        # Record reward
        if hasattr(self.locals, "rewards"):
            reward = self.locals["rewards"][0]
            self.reward_history.append(reward)

        # Log progress
        if self.n_calls % self.check_freq == 0:
            self._log_progress()

        return True

    def _log_progress(self):
        """Log training progress and action distribution."""
        if self.discrete_actions:
            total_actions = len(self.discrete_actions)
            discrete_counts = np.bincount(self.discrete_actions, minlength=3)

            action_dist = {
                "HOLD": discrete_counts[0] / total_actions,
                "BUY": discrete_counts[1] / total_actions,
                "SELL": discrete_counts[2] / total_actions,
            }

            self._logger.info(
                f"Step {self.n_calls}: Action Distribution - "
                f"HOLD: {action_dist['HOLD']:.1%}, "
                f"BUY: {action_dist['BUY']:.1%}, "
                f"SELL: {action_dist['SELL']:.1%}"
            )

            # Log continuous action statistics
            recent_continuous = self.continuous_actions[-1000:]  # Last 1000 actions
            if recent_continuous:
                mean_action = np.mean(recent_continuous)
                std_action = np.std(recent_continuous)
                self._logger.info(
                    f"Step {self.n_calls}: Continuous Action Stats - "
                    f"Mean: {mean_action:.3f}, Std: {std_action:.3f}"
                )

        if self.reward_history:
            recent_rewards = self.reward_history[-1000:]  # Last 1000 rewards
            avg_reward = np.mean(recent_rewards)
            self._logger.info(
                f"Step {self.n_calls}: Recent Avg Reward: {avg_reward:.4f}"
            )


def create_environment(config_path: str) -> HeavyTradingEnv:
    """Create training environment with corrected reward function."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    print("DEBUG: Loaded config_data keys:", list(config_data.keys()))
    print("DEBUG: environment keys:", list(config_data.get("environment", {}).keys()))
    print(
        "DEBUG: use_continuous_actions:",
        config_data.get("environment", {}).get("use_continuous_actions"),
    )

    # Get environment config section and pass it directly to from_dict
    env_config_dict = config_data.get("environment", {})

    # Merge reward_settings into environment config
    if "reward_settings" in config_data:
        env_config_dict["reward_settings"] = config_data["reward_settings"]

    # Override curriculum stage to ultra_profit for maximum profitability
    env_config_dict["curriculum_stage"] = "ultra_profit"

    print("DEBUG: Final env_config_dict keys:", list(env_config_dict.keys()))
    print(
        "DEBUG: Final use_continuous_actions:",
        env_config_dict.get("use_continuous_actions"),
    )

    env_config = EnvironmentConfig.from_dict(env_config_dict)
    print("DEBUG: EnvironmentConfig created")

    env = HeavyTradingEnv(df=pd.read_csv("btc_jpy_real_dataset.csv"), config=env_config)
    print("DEBUG: HeavyTradingEnv created, action_space:", type(env.action_space))

    # Wrap with monitor for logging
    env = Monitor(env)

    return env


def train_sac_model(
    env: HeavyTradingEnv,
    total_timesteps: int = 50000,
    model_save_path: str = "checkpoints/sac_corrected",
    log_interval: int = 1000,
) -> SAC:
    """Train SAC model with corrected reward function."""
    logger = get_logger(__name__)

    # Create SAC model with corrected hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50000,  # Increased buffer
        learning_starts=1000,  # Start learning later
        batch_size=256,  # Larger batch size
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",  # Auto-tune entropy
        target_update_interval=1,
        target_entropy="auto",
        verbose=1,
    )

    # Create callback for monitoring
    callback = TrainingProgressCallback(check_freq=log_interval)

    logger.info("Starting SAC training with corrected reward function")
    logger.info(f"Total timesteps: {total_timesteps}")

    start_time = time.time()

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    training_time = time.time() - start_time
    logger.info(".2f")

    # Save the trained model
    model.save(model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    return model


def validate_training(
    model: SAC, env: HeavyTradingEnv, n_episodes: int = 10
) -> Dict[str, Any]:
    """Validate the trained SAC model with continuous actions."""
    logger = get_logger(__name__)

    episode_rewards = []
    continuous_actions = []
    discrete_actions = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < 2000:  # Limit episode length
            # Get continuous action from SAC
            action_continuous, _ = model.predict(obs, deterministic=True)

            # Convert to scalar if needed
            if isinstance(action_continuous, np.ndarray):
                action_continuous = action_continuous.item()

            # Convert continuous to discrete action using consistent thresholds
            discrete_action = continuous_to_discrete_action(action_continuous)

            # Step environment with discrete action
            obs, reward, terminated, truncated, info = env.step(discrete_action)

            episode_reward += reward
            continuous_actions.append(action_continuous)
            discrete_actions.append(discrete_action)

            done = terminated or truncated
            step += 1

        episode_rewards.append(episode_reward)
        logger.info(f"Validation Episode {episode + 1}: Reward = {episode_reward:.2f}")

    # Calculate action distribution from discrete actions
    total_actions = len(discrete_actions)
    action_counts = np.bincount(discrete_actions, minlength=3)

    action_distribution = {
        "HOLD": action_counts[0] / total_actions if total_actions > 0 else 0,
        "BUY": action_counts[1] / total_actions if total_actions > 0 else 0,
        "SELL": action_counts[2] / total_actions if total_actions > 0 else 0,
    }

    # Calculate continuous action statistics
    continuous_stats = {
        "mean": np.mean(continuous_actions),
        "std": np.std(continuous_actions),
        "min": np.min(continuous_actions),
        "max": np.max(continuous_actions),
    }

    results = {
        "avg_episode_reward": np.mean(episode_rewards),
        "std_episode_reward": np.std(episode_rewards),
        "action_distribution": action_distribution,
        "continuous_action_stats": continuous_stats,
        "total_episodes": n_episodes,
        "total_actions": total_actions,
    }

    logger.info("Validation Results:")
    logger.info(f"  Average Episode Reward: {results['avg_episode_reward']:.2f}")
    logger.info(
        f"  Action Distribution: HOLD={action_distribution['HOLD']:.1%}, "
        f"BUY={action_distribution['BUY']:.1%}, SELL={action_distribution['SELL']:.1%}"
    )
    logger.info(
        f"  Continuous Action Stats: Mean={continuous_stats['mean']:.3f}, "
        f"Std={continuous_stats['std']:.3f}"
    )

    return results


def main():
    # Configuration
    config_path = "config/sac_v413_ultra_profit_config.json"
    model_save_path = "checkpoints/sac_v413_ultra_profit.zip"
    total_timesteps = 50000  # Longer training for ultra profit optimization

    try:
        # Create environment
        env = create_environment(config_path)

        # Train model
        model = train_sac_model(
            env=env, total_timesteps=total_timesteps, model_save_path=model_save_path
        )

        # Validate training
        validation_results = validate_training(model, env, n_episodes=5)

        # Save validation results
        with open(
            "results/sac_v413_ultra_profit_validation.json", "w", encoding="utf-8"
        ) as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("SAC ULTRA PROFIT TRAINING COMPLETED")
        print("=" * 60)
        print(f"Model saved: {model_save_path}")
        print(".2f")
        print(".1%")
        print(".1%")
        print(".1%")
        print(".3f")
        print("=" * 60)

    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
