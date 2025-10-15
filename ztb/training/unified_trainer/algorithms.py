#!/usr/bin/env python3
"""
Algorithm-specific training implementations for Unified Trainer.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.utils.logging_utils import get_logger


class TrainingProgressCallback(BaseCallback):
    """Enhanced callback for monitoring training progress and action distribution."""

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.continuous_actions = []
        self.discrete_actions = []
        self.reward_history = []
        self.episode_rewards = []
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def _on_step(self) -> bool:
        # Record continuous action taken
        try:
            actions = self.locals.get('actions')
            if actions is not None:
                continuous_action = actions[0]
                if isinstance(continuous_action, np.ndarray):
                    continuous_action = continuous_action.item()
                self.continuous_actions.append(continuous_action)

                # Convert to discrete action for tracking
                discrete_action = self._continuous_to_discrete_action(continuous_action)
                self.discrete_actions.append(discrete_action)
                print(f"Debug: Recorded action {continuous_action:.6f} -> {discrete_action}")  # Debug print
            else:
                print(f"Debug: Actions not available - actions: {actions}")
        except Exception as e:
            print(f"Warning: Failed to record action: {e}")

        # Record reward
        try:
            if hasattr(self.locals, 'rewards') and self.locals['rewards'] is not None:
                reward = self.locals['rewards'][0]
                self.reward_history.append(reward)
        except Exception as e:
            print(f"Warning: Failed to record reward: {e}")

        # Log progress
        if self.n_calls % self.check_freq == 0:
            self._log_progress()

        return True

    def _continuous_to_discrete_action(self, continuous_action: float, buy_threshold: float = 0.1, sell_threshold: float = -0.1) -> int:
        """Convert continuous action (-1 to 1) to discrete action (0=HOLD, 1=BUY, 2=SELL)."""
        # Use the centralized continuous_to_discrete_action function for consistency
        return continuous_to_discrete_action(continuous_action)

    def _log_progress(self):
        """Log training progress and action distribution."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0

        # Always show progress, even if no actions recorded yet
        if self.discrete_actions:
            total_actions = len(self.discrete_actions)
            discrete_counts = np.bincount(self.discrete_actions, minlength=3)

            action_dist = {
                'HOLD': discrete_counts[0] / total_actions,
                'BUY': discrete_counts[1] / total_actions,
                'SELL': discrete_counts[2] / total_actions
            }

            print(f"Step {self.n_calls:6d} | "
                  f"Elapsed: {elapsed:6.1f}s | "
                  f"SPS: {steps_per_sec:5.1f} | "
                  f"HOLD: {action_dist['HOLD']:.1%} | "
                  f"BUY: {action_dist['BUY']:.1%} | "
                  f"SELL: {action_dist['SELL']:.1%} | "
                  f"Rewards: {len(self.reward_history)} recorded")
        else:
            # Show progress even when no actions recorded yet
            print(f"Step {self.n_calls:6d} | "
                  f"Elapsed: {elapsed:6.1f}s | "
                  f"SPS: {steps_per_sec:5.1f} | "
                  f"No actions recorded yet | "
                  f"Rewards: {len(self.reward_history)} recorded")


class BaseAlgorithmTrainer(ABC):
    """Base class for algorithm-specific trainers."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or get_logger(self.__class__.__name__)

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate configuration for this algorithm."""
        pass

    @abstractmethod
    def train(self) -> bool:
        """Execute training for this algorithm."""
        pass

    @abstractmethod
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        pass


class SACTrainer(BaseAlgorithmTrainer):
    """SAC algorithm trainer with enhanced UI and monitoring."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.model = None
        self.training_stats = {}

    def validate_config(self) -> bool:
        """Validate SAC configuration."""
        try:
            # Check required SAC hyperparameters
            sac_config = self.config.get('sac_hyperparameters', {})
            required_keys = ['learning_rate', 'buffer_size', 'learning_starts', 'batch_size']
            for key in required_keys:
                if key not in sac_config:
                    self.logger.error(f"Missing SAC hyperparameter: {key}")
                    return False

            # Check environment config
            env_config = self.config.get('environment', {})
            required_env_keys = ['initial_balance', 'transaction_cost', 'max_position_size']
            for key in required_env_keys:
                if key not in env_config:
                    self.logger.error(f"Missing environment config: {key}")
                    return False

            # Check data file
            data_path = self.config.get('data_path', 'btc_jpy_real_dataset.csv')
            if not os.path.exists(data_path):
                self.logger.error(f"Data file not found: {data_path}")
                return False

            self.logger.info("SAC configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"SAC configuration validation failed: {e}")
            return False

    def train(self) -> bool:
        """Execute SAC training with enhanced monitoring."""
        try:
            self.logger.info("🚀 Starting SAC training...")

            # Extract configurations
            sac_config = self.config.get('sac_hyperparameters', {})
            env_config_dict = self.config.get('environment', {})
            reward_settings = self.config.get('reward_settings', {})

            # Load data
            data_path = self.config.get('data_path', 'btc_jpy_real_dataset.csv')
            self.logger.info(f"📊 Loading data from {data_path}")
            df = pd.read_csv(data_path)
            self.logger.info(f"✅ Loaded {len(df)} data points")

            # Create environment config
            env_config = EnvironmentConfig()
            env_config.initial_portfolio_value = env_config_dict.get('initial_balance', 200000)
            env_config.transaction_cost = env_config_dict.get('transaction_cost', 1e-05)
            env_config.max_position_size = env_config_dict.get('max_position_size', 1.0)
            env_config.use_standardized_observations = env_config_dict.get('use_standardized_observations', True)
            env_config.curriculum_stage = env_config_dict.get('curriculum_stage', 'profit_optimized')
            env_config.use_continuous_actions = True  # SAC requires continuous actions

            # Set reward settings
            env_config.reward_scaling = reward_settings.get('reward_scale', 500.0)
            env_config.reward_clip_value = reward_settings.get('reward_clip_max', 200.0)

            # Create environment
            self.logger.info("🏗️  Creating trading environment...")
            env = HeavyTradingEnv(df=df, config=env_config)
            env = Monitor(env)

            # Create SAC model
            self.logger.info("🤖 Creating SAC model...")
            self.model = SAC(
                "MlpPolicy",
                env,
                learning_rate=sac_config.get('learning_rate', 0.0003),
                buffer_size=sac_config.get('buffer_size', 20000),
                learning_starts=sac_config.get('learning_starts', 500),
                batch_size=sac_config.get('batch_size', 128),
                tau=sac_config.get('tau', 0.005),
                gamma=sac_config.get('gamma', 0.99),
                train_freq=sac_config.get('train_freq', 1),
                gradient_steps=sac_config.get('gradient_steps', 1),
                ent_coef=sac_config.get('ent_coef', 0.01),
                target_update_interval=sac_config.get('target_update_interval', 1),
                target_entropy=sac_config.get('target_entropy', -1.0),
                verbose=0  # We'll handle logging ourselves
            )

            # Training parameters
            total_timesteps = self.config.get('total_timesteps', 50000)
            self.logger.info(f"🎯 Training for {total_timesteps:,} timesteps")

            # Create progress callback
            callback = TrainingProgressCallback(check_freq=1000)

            # Start training
            start_time = time.time()
            self.logger.info("🏃 Training started...")

            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True
            )

            # Training completed
            training_time = time.time() - start_time
            self.logger.info(f"✅ Training completed in {training_time:.1f} seconds")

            # Save model
            model_name = self.config.get('model_name', 'sac_model')
            model_path = f"models/{model_name}.zip"
            os.makedirs("models", exist_ok=True)

            self.logger.info(f"💾 Saving model to {model_path}")
            self.model.save(model_path)

            # Collect training statistics
            self.training_stats = {
                'total_timesteps': total_timesteps,
                'training_time': training_time,
                'steps_per_second': total_timesteps / training_time,
                'model_path': model_path,
                'final_reward': callback.reward_history[-1] if callback.reward_history else 0,
                'action_distribution': self._calculate_final_action_distribution(callback)
            }

            self.logger.info(f"📈 Training stats: {self.training_stats}")
            return True

        except Exception as e:
            self.logger.error(f"❌ SAC training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _calculate_final_action_distribution(self, callback: TrainingProgressCallback) -> Dict[str, float]:
        """Calculate final action distribution from callback data."""
        if not callback.discrete_actions:
            return {'HOLD': 0.0, 'BUY': 0.0, 'SELL': 0.0}

        total_actions = len(callback.discrete_actions)
        discrete_counts = np.bincount(callback.discrete_actions, minlength=3)

        return {
            'HOLD': discrete_counts[0] / total_actions,
            'BUY': discrete_counts[1] / total_actions,
            'SELL': discrete_counts[2] / total_actions
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.training_stats.copy()


class PPOTrainer(BaseAlgorithmTrainer):
    """PPO algorithm trainer (placeholder for future implementation)."""

    def validate_config(self) -> bool:
        """Validate PPO configuration."""
        self.logger.info("PPO configuration validation not yet implemented")
        return True

    def train(self) -> bool:
        """Execute PPO training."""
        self.logger.info("PPO training not yet implemented")
        return True

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {}


def create_algorithm_trainer(algorithm: str, config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> BaseAlgorithmTrainer:
    """Factory function to create algorithm-specific trainer."""
    algorithm = algorithm.lower()

    if algorithm == "sac":
        return SACTrainer(config, logger)
    elif algorithm == "ppo":
        return PPOTrainer(config, logger)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")