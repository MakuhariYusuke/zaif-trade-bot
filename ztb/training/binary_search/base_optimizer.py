#!/usr/bin/env python3
"""
Base class for hyperparameter optimization using binary search.
Provides common functionality for training callbacks and evaluation metrics.
"""

import argparse
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.utils.cli_common import CLIFormatter


class TrainingCallback(BaseCallback):
    """Callback for logging training progress and action distribution."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.actions_taken: List[int] = []
        self.episode_count = 0
        self.current_episode_actions: List[int] = []

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

                self.episode_lengths.append(length)
                self.episode_count += 1

                # Add current episode actions to total actions
                self.actions_taken.extend(self.current_episode_actions)

                # Reset for next episode
                self.current_episode_actions = []

                # Print episode summary every 10 episodes
                if self.episode_count % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                    print(f"Episode {self.episode_count}: Reward={reward:.4f}, Length={length}, Avg={avg_reward:.4f}")

        return True

    def get_training_stats(self) -> Dict[str, Union[float, int]]:
        """Get training statistics."""
        if not self.episode_rewards:
            return {"avg_reward": 0.0, "reward_std": 0.0, "best_reward": 0.0, "worst_reward": 0.0}

        avg_reward = np.mean(self.episode_rewards)
        reward_std = np.std(self.episode_rewards)
        best_reward = np.max(self.episode_rewards)
        worst_reward = np.min(self.episode_rewards)

        return {
            "avg_reward": float(avg_reward),
            "reward_std": float(reward_std),
            "best_reward": float(best_reward),
            "worst_reward": float(worst_reward)
        }

    def get_action_distribution(self) -> Dict[str, Union[int, float]]:
        """Get action distribution statistics."""
        if not self.actions_taken:
            return {"hold_count": 0, "buy_count": 0, "sell_count": 0, "total_actions": 0}

        hold_count = self.actions_taken.count(0)
        buy_count = self.actions_taken.count(1)
        sell_count = self.actions_taken.count(2)
        total_actions = len(self.actions_taken)

        return {
            "hold_count": hold_count,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_actions": total_actions,
            "hold_pct": hold_count / total_actions * 100 if total_actions > 0 else 0,
            "buy_pct": buy_count / total_actions * 100 if total_actions > 0 else 0,
            "sell_pct": sell_count / total_actions * 100 if total_actions > 0 else 0,
        }


class HyperparameterOptimizer(ABC):
    """Abstract base class for hyperparameter optimization."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent
        self.data_path = self.project_root / "ml-dataset-enhanced.csv"

        # Default environment configuration
        self.env_config = {
            "reward_scaling": 6.0,
            "transaction_cost": 0.001,
            "max_position_size": 1.0,
            "risk_free_rate": 0.02,
            "feature_set": "full",
            "initial_portfolio_value": 1000000.0,
            "curriculum_stage": "simple_portfolio",
            "reward_settings": {
                "enable_forced_diversity": False,
                "profit_bonus_multipliers": [1.0, 1.0, 1.0],
            },
        }

        # Default PPO parameters
        self.ppo_params: Dict[str, Any] = {
            "learning_rate": 5e-4,
            "gamma": 0.95,
            "gae_lambda": 0.8,
            "clip_range": 0.3,
            "vf_coef": 0.5,
            "max_grad_norm": 1.0,
            "target_kl": 0.005,
            "ent_coef": 0.05,
            "batch_size": 64,
            "n_epochs": 10,
            "n_steps": 2048,
            "verbose": 1,
            "tensorboard_log": "./tensorboard/",
            "normalize_advantage": False,
        }

    @property
    @abstractmethod
    def parameter_name(self) -> str:
        """Name of the parameter being optimized."""
        pass

    @abstractmethod
    def get_parameter_range(self) -> Tuple[float, float]:
        """Get the range (min, max) for binary search."""
        pass

    @abstractmethod
    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update PPO parameters with the test value."""
        pass

    def create_environment(self, **overrides: Any) -> HeavyTradingEnv:
        """Create training environment with optional config overrides."""
        config = self.env_config.copy()
        config.update(overrides)

        df = pd.read_csv(self.data_path)
        return HeavyTradingEnv(
            df=df,
            config=config,
            streaming_pipeline=None,
            stream_batch_size=1000,
            max_features=68
        )

    def create_model(self, env: Any) -> PPO:
        """Create PPO model with current parameters."""
        # Wrap environment
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        return PPO("MlpPolicy", env, **self.ppo_params)

    def train_model(self, total_timesteps: int = 100000) -> Tuple[PPO, TrainingCallback]:
        """Train model and return model and callback."""
        env = self.create_environment()
        model = self.create_model(env)
        callback = TrainingCallback()

        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )

        return model, callback

    def evaluate_result(self, callback: TrainingCallback) -> float:
        """Evaluate training result and return score (higher is better)."""
        stats = callback.get_training_stats()
        return stats["avg_reward"]

    def save_model(self, model: PPO, value: Union[int, float], prefix: str = "") -> str:
        """Save trained model."""
        param_str = f"{value:.6f}" if isinstance(value, float) else f"{value:03d}"
        model_path = f"models/{prefix}{self.parameter_name}_test_{self.parameter_name}_{param_str}.zip"
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        return model_path

    def print_results(self, callback: TrainingCallback) -> None:
        """Print training results."""
        stats = callback.get_training_stats()
        action_dist = callback.get_action_distribution()

        print(f"\n=== Training Results for {self.parameter_name} ===")
        print(f"Total episodes: {len(callback.episode_rewards)}")
        print(f"Average episode reward: {stats['avg_reward']:.6f}")
        print(f"Reward std: {stats['reward_std']:.6f}")
        print(f"Best episode reward: {stats['best_reward']:.6f}")
        print(f"Worst episode reward: {stats['worst_reward']:.6f}")

        print("\nAction distribution:")
        print(f"  HOLD: {action_dist['hold_count']} ({action_dist['hold_pct']:.1f}%)")
        print(f"  BUY: {action_dist['buy_count']} ({action_dist['buy_pct']:.1f}%)")
        print(f"  SELL: {action_dist['sell_count']} ({action_dist['sell_pct']:.1f}%)")

    def run_single_test(self, value: Union[int, float], total_timesteps: int = 100000) -> float:
        """Run a single training test with specified parameter value."""
        print(f"\n=== Training with {self.parameter_name}={value} ===")

        # Update parameters
        self.update_ppo_params(value)

        # Train model
        model, callback = self.train_model(total_timesteps)

        # Print results
        self.print_results(callback)

        # Save model
        self.save_model(model, value)

        # Return evaluation score
        return self.evaluate_result(callback)

    def binary_search_optimize(self, max_iterations: int = 10, total_timesteps: int = 100000) -> Tuple[Union[int, float], float]:
        """
        Perform binary search optimization.
        Returns (best_value, best_score).
        """
        min_val, max_val = self.get_parameter_range()
        best_value = min_val
        best_score = float('-inf')

        print(f"\n=== Binary Search Optimization for {self.parameter_name} ===")
        print(f"Parameter range: {min_val} to {max_val}")

        for iteration in range(max_iterations):
            # Test current best
            current_value = (min_val + max_val) / 2
            score = self.run_single_test(current_value, total_timesteps)

            print(f"Iteration {iteration + 1}: {self.parameter_name}={current_value}, score={score:.6f}")

            if score > best_score:
                best_score = score
                best_value = current_value

            # Decide next range based on score
            # This is a simple implementation - in practice, you might want more sophisticated logic
            if score > 0:  # Assuming positive score is good
                min_val = current_value
            else:
                max_val = current_value

        print(f"\nBest {self.parameter_name}: {best_value} (score: {best_score:.6f})")
        return best_value, best_score


class BinarySearchArgumentParser:
    """Common argument parser for binary search scripts."""

    @staticmethod
    def create_parser(description: str) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('--mode', choices=['single', 'binary'], default='single',
                           help=CLIFormatter.format_help('Optimization mode: single test or binary search', 'single', ['single', 'binary']))
        parser.add_argument('--max_iterations', type=int, default=10,
                           help=CLIFormatter.format_help('Maximum iterations for binary search', 10))
        parser.add_argument('--timesteps', type=int, default=100000,
                           help=CLIFormatter.format_help('Total timesteps for training', 100000))
        return parser

    @staticmethod
    def add_parameter_argument(parser: argparse.ArgumentParser, param_name: str,
                             param_type: type, default_value: Union[int, float]) -> None:
        """Add parameter-specific argument to parser."""
        if param_type == int:
            parser.add_argument(f'--{param_name}', type=int, default=default_value,
                               help=CLIFormatter.format_help(f'{param_name} value for single test', default_value))
        else:
            parser.add_argument(f'--{param_name}', type=float, default=default_value,
                               help=CLIFormatter.format_help(f'{param_name} value for single test', default_value))