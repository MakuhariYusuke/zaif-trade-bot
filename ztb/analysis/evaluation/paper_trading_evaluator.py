#!/usr/bin/env python3
"""
Paper Trading Evaluator - Integrated paper trading functionality

This module provides comprehensive paper trading evaluation capabilities
integrated from archived paper trading scripts.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.heavy_env import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger


class PaperTradingEvaluator:
    """Paper trading evaluator for comprehensive model evaluation."""

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """Initialize paper trading evaluator.

        Args:
            config: Environment configuration for paper trading
        """
        self.logger = get_logger(__name__)
        self.config = config or EnvironmentConfig()

        # Set default paper trading configuration
        self.config.initial_portfolio_value = 200000.0
        self.config.transaction_cost = 1e-05
        self.config.max_position_size = 1.0
        self.config.use_standardized_observations = True
        self.config.curriculum_stage = "profit_optimized"
        self.config.use_continuous_actions = True

    def load_model(self, model_path: str) -> SAC:
        """Load the trained SAC model.

        Args:
            model_path: Path to the trained model file

        Returns:
            Loaded SAC model
        """
        self.logger.info(f"Loading model from {model_path}")
        model = SAC.load(model_path)
        self.logger.info("Model loaded successfully")
        return model

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load BTC/JPY data for paper trading.

        Args:
            data_path: Path to the data file

        Returns:
            Loaded and processed DataFrame
        """
        self.logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        self.logger.info(f"Loaded {len(df)} data points")
        return df

    def create_environment(self, df: pd.DataFrame) -> HeavyTradingEnv:
        """Create environment for paper trading.

        Args:
            df: DataFrame with trading data

        Returns:
            Configured HeavyTradingEnv instance
        """
        env = HeavyTradingEnv(df=df, config=self.config)
        return env

    def run_paper_trading(
        self,
        model: SAC,
        env: HeavyTradingEnv,
        num_episodes: int = 10,
        max_steps_per_episode: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run paper trading simulation.

        Args:
            model: Trained SAC model
            env: Trading environment
            num_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode

        Returns:
            Dictionary containing paper trading results
        """
        self.logger.info(f"Starting paper trading with {num_episodes} episodes")

        all_episode_results = []
        total_rewards = []
        total_portfolio_values = []
        action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}

        for episode in range(num_episodes):
            self.logger.info(f"Episode {episode + 1}/{num_episodes}")
            obs, info = env.reset()
            episode_reward = 0
            episode_portfolio_values = []
            episode_actions = []

            done = False
            step = 0
            while not done:
                if max_steps_per_episode and step >= max_steps_per_episode:
                    break

                # Get action from model
                action, _ = model.predict(obs, deterministic=True)

                # Convert continuous action to discrete for tracking
                continuous_action = action[0] if isinstance(action, np.ndarray) else action
                if continuous_action < -0.1:
                    discrete_action = ACTION_SELL  # SELL
                    action_counts["SELL"] += 1
                elif continuous_action > 0.1:
                    discrete_action = ACTION_BUY  # BUY
                    action_counts["BUY"] += 1
                else:
                    discrete_action = ACTION_HOLD  # HOLD
                    action_counts["HOLD"] += 1

                episode_actions.append(discrete_action)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_portfolio_values.append(info.get("portfolio_value", 0))

                step += 1

            # Store episode results
            episode_result = {
                "episode": episode + 1,
                "total_reward": episode_reward,
                "final_portfolio_value": episode_portfolio_values[-1]
                if episode_portfolio_values
                else 0,
                "initial_portfolio_value": episode_portfolio_values[0]
                if episode_portfolio_values
                else 0,
                "steps": step,
                "action_distribution": {
                    "HOLD": episode_actions.count(ACTION_HOLD) / len(episode_actions)
                    if episode_actions
                    else 0,
                    "BUY": episode_actions.count(ACTION_BUY) / len(episode_actions)
                    if episode_actions
                    else 0,
                    "SELL": episode_actions.count(ACTION_SELL) / len(episode_actions)
                    if episode_actions
                    else 0,
                },
            }

            all_episode_results.append(episode_result)
            total_rewards.append(episode_reward)
            total_portfolio_values.append(episode_result["final_portfolio_value"])

            self.logger.info(
                f"Episode {episode + 1} completed: Reward={episode_reward:.2f}, "
                f"Portfolio={episode_result['final_portfolio_value']:.2f}"
            )

        # Calculate summary statistics
        summary = {
            "num_episodes": num_episodes,
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "avg_portfolio_value": np.mean(total_portfolio_values),
            "std_portfolio_value": np.std(total_portfolio_values),
            "total_action_counts": action_counts,
            "action_distribution_percent": {
                "HOLD": action_counts["HOLD"] / sum(action_counts.values()) * 100
                if sum(action_counts.values()) > 0
                else 0,
                "BUY": action_counts["BUY"] / sum(action_counts.values()) * 100
                if sum(action_counts.values()) > 0
                else 0,
                "SELL": action_counts["SELL"] / sum(action_counts.values()) * 100
                if sum(action_counts.values()) > 0
                else 0,
            },
            "episode_results": all_episode_results,
        }

        self.logger.info("Paper trading completed")
        self.logger.info(
            f"Average Reward: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}"
        )
        self.logger.info(
            f"Average Portfolio Value: {summary['avg_portfolio_value']:.2f} ± {summary['std_portfolio_value']:.2f}"
        )
        self.logger.info(
            f"Action Distribution: HOLD={summary['action_distribution_percent']['HOLD']:.1f}%, "
            f"BUY={summary['action_distribution_percent']['BUY']:.1f}%, "
            f"SELL={summary['action_distribution_percent']['SELL']:.1f}%"
        )

        return summary

    def evaluate_model(
        self,
        model_path: str,
        data_path: str,
        num_episodes: int = 10,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete paper trading evaluation pipeline.

        Args:
            model_path: Path to the trained model
            data_path: Path to the evaluation data
            num_episodes: Number of episodes to run
            output_path: Optional path to save results

        Returns:
            Paper trading evaluation results
        """
        try:
            # Load model and data
            model = self.load_model(model_path)
            df = self.load_data(data_path)

            # Create environment
            env = self.create_environment(df)

            # Run paper trading
            results = self.run_paper_trading(model, env, num_episodes=num_episodes)

            # Save results if output path provided
            if output_path:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                self.logger.info(f"Results saved to {output_path}")

            return results

        except Exception as e:
            self.logger.error(f"Paper trading evaluation failed: {e}")
            raise

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted summary of paper trading results.

        Args:
            results: Paper trading results dictionary
        """
        print("\n" + "=" * 60)
        print("PAPER TRADING EVALUATION RESULTS")
        print("=" * 60)
        print(
            f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}"
        )
        print(
            f"Average Portfolio Value: {results['avg_portfolio_value']:.2f} ± {results['std_portfolio_value']:.2f}"
        )
        print(
            f"HOLD: {results['action_distribution_percent']['HOLD']:.1f}%, "
            f"BUY: {results['action_distribution_percent']['BUY']:.1f}%, "
            f"SELL: {results['action_distribution_percent']['SELL']:.1f}%"
        )
        print("=" * 60)
