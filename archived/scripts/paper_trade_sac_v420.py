#!/usr/bin/env python3
"""
Paper Trading Script for SAC v420 with Hold-Relaxed Configuration

Tests the hold-relaxed reward system with reduced penalties for inactivity.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.heavy_env import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger




def load_data(data_path: str) -> pd.DataFrame:
    """Load BTC/JPY data for paper trading."""
    logger = get_logger(__name__)
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return env


def run_paper_trading(
    model: SAC,
    env: HeavyTradingEnv,
    num_episodes: int = 10,
    max_steps_per_episode: Optional[int] = None,
) -> Dict[str, Any]:
    """Run paper trading simulation."""
    logger = get_logger(__name__)
    logger.info(f"Starting paper trading with {num_episodes} episodes")

    for episode in range(num_episodes):
        logger.info(f"Episode {episode + 1}/{num_episodes}")
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

        logger.info(
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

    logger.info("Paper trading completed")
    logger.info(
        f"Average Reward: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}"
    )
    logger.info(
        f"Average Portfolio Value: {summary['avg_portfolio_value']:.2f} ± {summary['std_portfolio_value']:.2f}"
    )
    logger.info(
        f"Action Distribution: HOLD={summary['action_distribution_percent']['HOLD']:.1f}%, "
        f"BUY={summary['action_distribution_percent']['BUY']:.1f}%, "
        f"SELL={summary['action_distribution_percent']['SELL']:.1f}%"
    )

    return summary


def main():
    """Main function for SAC v420 paper trading."""
    logger = get_logger(__name__)

    # Configuration
    model_path = "models/sac_v420_hold_relaxed.zip"
    data_path = "data/btc_jpy_real_dataset.csv"
    num_episodes = 10

    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Load data
    try:
        df = load_data(data_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Create environment config (same as training)
    env_config = EnvironmentConfig()
    env_config.initial_portfolio_value = 200000.0
    env_config.transaction_cost = 1e-05
    env_config.max_position_size = 1.0
    env_config.use_standardized_observations = True
    env_config.curriculum_stage = "profit_optimized"
    env_config.use_continuous_actions = True

    # Create environment
    try:
        env = create_paper_trading_env(env_config, df)
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return

    # Run paper trading
    try:
        results = run_paper_trading(model, env, num_episodes=num_episodes)

        # Save results
        output_file = f"reports/paper_trade_sac_v420_results_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("PAPER TRADING RESULTS - SAC v420 (Hold Relaxed)")
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

    except Exception as e:
        logger.error(f"Paper trading failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
