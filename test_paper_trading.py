#!/usr/bin/env python3
"""
Paper Trading Test Script for Zaif Trade Bot.

Tests a trained PPO model in paper trading mode using historical data.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import cast

# 年間取引日数（一般的に252日）
TRADING_DAYS_PER_YEAR = 252

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ztb.trading.environment import HeavyTradingEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper trading test")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model (.zip)",
    )
    parser.add_argument(
        "--data-path",
        default="ml-dataset-enhanced.csv",
        help="Path to test data (default: ml-dataset-enhanced.csv)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=100000.0,
        help="Initial balance in JPY (default: 100000.0)",
    )
    parser.add_argument(
        "--feature-set",
        default="full",
        choices=["basic", "scalping", "trend", "momentum", "full"],
        help="Feature set to use (default: full)",
    )
    parser.add_argument(
        "--timeframe",
        default="1m",
        help="Trading timeframe (default: 1m)",
    )
    parser.add_argument(
        "--episode-repeats",
        type=int,
        default=1,
        help="Number of times to repeat each episode (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path)

    # Load test data
    logger.info(f"Loading test data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} rows of data")

    # Create environment config
    env_config = {
        "reward_scaling": 1.0,
        "transaction_cost": 0.0,  # Coincheck has 0% fees
        "max_position_size": 1.0,
        "feature_set": args.feature_set,
        "timeframe": args.timeframe,
    }

    # Run paper trading simulation (create fresh env for each episode)
    logger.info("Starting paper trading test")

    total_rewards = []
    total_returns = []
    win_count = 0
    total_trades = 0
    action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}

    for episode in range(args.episodes):
        for repeat in range(args.episode_repeats):
            episode_id = episode + 1
            repeat_id = repeat + 1
            logger.info(
                f"Starting episode {episode_id}/{args.episodes}, repeat {repeat_id}/{args.episode_repeats}"
            )

            # Create fresh environment for each episode
            env = HeavyTradingEnv(df=df, config=env_config, random_start=True)

            obs, _ = env.reset()  # Unpack the tuple
            logger.debug(
                f"Episode {episode_id} (repeat {repeat_id}) starting at step: {env.current_step}, random_start: {env.random_start}"
            )
            episode_reward = 0.0
            episode_trades = 0
            initial_balance = args.initial_balance
            current_balance = initial_balance

            for _ in range(args.max_steps):
                # Get action from model (reshape obs for VecEnv-trained model)
                obs_reshaped = obs.reshape(1, -1)
                action, _ = model.predict(obs_reshaped, deterministic=True)
                action_int = cast(
                    int, action[0]
                )  # Extract from array and convert to int

                logger.debug(f"Step {_}, obs[:3]: {obs[:3]}, action: {action_int}")

                # Execute action
                obs, reward, terminated, truncated, info = env.step(action_int)
                done = terminated or truncated

                episode_reward += reward
                episode_trades += (
                    1 if action != 0 else 0
                )  # Count non-hold actions as trades

                # Count actions
                if action == 0:
                    action_counts["HOLD"] += 1
                elif action == 1:
                    action_counts["BUY"] += 1
                elif action == 2:
                    action_counts["SELL"] += 1

                # Update balance based on actual pnl
                pnl = info.get("pnl", 0.0)
                current_balance += pnl

                if done:
                    break

            # Calculate episode results
            total_return = (current_balance - initial_balance) / initial_balance
            is_win = total_return > 0

            total_rewards.append(episode_reward)
            total_returns.append(total_return)
            win_count += 1 if is_win else 0
            total_trades += episode_trades

            logger.info(
                f"Episode {episode_id} (repeat {repeat_id}): Reward={episode_reward:.2f}, Return={total_return:.2%}, Trades={episode_trades}, Win={is_win}"
            )

    # Calculate overall results
    total_episodes = args.episodes * args.episode_repeats
    avg_reward = np.mean(total_rewards)
    avg_return = np.mean(total_returns)
    win_rate = win_count / total_episodes
    total_return_all = np.prod([1 + r for r in total_returns]) - 1

    # Calculate Sharpe ratio (simplified)
    if len(total_returns) > 1:
        sharpe_ratio = (
            np.mean(total_returns)
            / np.std(total_returns)
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )  # Annualized
    else:
        sharpe_ratio = 0.0

    # Calculate max drawdown (simplified)
    cumulative_returns = np.cumprod([1 + r for r in total_returns])
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    # Print results
    logger.info("Paper trading completed")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Average reward: {avg_reward:.2f}")
    logger.info(f"Average return: {avg_return:.2%}")
    logger.info(f"Total return: {total_return_all:.2%}")
    logger.info(f"Win rate: {win_rate:.2%}")
    logger.info(f"Max drawdown: {max_drawdown:.2%}")
    logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")
    logger.info(f"Total trades: {total_trades}")
    logger.info(
        f"Action distribution: BUY={action_counts['BUY']}, SELL={action_counts['SELL']}, HOLD={action_counts['HOLD']}"
    )
    logger.info(
        f"Action percentages: BUY={action_counts['BUY']/sum(action_counts.values()):.1%}, SELL={action_counts['SELL']/sum(action_counts.values()):.1%}, HOLD={action_counts['HOLD']/sum(action_counts.values()):.1%}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())  # type: ignore[no-untyped-call]
