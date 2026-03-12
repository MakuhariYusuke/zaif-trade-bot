#!/usr/bin/env python3
"""
Analyze v397a_aggressive reward distribution and action patterns.

This script analyzes why the aggressive trading model resulted in 92% HOLD actions.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def analyze_rewards_and_actions(
    model_path: str, data_path: str, max_steps: int = 5000
) -> dict:
    """Analyze reward distribution and action patterns."""

    logger.info(f"Loading model from {model_path}")
    model = SAC.load(model_path)

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows of data")

    # Create environment with v397a_aggressive settings
    env_config = {
        "initial_balance": 200000,
        "transaction_cost": 0.0005,
        "max_position_size": 1.0,  # 100% position size
        "use_continuous_actions": True,
        "use_standardized_observations": True,
        "continuous_to_discrete_threshold": 0.15,  # Aggressive threshold
        "reward_settings": {
            "use_simple_reward": True,
            "reward_scale": 100.0,
            "reward_clip_min": -1.0,
            "reward_clip_max": 1.0,
            "enable_inactivity_penalty": True,
            "inactivity_penalty_rate": 0.004,
            "inactivity_penalty_window": 5,
            "inactivity_hold_threshold": 0.02,
            "enable_opportunity_cost": True,
            "opportunity_cost_rate": 0.003,
            "opportunity_cost_window": 5,
            "position_hold_threshold": 0.1,
            "enable_trade_execution_bonus": True,
            "trade_execution_bonus_rate": 0.012,
            "trade_execution_position_threshold": 0.05,
            "trade_execution_action_multiplier": 1.2,
        },
    }

    env = HeavyTradingEnv(df=df, config=env_config)

    # Collect data during simulation
    obs, _ = env.reset()

    rewards = []
    continuous_actions = []
    discrete_actions = []
    positions = []
    position_changes = []
    pnls = []

    old_position = 0.0

    for step in range(min(max_steps, len(df) - 1)):
        # Get model action
        action_continuous, _ = model.predict(obs, deterministic=True)

        # Execute step
        obs, reward, done, truncated, info = env.step(action_continuous)

        # Record data
        rewards.append(reward)
        continuous_actions.append(float(action_continuous[0]))

        current_position = info.get("position", 0.0)
        positions.append(current_position)
        position_changes.append(abs(current_position - old_position))
        pnls.append(info.get("pnl", 0.0))

        # Determine discrete action
        threshold = 0.15
        if action_continuous[0] > threshold:
            discrete_action = ACTION_BUY  # BUY
        elif action_continuous[0] < -threshold:
            discrete_action = ACTION_SELL  # SELL
        else:
            discrete_action = ACTION_HOLD  # HOLD
        discrete_actions.append(discrete_action)

        old_position = current_position

        if done or truncated:
            break

    # Analyze results
    analysis = {
        "reward_stats": {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "median": float(np.median(rewards)),
            "zero_count": int(np.sum(np.array(rewards) == 0)),
            "zero_pct": float(np.sum(np.array(rewards) == 0) / len(rewards) * 100),
            "negative_count": int(np.sum(np.array(rewards) < 0)),
            "negative_pct": float(np.sum(np.array(rewards) < 0) / len(rewards) * 100),
            "positive_count": int(np.sum(np.array(rewards) > 0)),
            "positive_pct": float(np.sum(np.array(rewards) > 0) / len(rewards) * 100),
        },
        "continuous_action_stats": {
            "mean": float(np.mean(continuous_actions)),
            "std": float(np.std(continuous_actions)),
            "min": float(np.min(continuous_actions)),
            "max": float(np.max(continuous_actions)),
            "median": float(np.median(continuous_actions)),
        },
        "discrete_action_distribution": {
            "HOLD_count": int(np.sum(np.array(discrete_actions) == 0)),
            "BUY_count": int(np.sum(np.array(discrete_actions) == 1)),
            "SELL_count": int(np.sum(np.array(discrete_actions) == 2)),
            "HOLD_pct": float(
                np.sum(np.array(discrete_actions) == 0) / len(discrete_actions) * 100
            ),
            "BUY_pct": float(
                np.sum(np.array(discrete_actions) == 1) / len(discrete_actions) * 100
            ),
            "SELL_pct": float(
                np.sum(np.array(discrete_actions) == 2) / len(discrete_actions) * 100
            ),
        },
        "position_stats": {
            "mean": float(np.mean(positions)),
            "std": float(np.std(positions)),
            "min": float(np.min(positions)),
            "max": float(np.max(positions)),
            "zero_position_pct": float(
                np.sum(np.array(positions) == 0) / len(positions) * 100
            ),
        },
        "position_change_stats": {
            "mean": float(np.mean(position_changes)),
            "std": float(np.std(position_changes)),
            "max": float(np.max(position_changes)),
            "significant_changes": int(
                np.sum(np.array(position_changes) > 0.05)
            ),  # >5% changes
        },
        "pnl_stats": {
            "mean": float(np.mean(pnls)),
            "std": float(np.std(pnls)),
            "total": float(np.sum(pnls)),
        },
    }

    # Detailed reward breakdown
    logger.info("\n" + "=" * 60)
    logger.info("=== REWARD ANALYSIS ===")
    logger.info("=" * 60)
    logger.info(f"Mean reward: {analysis['reward_stats']['mean']:.6f}")
    logger.info(f"Std reward: {analysis['reward_stats']['std']:.6f}")
    logger.info(
        f"Reward range: [{analysis['reward_stats']['min']:.6f}, {analysis['reward_stats']['max']:.6f}]"
    )
    logger.info(f"Zero rewards: {analysis['reward_stats']['zero_pct']:.2f}%")
    logger.info(f"Negative rewards: {analysis['reward_stats']['negative_pct']:.2f}%")
    logger.info(f"Positive rewards: {analysis['reward_stats']['positive_pct']:.2f}%")

    logger.info("\n" + "=" * 60)
    logger.info("=== ACTION ANALYSIS ===")
    logger.info("=" * 60)
    logger.info(
        f"Continuous action mean: {analysis['continuous_action_stats']['mean']:.4f}"
    )
    logger.info(
        f"Continuous action std: {analysis['continuous_action_stats']['std']:.4f}"
    )
    logger.info(
        f"Continuous action range: [{analysis['continuous_action_stats']['min']:.4f}, {analysis['continuous_action_stats']['max']:.4f}]"
    )
    logger.info("\nDiscrete action distribution:")
    logger.info(f"  HOLD: {analysis['discrete_action_distribution']['HOLD_pct']:.2f}%")
    logger.info(f"  BUY:  {analysis['discrete_action_distribution']['BUY_pct']:.2f}%")
    logger.info(f"  SELL: {analysis['discrete_action_distribution']['SELL_pct']:.2f}%")

    logger.info("\n" + "=" * 60)
    logger.info("=== POSITION ANALYSIS ===")
    logger.info("=" * 60)
    logger.info(
        f"Zero position: {analysis['position_stats']['zero_position_pct']:.2f}%"
    )
    logger.info(
        f"Position range: [{analysis['position_stats']['min']:.4f}, {analysis['position_stats']['max']:.4f}]"
    )
    logger.info(
        f"Significant position changes (>5%): {analysis['position_change_stats']['significant_changes']}"
    )
    logger.info(
        f"Mean position change: {analysis['position_change_stats']['mean']:.6f}"
    )

    logger.info("\n" + "=" * 60)
    logger.info("=== PnL ANALYSIS ===")
    logger.info("=" * 60)
    logger.info(f"Total PnL: {analysis['pnl_stats']['total']:.2f}")
    logger.info(f"Mean PnL per step: {analysis['pnl_stats']['mean']:.6f}")

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze v397a_aggressive rewards and actions"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/sac_session/sac_v397a_aggressive_final.zip",
        help="Path to trained model",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="btc_jpy_real_dataset.csv",
        help="Path to market data CSV",
    )
    parser.add_argument(
        "--max-steps", type=int, default=5000, help="Maximum steps to analyze"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/evaluation/v397a_reward_analysis.json",
        help="Output file for analysis results",
    )

    args = parser.parse_args()

    # Run analysis
    analysis = analyze_rewards_and_actions(
        model_path=args.model_path, data_path=args.data_path, max_steps=args.max_steps
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"\n✅ Analysis saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()
