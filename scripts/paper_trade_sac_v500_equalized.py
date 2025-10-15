#!/usr/bin/env python3
"""
Paper Trading Script for SAC v500 Equalized

Tests the equalized reward system for balanced action ratios.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from ztb.trading.environment.heavy_env import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.constants import ACTION_HOLD, ACTION_BUY, ACTION_SELL
from ztb.utils.logging_utils import get_logger

def load_model(model_path: str) -> SAC:
    """Load the trained SAC model."""
    logger = get_logger(__name__)
    logger.info(f"Loading model from {model_path}")
    model = SAC.load(model_path)
    logger.info("Model loaded successfully")
    return model

def load_data(data_path: str) -> pd.DataFrame:
    """Load BTC/JPY data for paper trading."""
    logger = get_logger(__name__)
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Loaded {len(df)} data points")
    return df

def create_paper_trading_env(config: EnvironmentConfig, df: pd.DataFrame) -> HeavyTradingEnv:
    """Create environment for paper trading."""
    env = HeavyTradingEnv(df=df, config=config)
    return env

def run_paper_trading(
    model: SAC,
    env: HeavyTradingEnv,
    max_steps: int = 5000
) -> Dict[str, Any]:
    """Run paper trading simulation."""
    logger = get_logger(__name__)
    logger.info(f"Starting paper trading with max_steps={max_steps}")

    obs, info = env.reset()
    total_reward = 0.0
    step_count = 0
    action_counts = {0: 0, 1: 0, 2: 0}  # BUY, HOLD, SELL

    start_time = time.time()

    while step_count < max_steps:
        action, _states = model.predict(obs, deterministic=True)

        # Convert continuous action to discrete
        if action[0] > 0.05:
            discrete_action = ACTION_BUY  # BUY
        elif action[0] < -0.3:
            discrete_action = ACTION_SELL  # SELL
        else:
            discrete_action = ACTION_HOLD  # HOLD

        action_counts[discrete_action] += 1

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        if step_count % 1000 == 0:
            logger.info(f"Step {step_count}/{max_steps}, Portfolio: {env.portfolio_value:.2f} JPY")

        if terminated or truncated:
            logger.info("Episode ended")
            break

    end_time = time.time()
    duration = end_time - start_time

    results = {
        "total_steps": step_count,
        "total_reward": total_reward,
        "duration_seconds": duration,
        "initial_portfolio": env.initial_balance,
        "final_portfolio": env.portfolio_value,
        "total_return_percent": ((env.portfolio_value - env.initial_balance) / env.initial_balance) * 100,
        "total_trades": env.total_trades,
        "action_distribution": action_counts,
        "avg_reward_per_step": total_reward / step_count if step_count > 0 else 0
    }

    logger.info(f"Paper trading completed in {duration:.2f}s")
    logger.info(f"Final Portfolio: {env.portfolio_value:.2f} JPY")
    logger.info(f"Total Return: {results['total_return_percent']:.2f}%")
    logger.info(f"Total Trades: {env.total_trades}")
    logger.info(f"Action Distribution: {action_counts}")

    return results

def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    logger = get_logger(__name__)
    logger.info(f"Saving results to {output_path}")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved successfully")

def main():
    """Main function."""
    logger = get_logger(__name__)

    # Configuration
    model_name = "sac_v500_equalized"
    config_path = "config/sac_v500_equalized_config.json"
    model_path = f"models/{model_name}.zip"
    data_path = "btc_jpy_real_dataset.csv"
    output_path = f"results/paper_trade_{model_name}.json"
    max_steps = 5000

    logger.info(f"Starting paper trading for {model_name}")

    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # Create environment config
        env_config = EnvironmentConfig.from_dict(config_data)

        # Load model
        model = load_model(model_path)

        # Load data
        df = load_data(data_path)

        # Create environment
        env = create_paper_trading_env(env_config, df)

        # Run paper trading
        results = run_paper_trading(model, env, max_steps)

        # Save results
        save_results(results, output_path)

        logger.info("Paper trading completed successfully!")

    except Exception as e:
        logger.error(f"Error during paper trading: {e}")
        raise

if __name__ == "__main__":
    main()