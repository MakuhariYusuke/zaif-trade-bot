#!/usr/bin/env python3
"""
SAC v440 Backtest Script - Pure PnL-Based Model

Backtest the simplified PnL-based SAC model.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from stable_baselines3 import SAC

from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.training.environments.heavy_trading_env import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_environment(config: dict, data_path: str):
    """Create environment matching training setup."""
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows of market data")

    # Basic feature columns for PnL-focused trading
    feature_columns = []
    if "features" in config:
        for category in ["technical_indicators", "price_features", "volatility_features", "volume_features", "momentum_features", "trend_features", "oscillator_features", "support_resistance"]:
            if category in config["features"]:
                feature_columns.extend(config["features"][category])

    # Remove duplicates
    feature_columns = list(set(feature_columns))

    # Add standard features
    feature_columns.extend(["balance_norm", "position", "unrealized_norm"])

    logger.info(f"Using {len(feature_columns)} features: {feature_columns}")

    # Create reward settings for pure PnL
    reward_settings: RewardSettings = {
        "position_soft_cap": 1.0,
        "position_penalty_scale": 0.0,  # No position penalty for pure PnL
        "position_penalty_exp": 1.0,
        "inventory_window": 10,
        "inventory_penalty_scale": 0.0,  # No inventory penalty
        "trade_frequency_penalty": 0.0,  # No frequency penalty
        "trade_frequency_halflife": 50,
        "trade_cooldown_steps": 0,
        "trade_cooldown_penalty": 0.0,
        "max_consecutive_trades": 100,
        "consecutive_trade_penalty": 0.0,
        "volatility_window": 20,
        "volatility_penalty_scale": 0.0,  # No volatility penalty
        "sharpe_bonus_scale": 0.0,  # No Sharpe bonus
        "sortino_bonus_scale": 0.0,
        "calmar_bonus_scale": 0.0,
        "reward_clip_value": 10.0,
        "profit_bonus_multipliers": [1.0],  # Simple profit bonus
        "enable_forced_diversity": False,
        "custom_reward_params": {},
        "balance_penalty": 0.0,  # No balance penalty
        "balance_penalty_tolerance": 0.1,
        "profit_weight": config["reward_function"]["base_profit_bonus"],
        "risk_weight": config["reward_function"]["loss_penalty_coeff"],
        "consistency_weight": 0.0,
        "ultra_profit_multiplier": 1.0,
        "ultra_risk_multiplier": 1.0
    }

    # Create environment config (simplified)
    env_config = EnvironmentConfig(
        initial_portfolio_value=config["environment"]["initial_balance"],
        transaction_cost=config["environment"]["commission"],
        max_position_size=config["environment"]["max_position_size"],
        reward_scaling=config["environment"]["reward_scaling"],
        feature_names=feature_columns,
        curriculum_stage="pnl_focused",
        correlation_reduction=False,  # Disable for simplicity
        stop_loss_threshold=0.5,  # High threshold to avoid interference
        max_consecutive_trades=100,
        min_holding_period=1,
        reward_position_soft_cap=1.0,
        reward_position_penalty_scale=0.0,  # No position penalty
        reward_position_penalty_exponent=1.0,
        reward_inventory_window=10,
        reward_inventory_penalty_scale=0.0,  # No inventory penalty
        reward_trade_frequency_penalty=0.0,  # No frequency penalty
        reward_trade_frequency_halflife=50,
        reward_trade_cooldown_steps=0,
        reward_trade_cooldown_penalty=0.0,
        reward_max_consecutive_trades=100,
        reward_consecutive_trade_penalty=0.0,
        reward_volatility_window=20,
        reward_volatility_penalty_scale=0.0,  # No volatility penalty
        reward_sharpe_bonus_scale=0.0,  # No Sharpe bonus
        reward_clip_value=10.0,
        reward_profit_bonus_multipliers=[1.0],  # Simple profit bonus
        enable_forced_diversity=False
    )
    # Add missing attributes for compatibility
    env_config.initial_balance = config["environment"]["initial_balance"]
    env_config.max_steps = config["environment"]["max_steps"]
    env_config.slippage = config["environment"]["slippage"]
    env_config.commission = config["environment"]["commission"]

    # Create environment
    env = HeavyTradingEnv(df, env_config, feature_columns=feature_columns, reward_settings=reward_settings)
    logger.info("Environment created successfully")

    return env


def run_backtest(model_path: str, config: dict, num_episodes: int = 10):
    """Run backtest and collect results."""
    # Load model
    model = SAC.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    # Create environment
    env = create_environment(config, "data/btc_jpy_featured_dataset.csv")

    results = []
    total_rewards = []
    total_trades = []
    total_returns = []

    logger.info(f"🚀 Starting backtest with {num_episodes} episodes...")

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_trades = 0
        done = False
        step_count = 0

        while not done and step_count < config["environment"]["max_steps"]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            step_count += 1

            # Count trades (position changes)
            if hasattr(env, 'last_action') and env.last_action is not None:
                if abs(action[0]) > 0.01:  # Significant position change
                    episode_trades += 1

        # Calculate return percentage
        final_balance = env.balance
        initial_balance = config["environment"]["initial_balance"]
        return_pct = ((final_balance - initial_balance) / initial_balance) * 100

        results.append({
            "episode": episode + 1,
            "reward": float(episode_reward),
            "trades": episode_trades,
            "return_pct": float(return_pct),
            "final_balance": float(final_balance)
        })

        total_rewards.append(episode_reward)
        total_trades.append(episode_trades)
        total_returns.append(return_pct)

        logger.info(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Trades={episode_trades}, Return={return_pct:.2f}%")

    # Calculate summary statistics
    avg_reward = np.mean(total_rewards)
    avg_trades = np.mean(total_trades)
    avg_return = np.mean(total_returns)
    win_rate = np.mean([1 if r > 0 else 0 for r in total_returns]) * 100

    # Calculate Sharpe ratio (simplified)
    if len(total_returns) > 1:
        sharpe_ratio = avg_return / (np.std(total_returns) + 1e-8)
    else:
        sharpe_ratio = 0

    summary = {
        "total_episodes": num_episodes,
        "average_reward": float(avg_reward),
        "average_trades": float(avg_trades),
        "win_rate": float(win_rate),
        "total_return": float(avg_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(min(total_returns)) if total_returns else 0,
        "model_path": model_path,
        "config_version": config["version"],
        "approach": "pure_pnl_based"
    }

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Backtest SAC v440 PnL Model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/v440/sac_v440_pnl_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/v440",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration: {config['model_name']}")

    # Run backtest
    results, summary = run_backtest(args.model, config, args.episodes)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = output_dir / "backtest_results_v440.json"
    with open(results_file, 'w') as f:
        json.dump({
            "summary": summary,
            "episodes": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    logger.info("✅ Backtest completed! Results saved to results/v440/backtest_results_v440.json")

    # Print summary
    logger.info("📊 Backtest Summary:")
    logger.info(f"Total Episodes: {summary['total_episodes']}")
    logger.info(f"Average Reward: {summary['average_reward']:.2f}")
    logger.info(f"Average Trades: {summary['average_trades']:.1f}")
    logger.info(f"Win Rate: {summary['win_rate']:.1f}%")
    logger.info(f"Total Return: {summary['total_return']:.2f}%")
    logger.info(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {summary['max_drawdown']:.2f}%")


if __name__ == "__main__":
    main()