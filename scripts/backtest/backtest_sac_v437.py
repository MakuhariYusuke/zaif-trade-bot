#!/usr/bin/env python3
"""
SAC v437 Backtest Script

Backtest SAC v437 model with enhanced features and trading frequency control.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
from stable_baselines3 import SAC

from config.v437.sac_v437_enhanced_config import get_v437_config
from ztb.features.sac_v427_feature_engineering import create_v437_feature_set
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def backtest_v437_model(
    model_path: str,
    data_path: Optional[str] = None,
    output_dir: str = "backtest_experiments/v437.1",
    feature_set: str = "full",
    config_path: Optional[str] = None,
    n_episodes: int = 10,
    deterministic: bool = True,
):
    """
    Backtest SAC v437 model.

    Args:
        model_path: Path to trained model
        data_path: Path to test data
        output_dir: Output directory for results
        feature_set: Feature set to use
        config_path: Path to configuration file
        n_episodes: Number of backtest episodes
        deterministic: Whether to use deterministic actions
    """
    logger.info(f"Starting SAC v437 backtest with model: {model_path}")

    # Load configuration
    config = get_v437_config() if config_path is None else load_config(config_path)
    if data_path is None:
        data_path = config.get("data_path", "data/btc_jpy_real_dataset.csv")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"backtest_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Load and preprocess data
    logger.info(f"Loading test data from {data_path}")
    df = pd.read_csv(data_path)

    # Generate v437 features
    logger.info(f"Generating v437 features with {feature_set} set")
    features_df = create_v437_feature_set(data_path, feature_set=feature_set)

    logger.info(f"Generated {len(features_df.columns)} features")

    # Create environment
    env_config = {
        "initial_balance": config["environment"].get("initial_balance", 200000),
        "transaction_cost": config["environment"]["transaction_cost"],
        "max_position_size": config["environment"]["max_position_size"],
        "feature_set": feature_set,
        "reward_scaling": config["environment"].get("reward_scaling", 1.0),
        "risk_free_rate": config["environment"].get("risk_free_rate", 0.02),
        "timeframe": config["environment"].get("timeframe", "1m"),
        "exchange": config["environment"].get("exchange", "coincheck"),
        "stop_loss_threshold": config["environment"].get("stop_loss_threshold", 0.05),
        "max_consecutive_trades": config["environment"].get(
            "max_consecutive_trades", 10
        ),
        "min_holding_period": config["environment"].get("min_holding_period", 1),
    }

    env = HeavyTradingEnv(
        df=features_df, config=env_config, random_start=False
    )  # Use sequential data for backtest

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = SAC.load(model_path)

    # Run backtest
    results = []
    portfolio_values = []
    trades_history = []

    logger.info(f"Running {n_episodes} backtest episodes")

    for episode in range(n_episodes):
        logger.info(f"Episode {episode + 1}/{n_episodes}")

        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_trades = 0
        step_count = 0

        episode_portfolio = []
        episode_trades_data = []
        portfolio_value = env_config["initial_balance"]  # Initialize portfolio_value

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            step_count += 1

            # Record portfolio value
            portfolio_value = info.get("portfolio_value", env_config["initial_balance"])
            if portfolio_value is None:
                portfolio_value = env_config["initial_balance"]
            episode_portfolio.append(
                {
                    "step": step_count,
                    "portfolio_value": portfolio_value,
                    "reward": reward,
                }
            )

            # Record trades
            if info.get("trade_executed", False):
                episode_trades += 1
                trade_info = {
                    "episode": episode + 1,
                    "step": step_count,
                    "action": action,
                    "portfolio_value": portfolio_value,
                    "reward": reward,
                    **info,
                }
                episode_trades_data.append(trade_info)

        # Store episode results
        results.append(
            {
                "episode": episode + 1,
                "total_reward": episode_reward,
                "total_trades": episode_trades,
                "final_portfolio_value": portfolio_value,
                "total_steps": step_count,
                "avg_reward_per_step": episode_reward / step_count
                if step_count > 0
                else 0,
                "trades_per_step": episode_trades / step_count if step_count > 0 else 0,
            }
        )

        portfolio_values.extend(episode_portfolio)
        trades_history.extend(episode_trades_data)

        logger.info(
            f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
            f"Trades={episode_trades}, Final Value={portfolio_value:.2f}"
        )

    # Save results
    results_df = pd.DataFrame(results)
    portfolio_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades_history)

    results_file = os.path.join(run_dir, "backtest_results.json")
    portfolio_file = os.path.join(run_dir, "portfolio_values.csv")
    trades_file = os.path.join(run_dir, "trades_history.csv")

    results_df.to_json(results_file, orient="records", indent=2)
    portfolio_df.to_csv(portfolio_file, index=False)
    trades_df.to_csv(trades_file, index=False)

    # Calculate summary statistics
    summary = calculate_backtest_summary(results_df, portfolio_df, trades_df)

    summary_file = os.path.join(run_dir, "backtest_summary.json")
    with open(summary_file, "w") as f:
        import json

        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Backtest completed. Results saved to {run_dir}")
    logger.info(f"Summary: {summary}")

    return summary


def calculate_backtest_summary(results_df, portfolio_df, trades_df):
    """Calculate backtest summary statistics."""
    summary = {
        "total_episodes": len(results_df),
        "avg_total_reward": results_df["total_reward"].mean(),
        "std_total_reward": results_df["total_reward"].std(),
        "avg_final_portfolio_value": results_df["final_portfolio_value"].mean(),
        "std_final_portfolio_value": results_df["final_portfolio_value"].std(),
        "avg_total_trades": results_df["total_trades"].mean(),
        "avg_trades_per_step": results_df["trades_per_step"].mean(),
        "total_trades_all_episodes": trades_df.shape[0],
        "best_episode_reward": results_df["total_reward"].max(),
        "worst_episode_reward": results_df["total_reward"].min(),
        "reward_positive_ratio": (results_df["total_reward"] > 0).mean(),
        "portfolio_value_positive_ratio": (
            results_df["final_portfolio_value"] > 200000
        ).mean(),
    }

    # Calculate Sharpe-like ratio
    if len(results_df) > 1:
        returns = results_df["total_reward"]
        summary["sharpe_ratio"] = returns.mean() / (returns.std() + 1e-8)

    # Calculate max drawdown from portfolio values
    if not portfolio_df.empty:
        portfolio_values = portfolio_df.groupby("step")["portfolio_value"].mean()
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        summary["max_drawdown"] = drawdown.min()

    return summary


def load_config(config_path: str):
    """Load configuration from JSON file."""
    import json

    with open(config_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Backtest SAC v437 model")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to test data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backtest_experiments/v437.1",
        help="Output directory for results",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="full",
        choices=["full", "minimal", "high_quality"],
        help="Feature set to use",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of backtest episodes"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic actions"
    )

    args = parser.parse_args()

    # Run backtest
    summary = backtest_v437_model(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        feature_set=args.feature_set,
        config_path=args.config,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
    )

    print("Backtest Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
