#!/usr/bin/env python3
"""
Paper Trading Script for SAC v500 Equalized

Tests the equalized reward system for balanced action ratios.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from stable_baselines3 import SAC

from ztb.trading.environment.heavy_env import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
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
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} data points")
    return df


def create_paper_trading_env(
    config: EnvironmentConfig, df: pd.DataFrame
) -> HeavyTradingEnv:
    """Create environment for paper trading."""
    env = HeavyTradingEnv(df=df, config=config)
    return env


def run_paper_trading(
    model: SAC,
    data: pd.DataFrame,
    env: HeavyTradingEnv,
    max_steps: int = 2000,
    delay_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Run paper trading simulation."""
    logger = get_logger(__name__)

    # Initialize tracking variables
    portfolio_value = 200000.0
    position = 0.0
    trades_count = 0
    total_pnl = 0.0

    # Action counters for analysis
    action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL

    logger.info("Starting paper trading simulation")
    logger.info(f"Initial portfolio value: {portfolio_value:.2f} JPY")
    logger.info(f"Initial position: {position:.6f}")

    start_time = time.time()

    for step in range(max_steps):
        # Get current observation
        obs = env._get_observation()

        # Get action from model
        action_continuous, _ = model.predict(obs, deterministic=True)
        action_value = float(action_continuous[0])

        # Convert continuous action to discrete using updated thresholds
        if action_value > 0.05:  # BUY threshold
            action = 1
        elif action_value < -0.3:  # SELL threshold
            action = 2
        else:
            action = 0  # HOLD

        # Count actions
        action_counts[action] += 1

        # Execute action in environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Update tracking variables from environment
        position = env.position
        portfolio_value = env.portfolio_value
        total_pnl = env.total_pnl
        trades_count = env.trades_count

        # Log progress
        if step % 500 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Step {step}: Action={action} ({action_value:.3f}), Position={position:.6f}, PnL={total_pnl:.2f}"
            )

        if terminated or truncated:
            break

    # Calculate final results
    current_price = float(env.df.iloc[min(env.current_step, len(env.df) - 1)]["close"])
    final_portfolio_value = portfolio_value + (
        position * current_price if position > 0 else 0
    )
    total_return = (final_portfolio_value - 200000.0) / 200000.0 * 100

    results = {
        "total_steps": max_steps,
        "initial_portfolio": 200000.0,
        "final_portfolio": final_portfolio_value,
        "total_return_pct": total_return,
        "total_trades": trades_count,
        "total_pnl": total_pnl,
        "action_distribution": action_counts,
        "win_rate": 0.0,  # Simplified for this test
        "avg_trade_pnl": total_pnl / max(trades_count, 1),
    }

    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    logger = get_logger(__name__)
    logger.info(f"Saving results to {output_path}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved successfully")


def main():
    """Main function."""
    logger = get_logger(__name__)

    # Configuration
    model_name = "sac_v503_fine_tune_3"
    config_path = "config/sac_v503_fine_tune_3_config.json"
    model_path = f"models/{model_name}.zip"
    data_path = "data/btc_jpy_real_dataset.csv"
    output_path = f"results/paper_trade_{model_name}.json"
    max_steps = 5000

    logger.info(f"Starting paper trading for {model_name}")

    try:
        # Load configuration
        with open(config_path, "r") as f:
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
        results = run_paper_trading(model, df, env, max_steps)

        # Save results
        save_results(results, output_path)

        logger.info("Paper trading completed successfully!")

    except Exception as e:
        logger.error(f"Error during paper trading: {e}")
        raise


if __name__ == "__main__":
    main()
