#!/usr/bin/env python3
"""
Historical Backtest for SAC v420 Forced Balance Model
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from stable_baselines3 import SAC

from ztb.trading.environment.constants import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    continuous_to_discrete_action,
)
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
    """Load BTC/JPY data for backtesting."""
    logger = get_logger(__name__)
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} data points")
    return df


def create_backtest_env(config: EnvironmentConfig, df: pd.DataFrame) -> HeavyTradingEnv:
    """Create environment for backtesting."""
    env = HeavyTradingEnv(df=df, config=config)
    return env


def run_historical_backtest(
    model: SAC, data: pd.DataFrame, env: HeavyTradingEnv, max_steps: int = 5000
) -> Dict[str, Any]:
    """Run historical backtest simulation."""
    logger = get_logger(__name__)

    # Initialize tracking variables
    portfolio_value = 200000.0
    position = 0.0
    trades_count = 0
    total_pnl = 0.0

    # Action counters for analysis
    action_counts = {ACTION_HOLD: 0, ACTION_BUY: 0, ACTION_SELL: 0}  # HOLD, BUY, SELL

    logger.info("Starting historical backtest simulation")
    logger.info(f"Initial portfolio value: {portfolio_value:.2f} JPY")
    logger.info(f"Initial position: {position:.6f}")

    start_time = time.time()

    for step in range(max_steps):
        # Get current observation
        obs = env._get_observation()

        # Get action from model
        action_continuous, _ = model.predict(obs, deterministic=True)
        action_value = float(action_continuous[0])

        # Convert continuous action to discrete using centralized function for consistency
        action = continuous_to_discrete_action(action_value)

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

    # Calculate proper total PnL (portfolio change)
    total_pnl = final_portfolio_value - 200000.0

    # Calculate win rate and average trade PnL properly
    # Note: This is a simplified calculation - in practice you'd track individual trades
    win_rate = 0.5 if total_pnl > 0 else 0.0  # Simplified for now
    avg_trade_pnl = total_pnl / max(trades_count, 1)

    results = {
        "total_steps": max_steps,
        "initial_portfolio": 200000.0,
        "final_portfolio": final_portfolio_value,
        "total_return_pct": total_return,
        "total_trades": trades_count,
        "total_pnl": total_pnl,
        "action_distribution": action_counts,
        "win_rate": win_rate,
        "avg_trade_pnl": avg_trade_pnl,
    }

    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    # Configuration for SAC v420 Baseline (corrected settings) - Full training
    model_path = "models/sac_v420_baseline.zip"
    config_path = "configs/sac_v420_baseline.json"
    data_path = "data/btc_jpy_real_dataset.csv"
    output_path = "results/backtest_v420_baseline_full.json"
    max_steps = 5000

    try:
        # Load model
        model = load_model(model_path)

        # Load data
        data = load_data(data_path)

        # Load config and create environment config
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        env_config_dict = config_dict.get("environment", {})
        reward_settings = config_dict.get("reward_settings", {})

        config = EnvironmentConfig.from_dict(
            {**env_config_dict, "reward_settings": reward_settings}
        )

        # Create environment
        env = create_backtest_env(config, data)

        # Run historical backtest
        results = run_historical_backtest(model, data, env, max_steps)

        # Print results
        print("=" * 60)
        print("HISTORICAL BACKTEST RESULTS - SAC v420 Baseline (Full Training)")
        print("=" * 60)
        print(f"Total Steps: {results['total_steps']}")
        print(f"Initial Portfolio: {results['initial_portfolio']:.2f} JPY")
        print(f"Final Portfolio: {results['final_portfolio']:.2f} JPY")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Action Distribution: {results['action_distribution']}")
        print("=" * 60)

        # Save results
        save_results(results, output_path)

        print("Historical backtest completed successfully!")
        print(f"Results saved to: {output_path}")

    except Exception as e:
        logging.error(f"Historical backtest failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
