#!/usr/bin/env python3
"""
Paper Trading Script for SAC v404 with SELL Bias Correction

Tests the updated action thresholds to correct SELL bias.
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
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
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
    data: pd.DataFrame,
    env: HeavyTradingEnv,
    max_steps: int = 2000,
    delay_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Run paper trading simulation."""
    total_pnl = 0.0

    # Action counters for analysis
    action_counts = {ACTION_HOLD: 0, ACTION_BUY: 0, ACTION_SELL: 0}  # HOLD, BUY, SELL

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

        # Small delay if requested
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        if terminated or truncated:
            break

    # Calculate final results
    final_portfolio_value = portfolio_value + (
        position * env._get_current_price() if position > 0 else 0
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    max_steps = 2000

    try:
        # Load model
        model = load_model(model_path)

        # Load data
        data = load_data(data_path)

        # Create environment config with updated reward settings
        config = EnvironmentConfig(
            max_position_size=0.01,
            transaction_cost=0.0,  # No fees for paper trading
            reward_scaling=8000.0,
            reward_clip_value=80.0,
            reward_settings={
                "use_simple_reward": True,
                "reward_scale": 8000.0,
                "reward_clip_min": -80.0,
                "reward_clip_max": 80.0,
                "action_threshold_buy": 0.05,  # Updated for SELL bias correction
                "action_threshold_sell": -0.3,  # Updated for SELL bias correction
            },
        )

        # Create environment
        env = create_paper_trading_env(config, data)

        # Run paper trading
        results = run_paper_trading(model, data, env, max_steps)

        # Print results
        print("=" * 60)
        print("PAPER TRADING RESULTS (CORRECTED THRESHOLDS)")
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

        print("Paper trading completed successfully!")
        print(f"Results saved to: {output_path}")

    except Exception as e:
        logging.error(f"Paper trading failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
