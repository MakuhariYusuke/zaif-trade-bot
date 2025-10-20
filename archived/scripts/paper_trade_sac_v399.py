#!/usr/bin/env python3
"""
Paper Trading Script for SAC v399 Balanced Reward Model

This script simulates real-time trading using the trained SAC model
with continuous actions and improved reward function.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
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
    max_steps: int = 1000,
    delay_seconds: float = 0.1,
) -> Dict[str, Any]:
    """
    Run paper trading simulation.

    Args:
        model: Trained SAC model
        data: Historical data for simulation
        env: Trading environment
        max_steps: Maximum number of trading steps
        delay_seconds: Delay between steps for realistic simulation

    Returns:
        Dictionary with trading results
    """
    logger = get_logger(__name__)
    logger.info("Starting paper trading simulation")

    # Reset environment
    obs, info = env.reset()
    logger.info(f"Initial portfolio value: {env.portfolio_value:.2f} JPY")
    logger.info(f"Initial position: {env.position:.6f}")

    # Trading statistics
    action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL
    trades = []
    portfolio_values = [env.portfolio_value]
    positions = [env.position]

    step = 0
    done = False

    while not done and step < max_steps:
        try:
            # Get action from model (continuous action in [-1, 1])
            action, _ = model.predict(obs, deterministic=True)

            # Execute action in environment
            next_obs, reward, done, truncated, info = env.step(action)

            # Convert continuous action to discrete for logging
            if isinstance(action, (list, np.ndarray)):
                action_val = float(action[0]) if len(action) > 0 else 0.0
            else:
                action_val = float(action)

            discrete_action = ACTION_HOLD  # HOLD
            if action_val < -0.3:
                discrete_action = ACTION_SELL  # SELL
            elif action_val > 0.3:
                discrete_action = ACTION_BUY  # BUY

            action_counts[discrete_action] += 1

            # Record trade if position changed
            if abs(env.position - positions[-1]) > 0.0001:
                trade = {
                    "step": step,
                    "timestamp": data.iloc[min(step, len(data) - 1)]["timestamp"],
                    "action": discrete_action,
                    "continuous_action": action_val,
                    "position": env.position,
                    "old_position": positions[-1],
                    "portfolio_value": env.portfolio_value,
                    "reward": reward,
                    "pnl": env.portfolio_value - portfolio_values[0],
                }
                trades.append(trade)
                logger.info(
                    f"Step {step}: Action={discrete_action} ({action_val:.3f}), "
                    f"Position={env.position:.6f}, PnL={trade['pnl']:.2f}"
                )

            # Update tracking
            obs = next_obs
            portfolio_values.append(env.portfolio_value)
            positions.append(env.position)

            step += 1

            # Small delay for realistic simulation
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        except KeyboardInterrupt:
            logger.info("Paper trading interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error during paper trading at step {step}: {e}")
            break

    # Calculate final statistics
    total_return = (
        (env.portfolio_value - portfolio_values[0]) / portfolio_values[0] * 100
    )
    win_rate = len([t for t in trades if t["pnl"] > 0]) / len(trades) if trades else 0.0

    results = {
        "total_steps": step,
        "initial_portfolio": portfolio_values[0],
        "final_portfolio": env.portfolio_value,
        "total_return_percent": total_return,
        "total_trades": len(trades),
        "win_rate": win_rate,
        "action_distribution": action_counts,
        "trades": trades,
        "portfolio_history": portfolio_values,
        "position_history": positions,
    }

    logger.info("=" * 60)
    logger.info("PAPER TRADING RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Steps: {step}")
    logger.info(f"Initial Portfolio: {portfolio_values[0]:.2f} JPY")
    logger.info(f"Final Portfolio: {env.portfolio_value:.2f} JPY")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Total Trades: {len(trades)}")
    logger.info(f"Win Rate: {win_rate:.1%}")
    logger.info(f"Action Distribution: {action_counts}")
    logger.info("=" * 60)

    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save paper trading results to JSON file."""
    logger = get_logger(__name__)

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")


def main():
    """Main function for paper trading."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Configuration
    model_path = "checkpoints/sac_session/sac_v399_balanced_reward_final.zip"
    data_path = "btc_jpy_real_dataset.csv"
    output_path = "results/paper_trade_v399_results.json"
    max_steps = 2000  # Simulate 2000 steps
    delay_seconds = 0.01  # Small delay between steps

    try:
        # Load model
        model = load_model(model_path)

        # Load data
        data = load_data(data_path)

        # Create environment config
        config = EnvironmentConfig(
            max_position_size=0.01,
            transaction_cost=0.0,  # No fees for paper trading
            reward_scaling=2000.0,
            reward_clip_value=20.0,
            reward_settings={
                "use_simple_reward": True,
                "reward_scale": 2000.0,
                "reward_clip_min": -20.0,
                "reward_clip_max": 20.0,
            },
        )

        # Create environment
        env = create_paper_trading_env(config, data)

        # Run paper trading
        results = run_paper_trading(model, data, env, max_steps, delay_seconds)

        # Save results
        save_results(results, output_path)

        print("\nPaper trading completed successfully!")
        print(f"Results saved to: {output_path}")

    except Exception as e:
        logging.error(f"Paper trading failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
