#!/usr/bin/env python3
"""
Paper Trading Script for SAC v419 with Equalized Action Bonuses

Tests the equalized buy/sell action bonuses for symmetric trading behavior.
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
    data: pd.DataFrame,
    env: HeavyTradingEnv,
    max_steps: int = 2000,
    delay_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Run paper trading simulation."""
    total_pnl = 0.0

    # Action counters for analysis
    action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL

    logger.info("Starting paper trading simulation")
    logger.info(f"Initial portfolio value: {portfolio_value:.2f} JPY")
    logger.info(f"Initial position: {position:.6f}")

        env_config_dict = config_dict.get("environment", {})
        reward_settings = config_dict.get("reward_settings", {})

        config = EnvironmentConfig.from_dict(
            {**env_config_dict, "reward_settings": reward_settings}
        )

        # Create environment
        env = create_paper_trading_env(config, data)

        # Run paper trading
        results = run_paper_trading(model, data, env, max_steps)

        # Print results
        print("=" * 60)
        print("PAPER TRADING RESULTS - SAC v419 (Equalized Actions)")
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
