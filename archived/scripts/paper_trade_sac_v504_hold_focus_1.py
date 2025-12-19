#!/usr/bin/env python3
"""
Paper Trading Script for SAC v504 Hold Focus 1

Tests the strong hold preference system.
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
