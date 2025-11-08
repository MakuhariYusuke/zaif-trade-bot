#!/usr/bin/env python3
"""
Debug script for investigating and resolving SELL-lock action bias in SAC agent.

This script is designed to:
1.  Run a controlled training session with detailed logging.
2.  Capture internal SAC states like actor logits and critic Q-values.
3.  Analyze reward components to understand their influence on action selection.
4.  Provide detailed reports to diagnose the root cause of the SELL bias.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv

# --- Configuration ---
LOG_FILE = "logs/debug_sell_bias.log"
OUTPUT_CSV = "data/debug_sell_bias_output.csv"
TOTAL_TIMESTEPS = 5000
CONFIG_PATH = "config/sac_v444_default.json" # A default config to use

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_simple_dataframe(steps=5000):
    """Creates a simple DataFrame for reproducible debugging."""
    dates = pd.to_datetime(pd.date_range(start="2025-01-01", periods=steps, freq="T"))
    price = 5_000_000 + np.random.randn(steps).cumsum() * 100
    df = pd.DataFrame({
        "timestamp": dates,
        "open": price,
        "high": price + 100,
        "low": price - 100,
        "close": price,
        "volume": 10 + np.random.rand(steps) * 5,
    })
    # Add minimal features required by the environment
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['RSI'] = 50
    df['MACD'] = 0
    df['BB_Upper'] = df['SMA_20'] + 1000
    df['BB_Lower'] = df['SMA_20'] - 1000
    return df.fillna(method='bfill')


class DebugCallback(BaseCallback):
    """
    A callback to log detailed information for debugging action bias.
    """
    def __init__(self, verbose=0):
        super(DebugCallback, self).__init__(verbose)
        self.log_data = []

    def _on_step(self) -> bool:
        # Access info dict from the environment
        info = self.locals["infos"][0]
        
        step_data = {
            "step": self.num_timesteps,
            "action": info.get("action"),
            "reward": self.locals["rewards"][0],
            "pnl": info.get("pnl"),
            "position": info.get("position"),
            "portfolio_value": info.get("portfolio_value"),
            "market_regime": info.get("market_regime"),
            "raw_reward": info.get("raw_reward"),
            "balance_penalty": info.get("balance_penalty"),
            "actor_logits_0": info.get("actor_logits", [None]*3)[0],
            "actor_logits_1": info.get("actor_logits", [None]*3)[1],
            "actor_logits_2": info.get("actor_logits", [None]*3)[2],
            "critic_q_value_0": info.get("critic_q_values", [None]*3)[0],
            "critic_q_value_1": info.get("critic_q_values", [None]*3)[1],
            "critic_q_value_2": info.get("critic_q_values", [None]*3)[2],
            "entropy": info.get("entropy"),
        }
        self.log_data.append(step_data)
        
        if self.num_timesteps % 100 == 0:
            logger.info(f"Step {self.num_timesteps}: Action={step_data['action']}, Reward={step_data['reward']:.4f}, Position={step_data['position']:.4f}")
            logger.info(f"  Logits: {step_data['actor_logits_0']:.4f}, {step_data['actor_logits_1']:.4f}, {step_data['actor_logits_2']:.4f}")
            logger.info(f"  Q-Vals: {step_data['critic_q_value_0']:.4f}, {step_data['critic_q_value_1']:.4f}, {step_data['critic_q_value_2']:.4f}")

        return True

    def get_log_dataframe(self):
        return pd.DataFrame(self.log_data)


def run_debug_session(total_timesteps: int, config_path: str):
    """
    Runs the debugging training session.
    """
    logger.info("--- Starting SELL-Bias Debugging Session ---")

    # 1. Load Configuration
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        env_config = config.get("environment", {})
        sac_params = config.get("sac_hyperparameters", {})
        logger.info(f"Loaded configuration from {config_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}. Exiting.")
        return

    # 2. Create Environment
    df = create_simple_dataframe(steps=total_timesteps + 200)
    env = HeavyTradingEnv(df, config=env_config)
    # Enable debug info from env
    env.enable_debug_mode() 
    logger.info("Created HeavyTradingEnv in debug mode.")

    # 3. Setup SAC Model and Callback
    debug_callback = DebugCallback()
    
    # Configure SB3 logger
    sb3_logger = configure(folder="logs/sb3_logs/", format_strings=["stdout", "csv", "tensorboard"])

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/tensorboard/",
        **sac_params
    )
    env.enable_debug_mode(model) # Pass model to env for debug info
    model.set_logger(sb3_logger)
    logger.info("SAC model created.")

    # 4. Train the model
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=debug_callback)
        logger.info("Training finished.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        # Fall-through to save any data collected before the error

    # 5. Save results
    log_df = debug_callback.get_log_dataframe()
    log_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved detailed debug log to {OUTPUT_CSV}")

    # Final action distribution analysis
    if not log_df.empty:
        action_counts = log_df["action"].value_counts(normalize=True)
        logger.info("--- Final Action Distribution ---")
        logger.info(f"\n{action_counts.to_string()}")
        logger.info("---------------------------------")

    logger.info("--- Debugging Session Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug SAC Action Bias")
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS, help="Number of training timesteps."
    )
    parser.add_argument(
        "--config", type=str, default=CONFIG_PATH, help="Path to the environment and model config JSON file."
    )
    args = parser.parse_args()

    run_debug_session(total_timesteps=args.timesteps, config_path=args.config)
