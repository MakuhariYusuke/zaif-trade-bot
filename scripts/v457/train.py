#!/usr/bin/env python3
"""
v457 Training Script (Standardized & Corrected)
Based on scripts/v456/train_v456_production.py
Uses ztb.trading.environment.factory_v456 for Environment creation.

Corrections from Second Opinion:
1. Calculates Base Features (30 dim) before initialization to avoid factory fallback/warnings.
2. Loads Full Feature set (MTF, Regime is done by factory).
3. Reads `sac_hyperparameters` from config and applies them.
4. Reads and passes `execution_model` and `dynamic_threshold_mode` to Environment (if supported by wrapper/env).
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Import standardized feature calculator
from ztb.features.base_features_v456 import calculate_base_features
from ztb.trading.environment.utils.fast_intraday_env_v456_utils import (
    create_fast_intraday_env_v456,
)
from ztb.training.utils.v457_config_utils import (
    extract_env_config,
    extract_sac_params,
    load_config_dict,
    extract_seed,
)
from ztb.utils.seed_manager import set_global_seed

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("v457_train")


class SimpleTrainingCallback(BaseCallback):
    """Simple callback for training progress."""
    def __init__(self, verbose=0):
        super(SimpleTrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_reward = 0
        self.current_length = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        self.current_reward += reward
        self.current_length += 1
        
        if done:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            if self.verbose > 0:
                logger.info(f"Episode finished: reward={self.current_reward:.4f}, length={self.current_length}")
            self.current_reward = 0
            self.current_length = 0
        return True


def create_dummy_data(length=20000):
    """Create dummy OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=length, freq="1min")
    base_price = 1000.0 + np.cumsum(np.random.normal(0, 5, length))
    df = pd.DataFrame({
        "timestamp": dates,
        "open": base_price,
        "high": base_price + np.abs(np.random.normal(0, 2, length)),
        "low": base_price - np.abs(np.random.normal(0, 2, length)),
        "close": base_price + np.random.normal(0, 1, length),
        "volume": np.random.uniform(100, 1000, length),
    })
    return df

def main():
    parser = argparse.ArgumentParser(description="v457 Standardized Training")
    parser.add_argument("--steps", type=int, default=1000, help="Total timesteps to train")
    parser.add_argument("--data", type=str, default="data/btc_jpy_training_data.csv", help="Path to training data csv")
    parser.add_argument("--config", type=str, default="config/v457/base/config.yaml", help="Path to config yaml")
    parser.add_argument("--use_dummy_data", action="store_true", help="Force use of dummy data")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("v457 Training Pipeline (Corrected)")
    logger.info("=" * 60)

    # 1. Load Data
    data_path = Path(workspace_root / args.data)
    df = None
    
    if args.use_dummy_data:
        logger.info("Generating dummy data as requested...")
        df = create_dummy_data(length=20000)
    elif data_path.exists():
        logger.info(f"Loading data from {data_path}...")
        try:
            df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col=0)
            logger.info(f"Loaded {len(df)} rows.")
            if len(df) < 200:
                 logger.warning(f"Data length {len(df)} is too short. Generating dummy data.")
                 df = create_dummy_data(length=20000)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            sys.exit(1)
    else:
        logger.warning(f"Data file not found: {data_path}")
        logger.info("Generating dummy data fallback...")
        df = create_dummy_data(length=20000)

    # Calculate Base Features for v456 Factory Compliance
    logger.info(f"Pre-calculating base features for {len(df)} rows...")
    df = calculate_base_features(df, copy=False)
    logger.info("Base features calculation complete.")

    # 2. Setup Config
    config_file = Path(workspace_root / args.config)
    env_config_dict = {}
    sac_params = {}
    seed = None
    
    if config_file.exists():
        try:
            full_config = load_config_dict(config_file)
            env_config_dict = extract_env_config(full_config)
            sac_params = extract_sac_params(full_config)
            seed = extract_seed(full_config)

            logger.info(f"Loaded config from {config_file}")
            logger.info(f"SAC Hyperparameters: {sac_params}")
            if seed is not None:
                set_global_seed(seed)
                sac_params["seed"] = seed
                logger.info(f"Seed fixed: {seed}")
        except Exception as e:
            logger.warning(f"Failed to load config {config_file}: {e}. Using defaults.")
    else:
        logger.warning(f"Config file not found: {config_file}. Using defaults.")

    # 3. Create Environment via Factory
    logger.info("Initializing EnvironmentFactory...")
    env = create_fast_intraday_env_v456(df=df, env_config=env_config_dict)
    if env is None:
        logger.error("Failed to create environment.")
        sys.exit(1)
    del df

    if seed is not None:
        _, reset_info = env.reset(seed=seed)
        logger.info(f"Env reset: start_index={reset_info.get('start_index')}")
        
    # Attempt to inject extra config if environment supports it (legacy compatibility)
    if hasattr(env, "config"):
        if "execution_model" in env_config_dict:
             logger.info("Injecting execution model config to env...")
             env.config.execution_model = env_config_dict["execution_model"]

    # 4. Setup Agent
    logger.info("Initializing SAC Agent...")
    
    # Merge defaults with config params
    agent_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 1,
        "learning_rate": sac_params.get("learning_rate", 3e-4),
        "buffer_size": sac_params.get("buffer_size", 100000),
        "batch_size": sac_params.get("batch_size", 256),
        "gamma": sac_params.get("gamma", 0.99),
        "tau": sac_params.get("tau", 0.005),
        "ent_coef": sac_params.get("ent_coef", "auto"),
        "train_freq": sac_params.get("train_freq", 1),
        "gradient_steps": sac_params.get("gradient_steps", 1),
        "learning_starts": sac_params.get("learning_starts", 100),
    }

    model = SAC(**agent_kwargs)

    # 5. Train
    logger.info(f"Starting training for {args.steps} steps...")
    
    # Callback setup (simplified for v457 baseline)
    callback = SimpleTrainingCallback(verbose=1)
    
    try:
        model.learn(total_timesteps=args.steps, callback=callback, progress_bar=True)
        logger.info("Training completed.")
        
        # Save
        save_path = workspace_root / "models" / "v457" / f"sac_v457_{int(time.time())}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
