#!/usr/bin/env python3
"""
Quick 5000-step training script for V444 analysis
課題発見のための5000ステップ学習を実行
"""

import sys
from pathlib import Path
import json
import logging
import time

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Add ztb package to path
ztb_path = project_root / "ztb"
sys.path.insert(0, str(ztb_path))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data for quick training"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=5000, freq="1h")

    # Generate sample price data with trends and volatility
    base_price = 5000000
    trend = np.sin(np.arange(5000) * 0.01) * 0.1  # Long-term trend
    noise = np.random.normal(0, 0.003, 5000)  # Short-term noise
    volatility = np.random.normal(0, 0.01, 5000)  # Volatility clusters

    price_changes = trend + noise + volatility
    close = pd.Series(base_price * (1 + price_changes.cumsum()), index=dates)

    # Generate OHLCV
    high = close * (1 + np.abs(np.random.normal(0, 0.002, 5000)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, 5000)))
    open_price = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.uniform(1000, 10000, 5000), index=dates)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "timestamp": dates,
        }
    )

    # Add basic technical indicators
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["RSI"] = 50 + 50 * (df["close"] - df["close"].shift(1)).rolling(14).apply(
        lambda x: (x > 0).sum() / len(x) - 0.5, raw=False
    ).fillna(50)

    # Add volatility indicators
    df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
    df["BB_upper"] = df["SMA_20"] + 2 * df["close"].rolling(20).std()
    df["BB_lower"] = df["SMA_20"] - 2 * df["close"].rolling(20).std()

    # Add momentum indicators
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    # Fill NaN values
    df = df.fillna(method="bfill").fillna(method="ffill")

    return df


def main():
    """Execute 5000-step training for issue analysis"""
    logger.info("Starting 5000-step training for issue analysis...")

    # Create sample data
    df = create_sample_data()
    logger.info(f"Created sample data with {len(df)} rows")

    # Create environment
    env_config = {
        "initial_balance": 200000.0,
        "transaction_cost": 1e-5,
        "max_position_size": 1.0,
        "use_continuous_actions": True,
        "reward_settings": {
            "use_simple_reward": False,
            "reward_scale": 100.0,
        },
        "features": {
            "include_multi_timeframe_features": False,  # Disable multi-timeframe features
        }
    }

    try:
        env = HeavyTradingEnv(
            df=df,
            config=env_config,
            max_features=50  # Limit features for quick training
        )
        logger.info("Environment created successfully")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return

    # Create SAC model with overfitting prevention
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef=0.1,
        target_update_interval=1,
        verbose=1,
    )

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="models/checkpoints_5000step/",
        name_prefix="sac_5000step",
    )

    # Train for 5000 steps
    logger.info("Starting training for 5000 steps...")
    training_start_time = time.time()
    try:
        model.learn(
            total_timesteps=5000,
            callback=checkpoint_callback,
            progress_bar=True
        )
        training_time = time.time() - training_start_time
        logger.info("Training completed successfully")

        # Save final model using centralized utility
        model_path = "models/sac_v444_5000step_final.zip"
        from ztb.utils.training_utils import save_model
        save_model(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Display completion using centralized utility
        from ztb.utils.training_utils import display_training_complete
        final_metrics = {
            "total_timesteps": 5000,
            "model_path": model_path,
            "data_samples": len(df),
            "features_used": 50,
            "training_completed": True
        }
        display_training_complete(final_metrics, training_time)

        # Collect basic training statistics
        training_stats = {
            "total_timesteps": 5000,
            "model_path": model_path,
            "data_samples": len(df),
            "features_used": 50,
            "training_completed": True
        }

        # Save training stats
        stats_path = "analysis/training_stats_5000step.json"
        with open(stats_path, "w") as f:
            json.dump(training_stats, f, indent=2)
        logger.info(f"Training stats saved to {stats_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()