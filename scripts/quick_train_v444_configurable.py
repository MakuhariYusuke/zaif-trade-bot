#!/usr/bin/env python3
"""
Quick Train SAC v444 Configurable - Direct Environment Training

Fast training script for SAC v444 with direct environment usage.
Supports verbose output and quick testing of different configurations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from stable_baselines3 import SAC
    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Attempting to continue with available modules...")


class DirectTrainer:
    """Direct trainer without unified trainer complexity."""

    def __init__(self, config_path: str, verbose: bool = False):
        self.config_path = config_path
        self.verbose = verbose
        self.config = self._load_config()
        self.logger = self._setup_logging()

    def _load_config(self) -> dict:
        """Load config directly from JSON."""
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Config: {self.config_path}")
        return logger

    def _load_data(self) -> pd.DataFrame:
        """Load sample data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=2000, freq="1h")
        base_price = 5000000
        price_changes = np.random.normal(0, 0.005, 2000).cumsum()
        close = pd.Series(base_price * (1 + price_changes), index=dates)
        high = close * (1 + np.abs(np.random.normal(0, 0.002, 2000)))
        low = close * (1 - np.abs(np.random.normal(0, 0.002, 2000)))
        open_price = close.shift(1).fillna(close.iloc[0])
        volume = pd.Series(np.random.uniform(1000, 10000, 2000), index=dates)

        df = pd.DataFrame({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "timestamp": dates,
        })

        # Add technical indicators
        df["SMA_20"] = df["close"].rolling(20).mean()
        df["SMA_50"] = df["close"].rolling(50).mean()
        df["RSI"] = 50
        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["BB_Upper"] = df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
        df["BB_Lower"] = df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()

        return df.ffill().bfill()

    def _prepare_env_config(self) -> dict:
        """Prepare environment config by expanding nested parameters."""
        env_config = self.config['environment'].copy()

        # Expand nested configs
        if 'behavior_optimization' in env_config:
            env_config.update(env_config['behavior_optimization'])

        if 'action_bonuses' in env_config:
            env_config.update(env_config['action_bonuses'])

        return env_config

    def train(self) -> bool:
        """Execute training."""
        try:
            self.logger.info("="*80)
            self.logger.info("🚀 Starting direct SAC v444 training")
            self.logger.info("="*80)

            # Load data
            df = self._load_data()

            # Setup environment
            env_config = self._prepare_env_config()
            env = HeavyTradingEnv(df, env_config)

            # Create model
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=0.0003,
                buffer_size=1000000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                ent_coef='auto_1.0',
                target_update_interval=1,
                verbose=2 if self.verbose else 0,
            )

            # Train for 2000 steps
            self.logger.info("Training for 2000 timesteps...")
            model.learn(total_timesteps=2000)

            self.logger.info("✅ Training completed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
            return False


def main() -> bool:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Quick Train SAC v444 - Direct Environment Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    try:
        print("🚀 Quick Train SAC v444 - Direct Environment Training")
        print(f"Configuration: {args.config}")
        if args.verbose:
            print("Verbose mode enabled")

        trainer = DirectTrainer(args.config, verbose=args.verbose)
        success = trainer.train()

        if success:
            print("✅ SAC v444 training completed successfully!")
            return True
        else:
            print("❌ SAC v444 training failed!")
            return False

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)