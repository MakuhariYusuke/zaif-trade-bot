#!/usr/bin/env python3
"""
SAC v444 Training Script with Config Support
設定ファイルからパラメータを読み込み、カスタマイズ可能なtraining実行
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from stable_baselines3 import SAC

    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Attempting to continue with available modules...")


class ConfigurableTrainer:
    """設定ベースのtrainer"""

    def __init__(self, config_path: str, verbose: bool = False):
        self.config_path = config_path
        self.verbose = verbose
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.results = {}

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _setup_logging(self) -> logging.Logger:
        """ログ設定"""
        log_dir = Path(self.config["logging"]["tensorboard_log"]).parent.parent
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Model: {self.config['model_name']}")

        return logger

    def _load_or_create_data(self) -> pd.DataFrame:
        """データを読み込むまたは生成"""
        data_path = self.config["training"]["data_config"].get("csv_path")

        if data_path and Path(data_path).exists():
            self.logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            self.logger.info(f"Data shape: {df.shape}")
            return df
        else:
            self.logger.info("Creating sample data...")
            return self._create_sample_data()

    def _create_sample_data(self, periods: int = 2000) -> pd.DataFrame:
        """サンプルデータを生成"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=periods, freq="1h")

        # Generate price data
        base_price = 5000000
        price_changes = np.random.normal(0, 0.005, periods).cumsum()
        close = pd.Series(base_price * (1 + price_changes), index=dates)

        # Generate OHLCV
        high = close * (1 + np.abs(np.random.normal(0, 0.002, periods)))
        low = close * (1 - np.abs(np.random.normal(0, 0.002, periods)))
        open_price = close.shift(1).fillna(close.iloc[0])
        volume = pd.Series(np.random.uniform(1000, 10000, periods), index=dates)

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

        # Add technical indicators
        df["SMA_20"] = df["close"].rolling(20).mean()
        df["SMA_50"] = df["close"].rolling(50).mean()
        df["RSI"] = 50
        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["BB_Upper"] = (
            df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
        )
        df["BB_Lower"] = (
            df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()
        )

        return df.ffill().bfill()

    def train(self) -> Tuple[str, str]:
        """Training実行"""
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 Starting training: {self.config['description']}")
        self.logger.info("=" * 80)

        try:
            # Load data
            df = self._load_or_create_data()

            # Setup environment
            self.logger.info("Setting up trading environment...")
            env_config = self._prepare_env_config()
            env = HeavyTradingEnv(df, env_config)

            # Create model
            self.logger.info("Creating SAC model...")
            model_config = self.config["training"]["sac_hyperparameters"]

            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=model_config.get("learning_rate", 0.0003),
                buffer_size=model_config.get("buffer_size", 1000000),
                learning_starts=model_config.get("learning_starts", 1000),
                batch_size=model_config.get("batch_size", 256),
                tau=model_config.get("tau", 0.005),
                gamma=model_config.get("gamma", 0.99),
                ent_coef="auto_1.0"
                if model_config.get("ent_coef") == "auto_1.0"
                else 0.01,
                target_update_interval=model_config.get("target_update_interval", 1),
                verbose=2 if self.verbose else 0,
            )

            # Train
            total_timesteps = self.config["training"].get("total_timesteps", 2000)
            self.logger.info(f"Training for {total_timesteps} timesteps...")

            model.learn(total_timesteps=total_timesteps)

            self.logger.info("✅ Training completed")

            # Save model
            model_path = self._save_model(model)

            # Generate summary
            summary = self._generate_summary()

            return model_path, summary

        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
            raise

    def _prepare_env_config(self) -> Dict[str, Any]:
        """環境設定を準備"""
        env_config = self.config["environment"].copy()

        # Expand nested configs
        if "behavior_optimization" in env_config:
            env_config.update(env_config["behavior_optimization"])

        if "action_bonuses" in env_config:
            env_config.update(env_config["action_bonuses"])

        # Add curriculum_stage from training config if available
        if (
            "training" in self.config
            and "curriculum_learning" in self.config["training"]
        ):
            curriculum_config = self.config["training"]["curriculum_learning"]
            if "curriculum_stage" in curriculum_config:
                env_config["curriculum_stage"] = curriculum_config["curriculum_stage"]
                self.logger.info(
                    f"Curriculum stage set to: {curriculum_config['curriculum_stage']}"
                )

        return env_config

    def _save_model(self, model) -> str:
        """モデルを保存"""
        model_dir = Path(self.config["model_save_path"]).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = str(self.config["model_save_path"])
        model.save(model_path)

        self.logger.info(f"Model saved to {model_path}")
        return model_path

    def _generate_summary(self) -> str:
        """サマリーを生成"""
        summary = f"""
Training Summary
================
Model: {self.config['model_name']}
Config: {self.config_path}
Timestamp: {datetime.now().isoformat()}

Key Parameters:
  • Total Timesteps: {self.config['training'].get('total_timesteps', 2000)}
  • Balance Penalty: {self.config['environment']['behavior_optimization']['balance_penalty']}
  • Buy Bonus: {self.config['environment']['action_bonuses']['buy_action_bonus']}
  • SELL Bonus: {self.config['environment']['action_bonuses']['sell_action_bonus']}
  • Hold Bonus: {self.config['environment']['action_bonuses']['hold_action_bonus']}

Expected Improvements:
  • Mean Reward: Significantly improved (target: > -5000)
  • BUY Ratio: Increased (target: 30-40%)
  • SELL Ratio: Decreased (target: 30-40%)
  • HOLD Ratio: Maintained/increased (target: 20-30%)
"""
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="SAC v444 Training with Config Support"
    )
    parser.add_argument(
        "--config",
        default="config/sac_v444_3_balanced_penalty_scale_200.json",
        help="Path to config file",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    try:
        trainer = ConfigurableTrainer(args.config, verbose=args.verbose)
        model_path, summary = trainer.train()

        print(summary)
        print("\n✅ Training completed!")
        print(f"Model saved to: {model_path}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
