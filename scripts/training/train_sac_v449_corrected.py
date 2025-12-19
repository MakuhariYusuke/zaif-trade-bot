#!/usr/bin/env python3
"""
Train SAC v449 Corrected
Fixed feature engineering training script with VecNormalize
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
try:
    import torch
except ImportError:
    pass

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ztb.config.unified_config import UnifiedConfig
from ztb.features.models.sac.sac_v427_feature_engineering import SACv427FeatureEngineer
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.training.environments.heavy_trading_env import HeavyTradingEnv
from ztb.utils.logging_utils import setup_logging

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def train():
    config_path = "config/sac_v446_fixed.json"  # Reusing v446 config for now
    model_name = "sac_v449_corrected"
    data_path = "data/btc_jpy_5m_dataset.csv"
    total_timesteps = 50000  # Adjust as needed

    logger.info(f"🚀 Starting training for {model_name}")
    logger.info(f"Config: {config_path}")

    # Load Config
    try:
        unified_config = UnifiedConfig.from_file(config_path)
        config = unified_config.to_dict()
        logger.info("✅ Config loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        # Fallback to default if file missing
        config = {}

    # Load Data
    if Path(data_path).exists():
        data = pd.read_csv(data_path)
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data.set_index("timestamp", inplace=True)
        logger.info(f"✅ Data loaded: {len(data)} rows")
    else:
        logger.error(f"❌ Data file not found: {data_path}")
        return

    # Feature Engineering
    logger.info("Applying feature engineering...")
    try:
        feature_engineer = SACv427FeatureEngineer()
        featured_data = feature_engineer.generate_v427_features(data)

        # Align index
        if len(featured_data) != len(data):
            if isinstance(data.index, pd.DatetimeIndex):
                featured_data = featured_data.reindex(data.index)

        logger.info(f"✅ Features generated: {featured_data.shape}")
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        return

    # Prepare Environment
    try:
        env_config_dict = config.get("environment", {})
        env_config = EnvironmentConfig.from_dict(env_config_dict)

        feature_columns = [col for col in featured_data.columns if col != "timestamp"]

        # Padding if needed (v446 expects 182 features?)
        required_features = 182
        if len(feature_columns) < required_features:
            padding_needed = required_features - len(feature_columns)
            logger.warning(f"Padding {padding_needed} features")
            for pad_idx in range(padding_needed):
                pad_col = f"feature_padding_{pad_idx}"
                featured_data[pad_col] = 0.0
                feature_columns.append(pad_col)

        featured_data = featured_data.fillna(0).astype(np.float32)


        env = DummyVecEnv([make_env])

        # WRAP IN VECNORMALIZE - CRITICAL FIX
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        logger.info("✅ Environment initialized with VecNormalize")
    except Exception as e:
        logger.error(f"❌ Environment setup failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return

    # Initialize Model
    try:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            **config.get("training", {}).get("model_params", {}),
        )
        logger.info("✅ Model initialized")
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        return

    # Train
    logger.info(f"Starting training for {total_timesteps} steps...")
    try:
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, save_path="./models/checkpoints/", name_prefix=model_name
        )

        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        logger.info("✅ Training completed")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return

    # Save Model AND VecNormalize stats
    save_path = f"models/{model_name}"
    model.save(save_path)
    env.save(f"models/{model_name}_vecnormalize.pkl")
    logger.info(f"✅ Model saved to {save_path}.zip")
    logger.info(f"✅ Normalization stats saved to models/{model_name}_vecnormalize.pkl")


if __name__ == "__main__":
    train()
