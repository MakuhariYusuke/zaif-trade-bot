#!/usr/bin/env python3
"""
Optimized Fine-tuning script for v452.
Uses optimized thresholds from auto-tuning and reduced logging.
"""

import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.utils.logging_utils import get_logger, setup_logging
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig

# Configuration
BASE_MODEL_PATH = "models/sac_v451_phase7_regime_aware.zip"
DATA_PATH = "data/btc_jpy_1m_v451.csv"
THRESHOLDS_PATH = "config/threshold_optimized.json"
FINE_TUNE_LR = 1e-5
TOTAL_TIMESTEPS = 10000  # Increased from 5000 for better learning
OUTPUT_MODEL_NAME = "sac_v452_optimized_10k"

def run_fine_tuning():
    # Setup logging with reduced level
    setup_logging(level=logging.WARNING)
    logger = get_logger("fine_tuning_optimized")
    
    # Manually print important info since log level is high
    print("Starting optimized fine-tuning v452")
    print(f"Base model: {BASE_MODEL_PATH}")
    print(f"Data: {DATA_PATH}")
    print(f"Thresholds: {THRESHOLDS_PATH}")
    
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"Error: Base model not found: {BASE_MODEL_PATH}")
        return
        
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found: {DATA_PATH}")
        return

    if not os.path.exists(THRESHOLDS_PATH):
        print(f"Error: Thresholds file not found: {THRESHOLDS_PATH}")
        return

    # Load optimized thresholds
    with open(THRESHOLDS_PATH, 'r') as f:
        optimized_thresholds = json.load(f)
    print(f"Loaded thresholds: {optimized_thresholds}")

    # Load Data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"], index_col="timestamp")
    
    # Create Environment Config
    env_config = EnvironmentConfig(
        feature_set="default",
        use_continuous_actions=True,
        target_feature_count=138, # Fix for feature mismatch
        correlation_reduction=True,
        # Add other necessary config params
        max_steps=1000,
    )
    
    # Inject optimized thresholds (Monkey-patching)
    # ThresholdManager looks for 'regime_threshold_multipliers' in config
    env_config.regime_threshold_multipliers = optimized_thresholds
    
    # Create Environment
    print("Creating environment...")
    env = HeavyTradingEnv(df=df, config=env_config)
    
    # Trainer Config
    trainer_config = {
        "training": {
            "algorithm": "sac",
            "total_timesteps": TOTAL_TIMESTEPS,
            "data_config": {
                "data_path": DATA_PATH
            },
            "sac_hyperparameters": {
                "learning_rate": FINE_TUNE_LR,
                "buffer_size": 1000000,
                "learning_starts": 0,
                "batch_size": 256,
                "ent_coef": "auto",
                "train_freq": 1,
                "gradient_steps": 1,
            },
            "checkpoint_dir": "models/checkpoints_finetune_opt",
        },
        "environment": {
             "use_continuous_actions": True,
             "action_space_type": "continuous",
        }
    }
    
    # Initialize Trainer with custom env
    trainer = SACTrainer(trainer_config, env=env, logger=logger)
    
    # Load Pre-trained Model
    print("Loading pre-trained model...")
    try:
        loaded_model = SAC.load(BASE_MODEL_PATH)
        
        # Update Learning Rate
        loaded_model.learning_rate = FINE_TUNE_LR
        
        def constant_lr_schedule(progress_remaining):
            return FINE_TUNE_LR
            
        loaded_model.lr_schedule = constant_lr_schedule
        
        if loaded_model.policy and hasattr(loaded_model.policy, "optimizer") and loaded_model.policy.optimizer:
            for param_group in loaded_model.policy.optimizer.param_groups:
                param_group["lr"] = FINE_TUNE_LR
        
        trainer.model = loaded_model
        print("Pre-trained model loaded and injected into trainer.")
        
    except Exception as e:
        print(f"Failed to load pre-trained model: {e}")
        return

    # Run Training
    print("Starting training...")
    success = trainer.train()
    
    if success:
        print("Fine-tuning completed successfully.")
        # Save the fine-tuned model
        save_path = os.path.join("models", f"{OUTPUT_MODEL_NAME}.zip")
        trainer.model.save(save_path)
        print(f"Saved fine-tuned model to: {save_path}")
    else:
        print("Fine-tuning failed.")

if __name__ == "__main__":
    run_fine_tuning()
