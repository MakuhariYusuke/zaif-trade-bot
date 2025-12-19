#!/usr/bin/env python3
"""
Fine-tuning script for v452 (Transfer Learning).
Loads a pre-trained model and trains it on recent data with a lower learning rate.
"""

import os
import sys
from pathlib import Path
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.utils.logging_utils import get_logger, setup_logging

# Configuration
BASE_MODEL_PATH = "models/sac_v451_phase7_regime_aware.zip"
DATA_PATH = "data/btc_jpy_1m_v451.csv"
FINE_TUNE_LR = 1e-5
TOTAL_TIMESTEPS = 5000  # Short duration for verification (5000 steps)
OUTPUT_MODEL_NAME = "sac_v452_fine_tuned_5k"

def run_fine_tuning():
    setup_logging()
    logger = get_logger("fine_tuning_v452")
    
    logger.info("Starting fine-tuning v452")
    logger.info(f"Base model: {BASE_MODEL_PATH}")
    logger.info(f"Data: {DATA_PATH}")
    logger.info(f"Learning Rate: {FINE_TUNE_LR}")
    
    if not os.path.exists(BASE_MODEL_PATH):
        logger.error(f"Base model not found: {BASE_MODEL_PATH}")
        return
        
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found: {DATA_PATH}")
        return

    # Load base config (you might want to load a specific config file if needed)
    # For now, we'll construct a minimal config for SACTrainer
    config = {
        "training": {
            "algorithm": "sac",
            "total_timesteps": TOTAL_TIMESTEPS,
            "data_config": {
                "data_path": DATA_PATH
            },
            "sac_hyperparameters": {
                "learning_rate": FINE_TUNE_LR,
                "buffer_size": 1000000,
                "learning_starts": 0, # Start learning immediately since we have a pre-trained model
                "batch_size": 256,
                "ent_coef": "auto",
                "train_freq": 1,
                "gradient_steps": 1,
            },
            "environment": {
                "config": {
                    "max_episode_steps": 1000, # Adjust as needed
                    "feature_set": "default", # Or whatever v451 used
                    # Add other env config params if needed
                }
            },
            "checkpoint_dir": "models/checkpoints_finetune",
        },
        "environment": { # Fallback location
             "use_continuous_actions": True,
             "action_space_type": "continuous",
        }
    }
    
    # Initialize Trainer
    trainer = SACTrainer(config, logger=logger)
    
    # Load Pre-trained Model
    logger.info("Loading pre-trained model...")
    try:
        # We need to load the model using SAC.load, but we need the environment first.
        # SACTrainer creates the environment in _execute_sac_training.
        # However, we can't easily inject the model *after* env creation but *before* training starts
        # if we use trainer.train().
        
        # Workaround: We will manually create the environment and model, then assign to trainer.
        # But SACTrainer.train() calls execute_training_pipeline which calls _execute_sac_training.
        # _execute_sac_training creates the env.
        
        # Let's use a trick: SACTrainer checks 'if self.model is None'.
        # If we set self.model, it uses it.
        # But self.model needs an environment set.
        # And the environment is created inside _execute_sac_training.
        
        # So, we can't set self.model fully before calling train() because we don't have the env yet.
        # AND if we create env outside, we duplicate logic.
        
        # However, SAC.load() can load without an env, and we can set env later.
        # But SACTrainer._execute_sac_training does:
        # if self.model is None: ... else: self.model.set_env(wrapped_env)
        
        # So we can load the model without env, assign to self.model, and SACTrainer will set the env.
        
        loaded_model = SAC.load(BASE_MODEL_PATH)
        
        # Update Learning Rate
        # SAC stores learning rate in self.learning_rate (which can be a float or a schedule)
        # We want to force it to our new constant LR.
        loaded_model.learning_rate = FINE_TUNE_LR
        
        # Also update the optimizer's learning rate if it's already initialized
        # (which it is, since we loaded it)
        # Note: SB3 might re-initialize optimizer on learn() if we are not careful, 
        # but usually it keeps it if we continue learning.
        # However, we want to CHANGE the LR.
        
        # We need to update the learning rate scheduler.
        # In SB3, learning_rate is a function (schedule).
        def constant_lr_schedule(progress_remaining):
            return FINE_TUNE_LR
            
        loaded_model.lr_schedule = constant_lr_schedule
        
        # If optimizer is already created, update its param groups
        if loaded_model.policy and hasattr(loaded_model.policy, "optimizer") and loaded_model.policy.optimizer:
            for param_group in loaded_model.policy.optimizer.param_groups:
                param_group["lr"] = FINE_TUNE_LR
        else:
            logger.info("Optimizer not found in loaded model. A new one will be created with the new learning rate.")
                
        trainer.model = loaded_model
        logger.info("Pre-trained model loaded and injected into trainer.")
        
    except Exception as e:
        logger.error(f"Failed to load pre-trained model: {e}")
        return

    # Run Training
    logger.info("Starting training...")
    success = trainer.train()
    
    if success:
        logger.info("Fine-tuning completed successfully.")
        # Save the fine-tuned model
        save_path = os.path.join("models", f"{OUTPUT_MODEL_NAME}.zip")
        trainer.model.save(save_path)
        logger.info(f"Saved fine-tuned model to: {save_path}")
    else:
        logger.error("Fine-tuning failed.")

if __name__ == "__main__":
    run_fine_tuning()
