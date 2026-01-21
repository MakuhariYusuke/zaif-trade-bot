#!/usr/bin/env python3
"""
v457.2 Profit-First Retraining Script
Based on simple v456 trainer + Integrated Config & Callbacks
(Moved to scripts/v457)
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# Project Path Setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ztb.features.base_features_v456 import calculate_base_features
from ztb.trading.environment.utils.fast_intraday_env_v456_utils import (
    create_fast_intraday_env_v456,
)
from ztb.training.utils.v457_config_utils import (
    load_config_dict,
    extract_env_config,
    extract_sac_params,
    extract_seed,
)
from ztb.training.callbacks.advanced_callbacks import (
    EarlyStoppingCallback,
    BestModelSaveCallback
)
from ztb.utils.seed_manager import set_global_seed

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="v457.2 Training: Profit-First Strategy")
    parser.add_argument('--config', type=str, required=True, help='Path to training config (json/yaml)')
    parser.add_argument('--csv-path', type=str, default='data/btc_jpy_1m_v451.csv', help='Training data path')
    parser.add_argument('--model-dir', type=str, default='models/v457_2', help='Directory to save models')
    args = parser.parse_args()
    
    # Path resolution
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / args.config
    csv_path = project_root / args.csv_path
    
    logger.info("="*70)
    logger.info("v457.2 Profit-First Training Pipeline")
    logger.info("="*70)
    
    # 1. Load Configuration
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
        
    logger.info(f"Loading config from {config_path}")
    full_config = load_config_dict(config_path)
    
    # Extract sections
    env_config = extract_env_config(full_config)
    sac_params = extract_sac_params(full_config)
    seed = extract_seed(full_config)
    training_meta = full_config.get("training", {})
    callback_config = full_config.get("callbacks", {})
    
    total_timesteps = training_meta.get("total_timesteps", 100000)
    
    logger.info(f"Total Timesteps: {total_timesteps:,}")

    if seed is not None:
        set_global_seed(seed)
        sac_params["seed"] = seed
        logger.info(f"Seed fixed: {seed}")

    # 2. Data Loading
    logger.info(f"📥 Loading data from {csv_path}")
    if not csv_path.exists():
        logger.error(f"Data file not found: {csv_path}")
        return 1
        
    df = pd.read_csv(csv_path)
    logger.info(f"✓ Loaded {len(df):,} bars")
    
    # Feature Calculation
    logger.info("Calculating features...")
    df = calculate_base_features(df, copy=False)
    
    # 3. Environment Creation
    logger.info("Creating training environment...")
    try:
        # Pass loaded env_config directly
        env = create_fast_intraday_env_v456(
            df=df,
            env_config=env_config,
        )
        if env is None:
            raise RuntimeError("Failed to create environment (returned None)")
        del df
        
        logger.info(f"✓ Environment created: obs_shape={env.observation_space.shape}")

        if seed is not None:
            _, reset_info = env.reset(seed=seed)
            logger.info(f"Env reset: start_index={reset_info.get('start_index')}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create environment: {e}", exc_info=True)
        return 1
    
    # 4. Model Creation
    logger.info("Creating SAC model...")
    logger.info(f"SAC Params: {json.dumps(sac_params, indent=2)}")
    
    try:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            **sac_params 
        )
        logger.info("✓ SAC model created")
    except Exception as e:
        logger.error(f"❌ Failed to create model: {e}", exc_info=True)
        return 1
        
    # 5. Callbacks Setup
    callbacks: List[BaseCallback] = []
    
    # Early Stopping
    if "early_stopping" in callback_config:
        es_conf = callback_config["early_stopping"]
        logger.info(f"Adding Early Stopping: {es_conf}")
        callbacks.append(EarlyStoppingCallback(
            metric_name="rollout/ep_rew_mean", 
            min_delta=es_conf.get("min_delta", 0.001),
            patience=es_conf.get("patience", 10000),
            check_interval=es_conf.get("check_interval", 1000)
        ))
        
    # Best Model Saving
    if "best_model" in callback_config:
        bm_conf = callback_config["best_model"]
        save_path = project_root / args.model_dir / "checkpoints"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Adding Best Model Saver: {bm_conf} -> {save_path}")
        callbacks.append(BestModelSaveCallback(
            save_path=save_path,
            metric_name=bm_conf.get("metric_name", "rollout/ep_rew_mean"),
            mode=bm_conf.get("mode", "max"),
            check_interval=bm_conf.get("check_interval", 1000)
        ))

    # 6. Training Execution
    logger.info(f"\n🚀 Starting training for {total_timesteps:,} steps")
    start_time = datetime.utcnow()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks) if callbacks else None,
            progress_bar=True,
        )
        logger.info("✅ Training completed successfully")
    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return 1
    finally:
        try:
             env.close()
        except:
             pass
    
    # 7. Final Model Save
    timestamp = int(pd.Timestamp.now().timestamp())
    output_dir = project_root / args.model_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"sac_v457_2_final_{timestamp}"
    
    try:
        model.save(str(model_path))
        logger.info(f"✓ Final Model saved: {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save final model: {e}", exc_info=True)
        return 1
    
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
