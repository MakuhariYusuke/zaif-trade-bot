#!/usr/bin/env python3
"""
v457.4 Training Script: Native 1D Action Support
"""

import argparse
import logging
import sys
import json
from pathlib import Path
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
    parser = argparse.ArgumentParser(description="v457.4 Training: Native 1D")
    parser.add_argument('--config', type=str, default='config/v457_4/train_config.json')
    parser.add_argument('--csv-path', type=str, default='data/btc_jpy_1m_v451.csv')
    parser.add_argument('--model-dir', type=str, default='models/v457_4')
    args = parser.parse_args()
    
    # Resolve Paths
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / args.config
    csv_path = project_root / args.csv_path
    model_dir = project_root / args.model_dir
    
    logger.info("="*70)
    logger.info("v457.4 Training: Native 1D Support")
    logger.info("="*70)
    
    # 1. Load Configuration
    full_config = load_config_dict(config_path)
    env_config = extract_env_config(full_config)
    sac_params = extract_sac_params(full_config)
    seed = extract_seed(full_config)
    training_meta = full_config.get("training", {})
    callback_config = full_config.get("callbacks", {})
    
    total_timesteps = training_meta.get("total_timesteps", 10000)

    if seed is not None:
        set_global_seed(seed)
        sac_params["seed"] = seed
        logger.info(f"Seed fixed: {seed}")
    
    # Check for Native 1D config
    action_type = env_config.get("action_space_type", "2d_position_ttl")
    logger.info(f"Action Space Type: {action_type}")
    
    # 2. Data Loading
    logger.info(f"📥 Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    df = calculate_base_features(df, copy=False)
    
    # 3. Environment Creation
    logger.info("Creating training environment...")
    
    env = create_fast_intraday_env_v456(
        df=df,
        env_config=env_config,
    )
    
    if env is None:
        logger.error("Environment creation failed")
        return 1

    if seed is not None:
        _, reset_info = env.reset(seed=seed)
        logger.info(f"Env reset: start_index={reset_info.get('start_index')}")

    logger.info(f"✓ Environment created: obs={env.observation_space.shape}, action={env.action_space.shape}")
    
    # 4. Model Creation
    logger.info("Creating SAC model...")
    logger.info(f"SAC Params: {json.dumps(sac_params, indent=2)}")
    
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        **sac_params 
    )
    
    # 5. Callbacks
    callbacks: List[BaseCallback] = []
    
    save_path = model_dir / "checkpoints"
    save_path.mkdir(parents=True, exist_ok=True)
    
    if "best_model" in callback_config:
        bm_conf = callback_config["best_model"]
        callbacks.append(BestModelSaveCallback(
            save_path=save_path,
            metric_name=bm_conf.get("metric_name", "rollout/ep_rew_mean"),
            mode=bm_conf.get("mode", "max"),
            check_interval=bm_conf.get("check_interval", 1000)
        ))

    # 6. Training Execution
    logger.info(f"\n🚀 Starting training...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )
        logger.info("✅ Training completed")
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
    output_dir = model_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"sac_v457_4_final_{timestamp}"
    
    model.save(str(model_path))
    logger.info(f"✓ Final Model saved: {model_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
