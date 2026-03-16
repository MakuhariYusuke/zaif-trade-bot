#!/usr/bin/env python3
"""
v457.4 Verification Script: Multi-Seed Training
Runs short training (10k steps) with multiple seeds to verify stability.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import SAC

# Project Path Setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ztb.features.base_features_v456 import calculate_base_features
from ztb.trading.environment.utils.fast_intraday_env_v456_utils import (
    create_fast_intraday_env_v456,
)
from ztb.training.utils.v457_config_utils import (
    load_config_dict,
    extract_env_config,
    extract_sac_params
)
from ztb.utils.seed_manager import set_global_seed

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_seed(seed: int, args, df: pd.DataFrame):
    logger.info("="*70)
    logger.info(f"v457.4 Training: Seed {seed}")
    logger.info("="*70)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / args.config
    model_dir = project_root / args.model_dir / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    full_config = load_config_dict(config_path)
    env_config = extract_env_config(full_config)
    sac_params = extract_sac_params(full_config)
    
    # Force 1D Action
    env_config["action_space_type"] = "1d_position"
    sac_params["seed"] = seed
    set_global_seed(seed)
    
    # 3. Environment Creation
    env = create_fast_intraday_env_v456(
        df=df,
        env_config=env_config,
    )
    
    # Seed Env
    _, reset_info = env.reset(seed=seed)
    logger.info(f"Env reset: start_index={reset_info.get('start_index')}")

    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        **sac_params 
    )
    
    # Training
    model.learn(total_timesteps=10000, progress_bar=True)
    
    # Save
    timestamp = int(pd.Timestamp.now().timestamp())
    model_path = model_dir / f"sac_v457_4_seed{seed}_{timestamp}"
    model.save(str(model_path))
    logger.info(f"✓ Model saved: {model_path}")
    
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Verification")
    parser.add_argument('--config', type=str, default='config/v457_4/train_config.json')
    parser.add_argument('--csv-path', type=str, default='data/btc_jpy_1m_v451.csv')
    parser.add_argument('--model-dir', type=str, default='models/v457_4_verify')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 999])
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent.parent
    csv_path = project_root / args.csv_path

    logger.info(f"📥 Loading data shared for all seeds: {csv_path}")
    df = pd.read_csv(csv_path)
    df = calculate_base_features(df, copy=False)

    for seed in args.seeds:
        train_seed(seed, args, df)

if __name__ == "__main__":
    sys.exit(main())
