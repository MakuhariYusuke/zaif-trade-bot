#!/usr/bin/env python3
"""
Shared training entrypoint utilities for v457 training variants.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# Project Path Setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ztb.io.data_loader import DataLoader
from ztb.features.base_features_v456 import calculate_base_features
from ztb.trading.environment.utils.fast_intraday_env_v456_utils import (
    create_fast_intraday_env_v456,
)
from ztb.training.callbacks.advanced_callbacks import (
    BestModelSaveCallback,
    EarlyStoppingCallback,
)
from ztb.training.utils.v457_config_utils import (
    extract_env_config,
    extract_sac_params,
    extract_seed,
    load_config_dict,
)
from ztb.utils.seed_manager import set_global_seed

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def run_training(
    *,
    description: str,
    default_config: str,
    default_csv_path: str,
    default_model_dir: str,
    output_prefix: str,
    apply_fixed_ttl: bool = False,
    fixed_ttl: float = 1.0,
    log_action_space_type: bool = False,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--csv-path", type=str, default=default_csv_path)
    parser.add_argument("--model-dir", type=str, default=default_model_dir)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / args.config
    if not config_path.exists() and str(args.config).startswith("config/"):
        config_path = project_root / str(args.config).replace("config/", "configs/", 1)
    csv_path = project_root / args.csv_path
    model_dir = project_root / args.model_dir

    logger.info("=" * 70)
    logger.info(description)
    logger.info("=" * 70)

    full_config = load_config_dict(config_path)
    env_config = extract_env_config(full_config)
    sac_params = extract_sac_params(full_config)
    seed = extract_seed(full_config)
    training_meta = full_config.get("training", {})
    callback_config = full_config.get("callbacks", {})
    total_timesteps = training_meta.get("total_timesteps", 10000)

    if log_action_space_type:
        action_type = env_config.get("action_space_type", "2d_position_ttl")
        logger.info(f"Action Space Type: {action_type}")

    if seed is not None:
        set_global_seed(seed)
        sac_params["seed"] = seed
        logger.info(f"Seed fixed: {seed}")

    logger.info(f"Loading data from {csv_path}")
    df = DataLoader.load_csv_strict(csv_path)
    df = calculate_base_features(df, copy=False)

    logger.info("Creating training environment...")
    env = create_fast_intraday_env_v456(
        df=df,
        env_config=env_config,
    )
    if env is None:
        logger.error("Environment creation failed")
        return 1

    if apply_fixed_ttl:
        from ztb.trading.environment.wrappers.fixed_ttl_wrapper import FixedTTLWrapper

        logger.info(f"Applying FixedTTLWrapper (TTL={fixed_ttl})")
        env = FixedTTLWrapper(env, fixed_ttl=fixed_ttl)

    if seed is not None:
        _, reset_info = env.reset(seed=seed)
        logger.info(f"Env reset: start_index={reset_info.get('start_index')}")

    logger.info(
        f"Environment created: obs={env.observation_space.shape}, action={env.action_space.shape}"
    )

    logger.info(f"SAC Params: {json.dumps(sac_params, indent=2)}")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        **sac_params,
    )

    callbacks: List[BaseCallback] = []
    save_path = model_dir / "checkpoints"
    save_path.mkdir(parents=True, exist_ok=True)

    if "early_stopping" in callback_config:
        es_conf = callback_config["early_stopping"]
        callbacks.append(
            EarlyStoppingCallback(
                metric_name=es_conf.get("metric_name", "rollout/ep_rew_mean"),
                min_delta=es_conf.get("min_delta", 0.001),
                patience=es_conf.get("patience", 10000),
                check_interval=es_conf.get("check_interval", 1000),
                window_size=es_conf.get("window_size", 1000),
                cv_threshold=es_conf.get("cv_threshold", 0.05),
            )
        )

    if "best_model" in callback_config:
        bm_conf = callback_config["best_model"]
        check_interval = bm_conf.get("check_interval", bm_conf.get("check_freq", 5000))
        callbacks.append(
            BestModelSaveCallback(
                save_path=save_path,
                model_name=bm_conf.get("model_name", "best_model"),
                metric_name=bm_conf.get("metric_name", "rollout/ep_rew_mean"),
                mode=bm_conf.get("mode", "max"),
                check_interval=check_interval,
            )
        )

    logger.info(f"Starting training for {total_timesteps} steps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks) if callbacks else None,
            progress_bar=True,
        )
        logger.info("Training completed")
    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        return 1
    finally:
        try:
            env.close()
        except Exception:
            pass

    timestamp = int(pd.Timestamp.now().timestamp())
    output_dir = model_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{output_prefix}_{timestamp}"
    model.save(str(model_path))
    logger.info(f"Final Model saved: {model_path}")
    return 0
