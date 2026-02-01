from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import pandas as pd

from ztb.trading.environment.factory_v456 import EnvironmentFactory
from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.utils.env_metrics import unwrap_env

logger = logging.getLogger(__name__)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def create_fast_intraday_env_v456(
    df: pd.DataFrame, env_config: Mapping[str, Any]
) -> Optional[FastIntradayEnvV456]:
    """Create a FastIntradayEnvV456 with config-driven parameters and reward wiring."""
    initial_balance = _coerce_float(env_config.get("initial_balance", 1_000_000.0), 1_000_000.0)
    max_position = _coerce_float(env_config.get("max_position_size", 1.0), 1.0)
    commission = _coerce_float(env_config.get("transaction_cost", 0.001), 0.001)

    factory = EnvironmentFactory(
        df=df,
        initial_balance=initial_balance,
        max_position=max_position,
        commission_rate=commission,
        config=env_config,
    )
    
    # Build env_kwargs from config
    known_utils_keys = [
        "max_ttl_steps", 
        "cooldown_steps", 
        "max_delta_per_step", 
        "min_delta",
        "drawdown_limit",
        "prewarm_steps",
        "action_space_type",
        "guidance_decay_steps",
        "max_steps"  # Walk-Forward評価用
    ]
    env_kwargs = {k: env_config[k] for k in known_utils_keys if k in env_config}
    
    env = factory.create_training_env(env_kwargs=env_kwargs)
    if env is None:
        logger.error("Failed to create FastIntradayEnvV456.")
        return None

    apply_reward_config(env, env_config)
    return env


def apply_reward_config(env: Any, env_config: Mapping[str, Any]) -> None:
    """Apply reward settings/scale/clip to FastIntradayEnvV456-compatible environments."""
    target_env = unwrap_env(env) or env

    reward_settings = env_config.get("reward_settings")
    if isinstance(reward_settings, Mapping):
        reward_settings = dict(reward_settings)

    if reward_settings:
        if hasattr(target_env, "reward_params"):
            target_env.reward_params = reward_settings
            logger.info("Injected reward_settings into env reward_params.")
        else:
            logger.warning("Env has no reward_params attribute; reward_settings ignored.")

    reward_scale = env_config.get("reward_scale")
    reward_clip = env_config.get("reward_clip") if "reward_clip" in env_config else None

    updated = False
    if reward_scale is not None and hasattr(target_env, "reward_scale"):
        target_env.reward_scale = _coerce_float(reward_scale, target_env.reward_scale)
        updated = True
    if "reward_clip" in env_config and hasattr(target_env, "reward_clip"):
        target_env.reward_clip = reward_clip
        updated = True

    if updated:
        logger.info("Injected reward scaling (reward_scale/reward_clip) into env.")
