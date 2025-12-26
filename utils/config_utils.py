#!/usr/bin/env python3
"""
Configuration Utilities

Common utilities for loading and merging configuration files.
Provides consistent config handling across the project.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_config_from_json(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file with consistent error handling.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logger.debug(f"Loaded configuration from {config_path}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def merge_training_configs(
    base_config: Dict[str, Any],
    env_config_path: Optional[Union[str, Path]] = None,
    reward_config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Merge training configuration files (environment and reward configs).

    Args:
        base_config: Base configuration dictionary
        env_config_path: Path to environment configuration file
        reward_config_path: Path to reward configuration file

    Returns:
        Merged configuration dictionary
    """
    config = base_config.copy()

    # Merge environment config
    if env_config_path:
        env_config_path = Path(env_config_path)
        if env_config_path.exists():
            env_config = load_config_from_json(env_config_path)
            if "environment" not in config:
                config["environment"] = {}
            config["environment"].update(env_config)
            logger.info(f"Merged environment config from {env_config_path}")
        else:
            logger.warning(f"Environment config file not found: {env_config_path}")

    # Merge reward config
    if reward_config_path:
        reward_config_path = Path(reward_config_path)
        if reward_config_path.exists():
            reward_config = load_config_from_json(reward_config_path)
            if "reward_function" not in config:
                config["reward_function"] = {}
            config["reward_function"].update(reward_config)
            logger.info(f"Merged reward config from {reward_config_path}")
        else:
            logger.warning(f"Reward config file not found: {reward_config_path}")

    return config