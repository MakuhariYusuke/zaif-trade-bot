from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ztb.utils.config_loader import load_yaml_config


def load_config_dict(config_path: Path) -> dict[str, Any]:
    """Load a YAML config as a dict, returning {} for invalid or empty content."""
    config = load_yaml_config(config_path)
    if not isinstance(config, dict):
        return {}
    return config


def extract_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract training section as a dict."""
    training = config.get("training")
    if isinstance(training, Mapping):
        return dict(training)
    return {}


def extract_env_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract training.environment section as a dict."""
    training = extract_training_config(config)
    env_config = training.get("environment")
    if isinstance(env_config, Mapping):
        return dict(env_config)
    return {}


def extract_sac_params(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract training.sac_hyperparameters section as a dict."""
    training = extract_training_config(config)
    sac_params = training.get("sac_hyperparameters")
    if isinstance(sac_params, Mapping):
        return dict(sac_params)
    return {}


def extract_seed(config: Mapping[str, Any]) -> Optional[int]:
    """Extract training seed from config (training.seed or training.sac_hyperparameters.seed)."""
    training = extract_training_config(config)
    seed = None
    if isinstance(training, Mapping):
        seed = training.get("seed")
        if seed is None:
            sac_params = training.get("sac_hyperparameters")
            if isinstance(sac_params, Mapping):
                seed = sac_params.get("seed")
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None
