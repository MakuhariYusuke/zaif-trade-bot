"""
SAC v437 Enhanced Configuration

Configuration for SAC v437 with enhanced features and trading frequency control.
"""

import json
from pathlib import Path


def get_v437_config():
    """
    Get SAC v437 configuration.

    Returns:
        Dictionary containing v437 configuration
    """
    config_path = Path(__file__).parent / "sac_v437_enhanced_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def get_v437_feature_config():
    """
    Get v437 feature configuration.

    Returns:
        Dictionary containing v437 feature configuration
    """
    config_path = (
        Path(__file__).parent.parent
        / "features"
        / "feature_sets"
        / "v437_enhanced_features.json"
    )

    if not config_path.exists():
        raise FileNotFoundError(f"Feature configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def update_v437_config(updates: dict):
    """
    Update v437 configuration with new values.

    Args:
        updates: Dictionary of configuration updates
    """
    config = get_v437_config()
    config.update(updates)

    config_path = Path(__file__).parent / "sac_v437_enhanced_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def create_v437_experiment_config(experiment_name: str, **kwargs):
    """
    Create experiment-specific configuration.

    Args:
        experiment_name: Name of the experiment
        **kwargs: Additional configuration parameters

    Returns:
        Dictionary containing experiment configuration
    """
    base_config = get_v437_config()

    # Update with experiment-specific settings
    experiment_config = base_config.copy()
    experiment_config["model_name"] = f"sac_v437_{experiment_name}"
    experiment_config["experiment_name"] = experiment_name

    # Apply custom parameters
    for key, value in kwargs.items():
        if key in experiment_config:
            experiment_config[key] = value
        elif key in experiment_config.get("environment", {}):
            experiment_config["environment"][key] = value
        elif key in experiment_config.get("sac_hyperparameters", {}):
            experiment_config["sac_hyperparameters"][key] = value

    return experiment_config


# Default v437 configuration for easy import
V437_CONFIG = get_v437_config()
V437_FEATURE_CONFIG = get_v437_feature_config()
