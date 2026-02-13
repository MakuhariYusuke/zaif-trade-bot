#!/usr/bin/env python3
"""
Backtest Initialization Utilities

Unified utilities for initializing backtest components.
Provides consistent setup for config, model, data, and environment.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.config_utils import load_config_unified
from ztb.utils.data_utils import load_csv_data
from ztb.utils.training_utils import load_model


def initialize_backtest_components(
    config_path: Union[str, Path],
    model_path: Union[str, Path],
    data_path: Union[str, Path],
    algorithm: str = "SAC",
    required_config_keys: Optional[list] = None,
    env_config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Initialize all backtest components with unified utilities.

    Args:
        config_path: Path to configuration file
        model_path: Path to model file
        data_path: Path to market data CSV file
        algorithm: Algorithm name (SAC, PPO, etc.)
        required_config_keys: Required keys in config
        env_config_overrides: Environment config overrides

    Returns:
        Dictionary containing initialized components:
        - config: Loaded configuration
        - model: Loaded model
        - data: Loaded market data
        - env: Created environment

    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If configuration is invalid
    """
    components = {}

    # Load configuration
    config = load_config_unified(
        config_path, required_keys=required_config_keys or ["training", "environment"]
    )
    components["config"] = config

    # Load market data
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = load_csv_data(data_path, index_col=0, parse_dates=True)
    components["data"] = data

    # Setup environment config
    env_config = config["training"]["environment"]["config"].copy()
    if env_config_overrides:
        env_config.update(env_config_overrides)

    # Create environment
    env = HeavyTradingEnv(data, env_config)
    components["env"] = env

    # Load model
    model = load_model(model_path, algorithm=algorithm)

    # Set environment for the model
    model.set_env(env)
    components["model"] = model

    return components


def setup_backtest_environment(
    config: Dict[str, Any],
    data: pd.DataFrame,
    env_config_overrides: Optional[Dict[str, Any]] = None,
) -> HeavyTradingEnv:
    """
    Setup backtest environment from config and data.

    Args:
        config: Configuration dictionary
        data: Market data DataFrame
        env_config_overrides: Environment config overrides

    Returns:
        Configured HeavyTradingEnv instance
    """
    # Extract environment config
    env_config = config["training"]["environment"]["config"].copy()
    if env_config_overrides:
        env_config.update(env_config_overrides)

    # Create and return environment
    return HeavyTradingEnv(data, env_config)


def validate_backtest_setup(
    config: Dict[str, Any], model, data: pd.DataFrame, env
) -> list:
    """
    Validate backtest setup components.

    Args:
        config: Configuration dictionary
        model: Loaded model
        data: Market data DataFrame
        env: Trading environment

    Returns:
        List of validation warnings/errors
    """
    warnings = []

    # Check data
    if data.empty:
        warnings.append("Market data is empty")

    if len(data) < 1000:
        warnings.append(f"Market data is very small: {len(data)} rows")

    # Check required data columns
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        warnings.append(f"Missing required data columns: {missing_columns}")

    # Check config
    if "training" not in config:
        warnings.append("Config missing 'training' section")

    if "environment" not in config:
        warnings.append("Config missing 'environment' section")

    # Check model
    if model is None:
        warnings.append("Model is None")
    else:
        # Check if model has required attributes
        if not hasattr(model, "predict"):
            warnings.append("Model missing 'predict' method")

    # Check environment
    if env is None:
        warnings.append("Environment is None")
    else:
        if not hasattr(env, "reset"):
            warnings.append("Environment missing 'reset' method")

        if not hasattr(env, "step"):
            warnings.append("Environment missing 'step' method")

    return warnings
