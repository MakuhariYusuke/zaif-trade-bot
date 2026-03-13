"""Type guards and advanced type validation utilities."""

from __future__ import annotations

from typing import Callable, get_args, get_origin

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import TypeGuard

from ztb.types.common import ConfigSection, ObjectMap

def is_valid_config(config: object) -> TypeGuard[ConfigSection]:
    """Type guard for valid configuration dictionaries."""
    return isinstance(config, dict) and len(config) > 0

def is_valid_training_config(config: object) -> TypeGuard[ConfigSection]:
    """Type guard for valid training configurations."""
    if not is_valid_config(config):
        return False
    required_keys = ["learning_rate", "batch_size", "total_timesteps"]
    return all(key in config for key in required_keys)

def is_valid_trading_config(config: object) -> TypeGuard[ConfigSection]:
    """Type guard for valid trading configurations."""
    if not is_valid_config(config):
        return False
    required_keys = ["initial_balance", "max_position_size"]
    return all(key in config for key in required_keys)

def is_valid_model_config(config: object) -> TypeGuard[ConfigSection]:
    """Type guard for valid model configurations."""
    if not is_valid_config(config):
        return False
    required_keys = ["learning_rate", "batch_size"]
    return all(key in config for key in required_keys)

def is_valid_reward(reward: object) -> TypeGuard[float]:
    """Type guard for valid reward values."""
    return isinstance(reward, (int, float, np.number)) and np.isfinite(float(reward))

def is_valid_action(action: object) -> TypeGuard[int]:
    """Type guard for valid discrete actions."""
    return isinstance(action, (int, np.integer)) and int(action) in [0, 1, 2]

def is_valid_continuous_action(action: object) -> TypeGuard[float]:
    """Type guard for valid continuous actions in [-1, 1]."""
    return isinstance(action, (int, float, np.number)) and -1.0 <= float(action) <= 1.0

def is_valid_observation(observation: object) -> TypeGuard[NDArray[np.float32]]:
    """Type guard for valid observation arrays."""
    if not isinstance(observation, np.ndarray):
        return False
    return bool(
        observation.dtype == np.float32
        and observation.ndim == 1
        and np.all(np.isfinite(observation))
    )

def is_valid_probability_distribution(probs: object) -> TypeGuard[NDArray[np.float32]]:
    """Type guard for valid probability distributions."""
    if not isinstance(probs, np.ndarray):
        return False
    return bool(
        probs.dtype == np.float32
        and probs.ndim == 1
        and np.all(np.isfinite(probs))
        and np.all(probs >= 0)
        and np.all(probs <= 1)
        and np.isclose(np.sum(probs), 1.0, atol=1e-6)
    )

def is_valid_portfolio_value(value: object) -> TypeGuard[float]:
    """Type guard for valid positive portfolio values."""
    return isinstance(value, (int, float, np.number)) and float(value) > 0

def is_valid_position(position: object) -> TypeGuard[float]:
    """Type guard for valid position values (-2 to 2)."""
    return isinstance(position, (int, float, np.number)) and -2.0 <= float(position) <= 2.0

def is_valid_loss(loss: object) -> TypeGuard[float]:
    """Type guard for valid non-negative finite loss values."""
    return (
        isinstance(loss, (int, float, np.number))
        and float(loss) >= 0
        and np.isfinite(float(loss))
    )

def validate_type_with_guard(
    value: object, type_guard_func: Callable[[object], bool], type_name: str
) -> None:
    """Validate a value using a type guard function."""
    if not type_guard_func(value):
        raise TypeError(f"Expected {type_name}, got {type(value).__name__}: {value}")

def is_valid_config_dict(config: object) -> TypeGuard[ConfigSection]:
    """Type guard for valid configuration dictionaries."""
    return isinstance(config, dict) and all(isinstance(k, str) for k in config)

def is_valid_feature_array(features: object) -> TypeGuard[NDArray[np.float32]]:
    """Type guard for valid feature arrays."""
    if not isinstance(features, np.ndarray):
        return False
    return bool(
        features.dtype in [np.float32, np.float64]
        and features.ndim == 2
        and np.all(np.isfinite(features))
    )

def is_valid_reward_array(rewards: object) -> TypeGuard[NDArray[np.float32]]:
    """Type guard for valid reward arrays."""
    if not isinstance(rewards, np.ndarray):
        return False
    return bool(
        rewards.dtype == np.float32
        and rewards.ndim == 1
        and np.all(np.isfinite(rewards))
    )

def is_valid_action_array(actions: object) -> TypeGuard[NDArray[np.int64]]:
    """Type guard for valid action arrays."""
    if not isinstance(actions, np.ndarray):
        return False
    return bool(
        actions.dtype == np.int64
        and actions.ndim == 1
        and np.all(np.isin(actions, [0, 1, 2]))
    )

def validate_generic_type(value: object, expected_type: object, name: str) -> None:
    """Validate a value against a (possibly generic) type."""
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is None:
        if isinstance(expected_type, type) and not isinstance(value, expected_type):
            raise TypeError(
                f"{name} must be {expected_type.__name__}, got {type(value).__name__}"
            )
        return

    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"{name} must be list, got {type(value).__name__}")
        if args and isinstance(args[0], type):
            if not all(isinstance(item, args[0]) for item in value):
                raise TypeError(f"{name} elements must be {args[0].__name__}")
        return

    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be dict, got {type(value).__name__}")
        if args and len(args) >= 2:
            key_type, value_type = args[0], args[1]
            if isinstance(key_type, type) and not all(
                isinstance(k, key_type) for k in value.keys()
            ):
                raise TypeError(f"{name} keys must be {key_type.__name__}")
            if isinstance(value_type, type) and not all(
                isinstance(v, value_type) for v in value.values()
            ):
                raise TypeError(f"{name} values must be {value_type.__name__}")

def is_valid_dataframe(df: object) -> TypeGuard[pd.DataFrame]:
    """Type guard for valid non-empty pandas DataFrame."""
    return bool(isinstance(df, pd.DataFrame) and not df.empty)

def is_valid_numpy_array(arr: object) -> TypeGuard[NDArray[np.generic]]:
    """Type guard for valid numpy arrays."""
    return bool(isinstance(arr, np.ndarray) and arr.size > 0)

def is_valid_features_dict(features: object) -> TypeGuard[ObjectMap]:
    """Type guard for valid feature dictionaries."""
    if not isinstance(features, dict):
        return False
    for key, value in features.items():
        if not isinstance(key, str):
            return False
        if not (isinstance(value, (list, tuple)) or is_valid_numpy_array(value)):
            return False
    return len(features) > 0

def is_valid_price_data(data: object) -> TypeGuard[NDArray[np.float64]]:
    """Type guard for valid price arrays (positive finite values)."""
    if not is_valid_numpy_array(data):
        return False
    return bool(np.all(np.isfinite(data)) and np.all(data > 0))

def is_valid_info_dict(info: object) -> TypeGuard[dict[str, float | int]]:
    """Type guard for trading environment info dictionaries."""
    if not isinstance(info, dict):
        return False
    for key, value in info.items():
        if not isinstance(key, str):
            return False
        if not isinstance(value, (int, float)):
            return False
    return True

def is_valid_stats_result(
    stats: object,
) -> TypeGuard[dict[str, float | list[float]]]:
    """Type guard for statistics result dictionaries."""
    if not isinstance(stats, dict):
        return False

    required_keys = {"mean", "std", "ci95"}
    if not all(key in stats for key in required_keys):
        return False

    if not isinstance(stats["mean"], (int, float)):
        return False
    if not isinstance(stats["std"], (int, float)):
        return False
    if not isinstance(stats["ci95"], list) or len(stats["ci95"]) != 2:
        return False
    if not all(isinstance(x, (int, float)) for x in stats["ci95"]):
        return False

    return True
