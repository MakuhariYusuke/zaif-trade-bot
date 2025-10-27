"""Type guards and advanced type validation utilities."""

from typing import Any, Dict, List, Optional, Union, get_origin, get_args, Callable
import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeGuard


def is_valid_config(config: Any) -> TypeGuard[Dict[str, Any]]:
    """Type guard for valid configuration dictionaries.

    Args:
        config: Value to check

    Returns:
        True if value is a non-empty dictionary
    """
    return isinstance(config, dict) and len(config) > 0


def is_valid_training_config(config: Any) -> TypeGuard[Dict[str, Any]]:
    """Type guard for valid training configurations.

    Args:
        config: Value to check

    Returns:
        True if value is a valid training config dict
    """
    if not is_valid_config(config):
        return False

    required_keys = ['learning_rate', 'batch_size', 'total_timesteps']
    return all(key in config for key in required_keys)


def is_valid_trading_config(config: Any) -> TypeGuard[Dict[str, Any]]:
    """Type guard for valid trading configurations.

    Args:
        config: Value to check

    Returns:
        True if value is a valid trading config dict
    """
    if not is_valid_config(config):
        return False

    required_keys = ['initial_balance', 'max_position_size']
    return all(key in config for key in required_keys)


def is_valid_model_config(config: Any) -> TypeGuard[Dict[str, Any]]:
    """Type guard for valid model configurations.

    Args:
        config: Value to check

    Returns:
        True if value is a valid model config dict
    """
    if not is_valid_config(config):
        return False

    required_keys = ['learning_rate', 'batch_size']
    return all(key in config for key in required_keys)


def is_valid_reward(reward: Any) -> TypeGuard[float]:
    """Type guard for valid reward values.

    Args:
        reward: Value to check

    Returns:
        True if value is a valid finite float reward
    """
    return isinstance(reward, (int, float, np.number)) and np.isfinite(float(reward))


def is_valid_action(action: Any) -> TypeGuard[int]:
    """Type guard for valid discrete actions.

    Args:
        action: Value to check

    Returns:
        True if value is a valid action (0=HOLD, 1=BUY, 2=SELL)
    """
    return isinstance(action, (int, np.integer)) and action in [0, 1, 2]


def is_valid_continuous_action(action: Any) -> TypeGuard[float]:
    """Type guard for valid continuous actions.

    Args:
        action: Value to check

    Returns:
        True if value is a valid continuous action in [-1, 1]
    """
    return isinstance(action, (int, float, np.number)) and -1.0 <= float(action) <= 1.0


def is_valid_observation(observation: Any) -> TypeGuard[NDArray[np.float32]]:
    """Type guard for valid observation arrays.

    Args:
        observation: Value to check

    Returns:
        True if value is a valid observation array
    """
    if not isinstance(observation, np.ndarray):
        return False

    return (observation.dtype == np.float32 and
            observation.ndim == 1 and
            np.all(np.isfinite(observation)))


def is_valid_probability_distribution(probs: Any) -> TypeGuard[NDArray[np.float32]]:
    """Type guard for valid probability distributions.

    Args:
        probs: Value to check

    Returns:
        True if value is a valid probability distribution
    """
    if not isinstance(probs, np.ndarray):
        return False

    return (probs.dtype == np.float32 and
            probs.ndim == 1 and
            np.all(np.isfinite(probs)) and
            np.all(probs >= 0) and
            np.all(probs <= 1) and
            np.isclose(np.sum(probs), 1.0, atol=1e-6))


def is_valid_portfolio_value(value: Any) -> TypeGuard[float]:
    """Type guard for valid portfolio values.

    Args:
        value: Value to check

    Returns:
        True if value is a valid positive portfolio value
    """
    return isinstance(value, (int, float, np.number)) and float(value) > 0


def is_valid_position(position: Any) -> TypeGuard[float]:
    """Type guard for valid position values.

    Args:
        position: Value to check

    Returns:
        True if value is a valid position (-2 to 2 range for safety)
    """
    return isinstance(position, (int, float, np.number)) and -2.0 <= float(position) <= 2.0


def is_valid_loss(loss: Any) -> TypeGuard[float]:
    """Type guard for valid loss values.

    Args:
        loss: Value to check

    Returns:
        True if value is a valid non-negative finite loss
    """
    return isinstance(loss, (int, float, np.number)) and float(loss) >= 0 and np.isfinite(float(loss))


def validate_type_with_guard(value: Any, type_guard_func: Callable[[Any], bool], type_name: str) -> None:
    """Validate a value using a type guard function.

    Args:
        value: Value to validate
        type_guard_func: Type guard function to use
        type_name: Human-readable type name for error messages

    Raises:
        TypeError: If validation fails
    """
    if not type_guard_func(value):
        raise TypeError(f"Expected {type_name}, got {type(value).__name__}: {value}")


def is_valid_config_dict(config: Any) -> TypeGuard[Dict[str, Any]]:
    """Type guard for valid configuration dictionaries.

    Args:
        config: Value to check

    Returns:
        True if value is a valid config dict
    """
    return isinstance(config, dict) and all(isinstance(k, str) for k in config.keys())


def is_valid_feature_array(features: Any) -> TypeGuard[NDArray[np.float32]]:
    """Type guard for valid feature arrays.

    Args:
        features: Value to check

    Returns:
        True if value is a valid feature array
    """
    if not isinstance(features, np.ndarray):
        return False

    return (features.dtype in [np.float32, np.float64] and
            features.ndim == 2 and  # (batch_size, feature_dim)
            np.all(np.isfinite(features)))


def is_valid_reward_array(rewards: Any) -> TypeGuard[NDArray[np.float32]]:
    """Type guard for valid reward arrays.

    Args:
        rewards: Value to check

    Returns:
        True if value is a valid reward array
    """
    if not isinstance(rewards, np.ndarray):
        return False

    return (rewards.dtype == np.float32 and
            rewards.ndim == 1 and
            np.all(np.isfinite(rewards)))


def is_valid_action_array(actions: Any) -> TypeGuard[NDArray[np.int64]]:
    """Type guard for valid action arrays.

    Args:
        actions: Value to check

    Returns:
        True if value is a valid action array
    """
    if not isinstance(actions, np.ndarray):
        return False

    return (actions.dtype == np.int64 and
            actions.ndim == 1 and
            np.all(np.isin(actions, [0, 1, 2])))


def validate_generic_type(value: Any, expected_type: Any, name: str) -> None:
    """Validate a value against a generic type (for advanced type checking).

    Args:
        value: Value to validate
        expected_type: Expected type (can be generic)
        name: Variable name for error messages

    Raises:
        TypeError: If validation fails
    """
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is None:
        # Simple type
        if not isinstance(value, expected_type):
            raise TypeError(f"{name} must be {expected_type.__name__}, got {type(value).__name__}")
    elif origin is list:
        if not isinstance(value, list):
            raise TypeError(f"{name} must be List, got {type(value).__name__}")
        if args and not all(isinstance(item, args[0]) for item in value):
            raise TypeError(f"{name} elements must be {args[0].__name__}")
    elif origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be Dict, got {type(value).__name__}")
        if args and len(args) >= 2:
            key_type, value_type = args[0], args[1]
            if not all(isinstance(k, key_type) for k in value.keys()):
                raise TypeError(f"{name} keys must be {key_type.__name__}")
            if not all(isinstance(v, value_type) for v in value.values()):
                raise TypeError(f"{name} values must be {value_type.__name__}")
    # Add more generic type validations as needed


def is_valid_dataframe(df: Any) -> TypeGuard[Any]:
    """Type guard for valid pandas DataFrame.

    Args:
        df: Value to check

    Returns:
        True if value is a non-empty pandas DataFrame
    """
    return bool(isinstance(df, pd.DataFrame) and not df.empty)


def is_valid_numpy_array(arr: Any) -> TypeGuard[NDArray[Any]]:
    """Type guard for valid numpy array.

    Args:
        arr: Value to check

    Returns:
        True if value is a valid numpy array
    """
    return bool(isinstance(arr, np.ndarray) and arr.size > 0)


def is_valid_features_dict(features: Any) -> TypeGuard[Dict[str, Any]]:
    """Type guard for valid features dictionary.

    Args:
        features: Value to check

    Returns:
        True if value is a valid features dict
    """
    if not isinstance(features, dict):
        return False

    # Check that all values are valid numpy arrays or lists
    for key, value in features.items():
        if not isinstance(key, str):
            return False
        if not (isinstance(value, (list, tuple)) or is_valid_numpy_array(value)):
            return False

    return len(features) > 0


def is_valid_price_data(data: Any) -> TypeGuard[NDArray[np.float64]]:
    """Type guard for valid price data array.

    Args:
        data: Value to check

    Returns:
        True if value is a valid price data array (positive finite values)
    """
    if not is_valid_numpy_array(data):
        return False

    return bool(np.all(np.isfinite(data)) and np.all(data > 0))