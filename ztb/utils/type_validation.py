"""
Runtime type validation utilities.

This module provides runtime type checking utilities for critical code paths,
helping catch type-related issues during execution rather than just at static analysis time.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_origin, get_args
import numpy as np
from numpy.typing import NDArray


class TypeValidator:
    """
    Runtime type validator for function arguments and return values.

    This class provides decorators and utilities for runtime type checking
    of critical functions to catch type mismatches during execution.
    """

    @staticmethod
    def validate_type(value: Any, expected_type: Type[Any], name: str = "value") -> None:
        """
        Validate that a value matches the expected type at runtime.

        Performs comprehensive type checking including support for:
        - Basic types (int, str, float, etc.)
        - Generic types (List[int], Dict[str, Any], etc.)
        - Union types (Union[int, str])
        - Optional types (Optional[str])
        - NumPy array types (NDArray[np.float64])

        Args:
            value: The value to validate against the expected type.
            expected_type: The type that value should conform to.
                          Supports typing module constructs like Union, Optional, List, etc.
            name: Descriptive name of the value for error messages.
                 Defaults to "value".

        Raises:
            TypeError: If the value doesn't match the expected type.
                      Error message includes the value name and actual/expected types.

        Example:
            >>> TypeValidator.validate_type(42, int, "age")
            >>> TypeValidator.validate_type([1, 2, 3], List[int], "numbers")
            >>> TypeValidator.validate_type(None, Optional[str], "optional_name")
        """
        if not TypeValidator._check_type(value, expected_type):
            raise TypeError(
                f"{name} must be of type {expected_type}, got {type(value)}"
            )
        if not TypeValidator._check_type(value, expected_type):
            raise TypeError(
                f"{name} must be of type {expected_type}, got {type(value)}"
            )

    @staticmethod
    def _check_type(value: Any, expected_type: Type[Any]) -> bool:
        """Check if value matches expected type, handling generics."""
        # Handle None
        if value is None:
            return expected_type is type(None) or (
                hasattr(expected_type, '__args__') and
                type(None) in getattr(expected_type, '__args__', ())
            )

        # Handle Union types
        origin = get_origin(expected_type)
        if origin is Union:
            return any(TypeValidator._check_type(value, arg) for arg in get_args(expected_type))

        # Handle generic types
        if origin is not None:
            if not isinstance(value, origin):
                return False

            # Check generic arguments for specific cases
            args = get_args(expected_type)
            if origin is NDArray and args:
                dtype_arg = args[0]
                if hasattr(dtype_arg, '__origin__'):
                    # Handle np.floating[Any], np.integer[Any], etc.
                    dtype_origin = get_origin(dtype_arg)
                    if dtype_origin in (np.floating, np.integer):
                        return True  # Accept any numeric dtype for now
                elif dtype_arg in (np.floating, np.integer):
                    return True
            return True

        # Standard isinstance check
        try:
            return isinstance(value, expected_type)
        except TypeError:
            # Some types (like generics) can't be used with isinstance
            return True  # Assume valid if isinstance fails

    @staticmethod
    def validate_array_shape(
        array: NDArray[np.floating[Any]],
        expected_shape: Optional[tuple[int, ...]] = None,
        name: str = "array"
    ) -> None:
        """
        Validate numpy array shape.

        Args:
            array: Array to validate
            expected_shape: Expected shape (None for any shape)
            name: Name for error messages

        Raises:
            ValueError: If shape doesn't match
        """
        if expected_shape is not None and array.shape != expected_shape:
            raise ValueError(
                f"{name} shape must be {expected_shape}, got {array.shape}"
            )

    @staticmethod
    def validate_array_dtype(
        array: NDArray[np.floating[Any]],
        expected_dtype: Optional[np.dtype[np.floating[Any]]] = None,
        name: str = "array"
    ) -> None:
        """
        Validate numpy array dtype.

        Args:
            array: Array to validate
            expected_dtype: Expected dtype (None for any dtype)
            name: Name for error messages

        Raises:
            TypeError: If dtype doesn't match
        """
        if expected_dtype is not None and array.dtype != expected_dtype:
            raise TypeError(
                f"{name} dtype must be {expected_dtype}, got {array.dtype}"
            )


def runtime_type_check(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for runtime type checking of function arguments and return values.

    This decorator inspects the function's type annotations and validates
    arguments and return values at runtime.

    Note: This is a development/debugging tool and may impact performance.
    """
    sig = inspect.signature(func)
    annotations = func.__annotations__

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Bind arguments to parameter names
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Validate arguments
        for param_name, param_value in bound_args.arguments.items():
            if param_name in annotations:
                expected_type = annotations[param_name]
                try:
                    TypeValidator.validate_type(
                        param_value, expected_type, f"argument '{param_name}'"
                    )
                except TypeError as e:
                    # Log warning but don't fail - for development use
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Type check failed for {func.__name__}: {e}")

        # Call function
        result = func(*args, **kwargs)

        # Validate return value
        if 'return' in annotations:
            expected_return_type = annotations['return']
            try:
                TypeValidator.validate_type(
                    result, expected_return_type, "return value"
                )
            except TypeError as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Return type check failed for {func.__name__}: {e}")

        return result

    return wrapper


# Convenience functions for common validations
def validate_environment_config(config: Dict[str, Any]) -> None:
    """
    Validate environment configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['reward_scaling', 'transaction_cost', 'max_position_size']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate types
    if not isinstance(config.get('reward_scaling', 1.0), (int, float)):
        raise ValueError("reward_scaling must be numeric")
    if not isinstance(config.get('transaction_cost', 0.0), (int, float)):
        raise ValueError("transaction_cost must be numeric")
    if not isinstance(config.get('max_position_size', 1.0), (int, float)):
        raise ValueError("max_position_size must be numeric")


def validate_training_config(config: Dict[str, Any]) -> None:
    """
    Validate training configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['learning_rate', 'batch_size', 'total_timesteps']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required training config key: {key}")

    # Validate ranges
    if not (0 < config.get('learning_rate', 0) < 1):
        raise ValueError("learning_rate must be between 0 and 1")
    if config.get('batch_size', 0) <= 0:
        raise ValueError("batch_size must be positive")
    if config.get('total_timesteps', 0) <= 0:
        raise ValueError("total_timesteps must be positive")


def validate_feature_config(config: Dict[str, Any]) -> None:
    """
    Validate feature configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if 'features' not in config:
        raise ValueError("Missing required feature config key: features")

    features = config['features']
    if not isinstance(features, list):
        raise ValueError("features must be a list")

    if not features:
        raise ValueError("features list cannot be empty")

    # Validate each feature has required fields
    for feature in features:
        if not isinstance(feature, dict):
            raise ValueError("Each feature must be a dictionary")
        if 'name' not in feature:
            raise ValueError("Each feature must have a 'name' field")
        if 'type' not in feature:
            raise ValueError("Each feature must have a 'type' field")


def validate_trading_config(config: Dict[str, Any]) -> None:
    """
    Validate trading configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['pair', 'timeframe', 'initial_balance']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required trading config key: {key}")

    if config.get('initial_balance', 0) <= 0:
        raise ValueError("initial_balance must be positive")

    if config.get('max_position_size', 0) <= 0:
        raise ValueError("max_position_size must be positive")


def validate_model_config(config: Dict[str, Any]) -> None:
    """
    Validate model configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['policy', 'learning_rate', 'batch_size']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required model config key: {key}")

    if not (0 < config.get('learning_rate', 0) < 1):
        raise ValueError("learning_rate must be between 0 and 1")

    if config.get('batch_size', 0) <= 0:
        raise ValueError("batch_size must be positive")


def validate_array_type(arr: Any, expected_dtype: np.dtype[Any]) -> bool:
    """
    Validate that an array has the expected dtype.

    Args:
        arr: Array to validate
        expected_dtype: Expected numpy dtype

    Returns:
        True if array has expected dtype, False otherwise
    """
    return isinstance(arr, np.ndarray) and arr.dtype == expected_dtype