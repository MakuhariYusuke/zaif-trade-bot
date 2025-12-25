"""
Generic types and utilities for improved type safety.

This module provides generic base classes and utilities that can be reused
across different components to ensure consistent typing patterns.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union, cast

import numpy as np
import pandas as pd

from ztb.utils.safety import safe_config_get

# Generic type variables
TConfig = TypeVar("TConfig")
TState = TypeVar("TState")
TValue = TypeVar("TValue")


class ConfigurableMixin(Generic[TConfig]):
    """
    Mixin class for components that can be configured with a dictionary.

    This provides a standard interface for configuration management with
    proper typing support.
    """

    def __init__(self, config: Optional[TConfig] = None) -> None:
        self._config: TConfig = config if config is not None else cast(TConfig, {})

    @property
    def config(self) -> TConfig:
        """Get current configuration."""
        return self._config

    @config.setter
    def config(self, value: TConfig) -> None:
        """Set configuration."""
        self._config = value

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        if hasattr(self._config, "update") and isinstance(self._config, dict):
            self._config.update(updates)

    def get_config_value(
        self, key: str, default: Optional[TValue] = None
    ) -> Optional[TValue]:
        """Get configuration value with optional default."""
        if isinstance(self._config, dict):
            return safe_config_get(self._config, key, default)
        return default


class StatisticsTracker(Generic[TState]):
    """
    Generic statistics tracker for monitoring component state.

    Provides a standard interface for tracking and reporting statistics
    across different types of components.
    """

    def __init__(self) -> None:
        self._statistics: Dict[str, Union[int, float, str, bool, None]] = {}
        self._state_history: list[TState] = []

    def update_statistics(
        self, key: str, value: Union[int, float, str, bool, None]
    ) -> None:
        """Update a specific statistic."""
        self._statistics[key] = value

    def get_statistics(self) -> Dict[str, Union[int, float, str, bool, None]]:
        """Get all current statistics."""
        return self._statistics.copy()

    def record_state(self, state: TState) -> None:
        """Record current state in history."""
        self._state_history.append(state)

    def get_state_history(self) -> list[TState]:
        """Get history of recorded states."""
        return self._state_history.copy()

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._statistics.clear()
        self._state_history.clear()


class ValidatableMixin(ABC):
    """
    Mixin for components that support validation.

    Provides a standard validation interface that can be implemented
    by different types of components.
    """

    @abstractmethod
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the component's current state.

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        ...

    def is_valid(self) -> bool:
        """Check if component is in a valid state."""
        valid, _ = self.validate()
        return valid

    def get_validation_errors(self) -> list[str]:
        """Get list of validation error messages."""
        _, errors = self.validate()
        return errors


# Type aliases for common patterns
ConfigDict = Dict[str, Any]
PathLike = Union[str, Path]
NumericArray = np.ndarray
FloatArray = np.ndarray
IntArray = np.ndarray

# Data model type aliases
DataFrame = pd.DataFrame
Series = pd.Series
ArrayLike = Union[np.ndarray[Any, np.dtype[Any]], List[float], List[int]]
Scalar = Union[int, float, bool, str]

# Trading domain types
PriceData = Dict[str, Union[float, int]]
MarketData = Dict[str, Any]
TradeAction = int
PositionSize = float
PortfolioValue = float
