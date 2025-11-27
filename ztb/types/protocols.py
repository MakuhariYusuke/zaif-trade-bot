"""
Type protocols for the Zaif Trading Bot.

This module defines protocol interfaces that standardize the behavior
of key components in the trading system.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable, Tuple, Union

import numpy as np
import pandas as pd

from ztb.types.common import Action, AnalysisData

try:
    from gymnasium import spaces
    from gymnasium.core import ActType
except ImportError:
    spaces = Any  # Fallback - gymnasium is required
    ActType = Any


@runtime_checkable
class TradingEnvironment(Protocol):
    """Protocol for trading environments."""

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        ...

    def step(
        self, action: Action
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        ...

    def render(self, mode: str = "human") -> Optional[Any]:
        """Render the environment."""
        ...

    def close(self) -> None:
        """Clean up environment resources."""
        ...


@runtime_checkable
class FeatureRegistryProtocol(Protocol):
    """Protocol for feature registries."""

    def register_feature(
        self,
        name: str,
        feature_func: Callable[[pd.DataFrame], Union[np.ndarray, pd.Series, float]],
    ) -> None:
        """Register a feature function."""
        ...

    def get_feature(
        self, name: str
    ) -> Callable[[pd.DataFrame], Union[np.ndarray, pd.Series, float]]:
        """Get a registered feature function."""
        ...

    def list_features(self) -> List[str]:
        """List all registered feature names."""
        ...

    def compute_features(
        self, data: AnalysisData
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """Compute all registered features for given data."""
        ...


@runtime_checkable
class SerializableProtocol(Protocol):
    """Protocol for serializable objects."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary representation."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SerializableProtocol":
        """Create object from dictionary representation."""
        ...
