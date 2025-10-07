"""
Type protocols for the Zaif Trading Bot.

This module defines protocol interfaces that standardize the behavior
of key components in the trading system.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
from abc import ABC, abstractmethod
import pandas as pd
import pandas as pd


class TradingEnvironment(Protocol):
    """Protocol for trading environments."""

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to initial state."""
        ...

    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        ...

    @abstractmethod
    def render(self, mode: str = 'human') -> Optional[Any]:
        """Render the environment."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        ...


class FeatureRegistryProtocol(Protocol):
    """Protocol for feature registries."""

    @abstractmethod
    def register_feature(self, name: str, feature_func: Any) -> None:
        """Register a feature function."""
        ...

    @abstractmethod
    def get_feature(self, name: str) -> Any:
        """Get a registered feature function."""
        ...

    @abstractmethod
    def list_features(self) -> List[str]:
        """List all registered feature names."""
        ...

    @abstractmethod
    def compute_features(self, data: Any) -> Union[Dict[str, Any], pd.DataFrame]:
        """Compute all registered features for given data."""
        ...