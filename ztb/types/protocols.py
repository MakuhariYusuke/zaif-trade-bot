"""
Type protocols for the Zaif Trading Bot.

This module defines protocol interfaces that standardize the behavior
of key components in the trading system.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
import pandas as pd


class TradingEnvironment(Protocol):
    """Protocol for trading environments."""

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to initial state."""
        ...

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        ...

    def render(self, mode: str = 'human') -> Optional[Any]:
        """Render the environment."""
        ...

    def close(self) -> None:
        """Clean up environment resources."""
        ...


class FeatureRegistryProtocol(Protocol):
    """Protocol for feature registries."""

    def register_feature(self, name: str, feature_func: Any) -> None:
        """Register a feature function."""
        ...

    def get_feature(self, name: str) -> Any:
        """Get a registered feature function."""
        ...

    def list_features(self) -> List[str]:
        """List all registered feature names."""
        ...

    def compute_features(self, data: Any) -> Union[Dict[str, Any], pd.DataFrame]:
        """Compute all registered features for given data."""
        ...


class SerializableProtocol(Protocol):
    """Protocol for serializable objects."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary representation."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SerializableProtocol":
        """Create object from dictionary representation."""
        ...