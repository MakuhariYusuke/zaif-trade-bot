"""
Type protocols for trading environment interfaces.

This module defines protocols that standardize the interfaces for trading environments,
improving type safety and code maintainability across the codebase.
"""

from typing import Any, Dict, Optional, Protocol, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Forward declarations for optional dependencies
try:
    import pandas as pd
    DataFrame = pd.DataFrame
except ImportError:
    DataFrame = Any  # type: ignore[misc,assignment]
from gymnasium import spaces
from numpy.typing import NDArray


class TradingEnvironment(Protocol):
    """
    Protocol defining the standard interface for trading environments.

    This protocol ensures type safety when working with different trading environment
    implementations while maintaining compatibility with Gymnasium interfaces.
    """

    # Core environment properties
    observation_space: spaces.Space
    action_space: spaces.Discrete

    # Configuration
    config: Any  # EnvironmentConfig or similar

    # State properties
    current_step: int
    max_steps: Optional[int]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Initial observation and info dictionary
        """
        ...

    def step(
        self,
        action: int
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0=hold, 1=buy, 2=sell)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        ...

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current environment state.

        Returns:
            Rendered image as numpy array, or None if not supported
        """
        ...

    def close(self) -> None:
        """Clean up environment resources."""
        ...

    def _get_observation(self) -> NDArray[np.float32]:
        """
        Get current observation from environment state.

        Returns:
            Current observation vector
        """
        ...

    def _calculate_reward(
        self,
        action: int,
        prev_position: float,
        current_position: float,
        step_pnl: float
    ) -> float:
        """
        Calculate reward for the given action and state transition.

        Args:
            action: Action taken
            prev_position: Previous position (-1, 0, 1)
            current_position: New position (-1, 0, 1)
            step_pnl: Profit/loss for this step

        Returns:
            Calculated reward value
        """
        ...

    @property
    def portfolio_value(self) -> float:
        """Current portfolio value."""
        ...

    @property
    def position(self) -> float:
        """Current position (-1=short, 0=neutral, 1=long)."""
        ...

    @property
    def unrealized_pnl(self) -> float:
        """Current unrealized profit/loss."""
        ...


class FeatureRegistryProtocol(Protocol):
    """
    Protocol for feature registry interfaces.

    Standardizes the interface for feature extraction and management.
    """

    def get_feature_names(self) -> list[str]:
        """Get list of all available feature names."""
        ...

    def compute_features(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        feature_set: str = "full"
    ) -> NDArray[np.float32]:
        """
        Compute features for given data.

        Args:
            data: Input data (DataFrame or dict)
            feature_set: Which feature set to compute

        Returns:
            Computed feature matrix
        """
        ...

    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Feature metadata dictionary
        """
        ...

