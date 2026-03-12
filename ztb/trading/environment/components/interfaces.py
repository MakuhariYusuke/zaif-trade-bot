"""
Interfaces for reward calculation components.

This module defines interfaces for the reward calculation system components,
following SOLID principles for better maintainability and testability.
"""

from abc import ABC, abstractmethod

import numpy as np

class IMarketRegimeDetector(ABC):
    """Interface for market regime detection."""

    @abstractmethod
    def detect_regime(self, current_price: float, step: int) -> str:
        """
        Detect current market regime.

        Args:
            current_price: Current market price
            step: Current step number

        Returns:
            Market regime: 'bull', 'bear', 'sideways', 'volatile'
        """
        pass

class IDynamicRewardShaper(ABC):
    """Interface for dynamic reward shaping."""

    @abstractmethod
    def shape_reward(
        self, base_reward: float, current_price: float, step: int, pnl: float
    ) -> float:
        """
        Apply dynamic reward shaping.

        Args:
            base_reward: Base reward before shaping
            current_price: Current market price
            step: Current step number
            pnl: Profit/Loss from action

        Returns:
            Shaped reward value
        """
        pass

class ISignalIntegrator(ABC):
    """Interface for signal integration."""

    @abstractmethod
    def integrate_signal(
        self, reward: float, observation: np.ndarray | None, action: int, step: int
    ) -> float:
        """
        Apply signal integration to reward.

        Args:
            reward: Base reward before signal integration
            observation: Current observation
            action: Action taken
            step: Current training step

        Returns:
            Modified reward with signal integration
        """
        pass

class IAsymmetricRewardScaler(ABC):
    """Interface for asymmetric reward scaling."""

    @abstractmethod
    def scale_reward(self, reward: float, position: float, pnl: float) -> float:
        """
        Apply asymmetric reward scaling.

        Args:
            reward: Base reward value
            position: Current position
            pnl: Profit/Loss from action

        Returns:
            Scaled reward value
        """
        pass
