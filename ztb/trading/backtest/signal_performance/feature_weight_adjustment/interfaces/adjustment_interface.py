"""
Weight Adjustment Interface

Defines the contract for weight adjustment implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd


class WeightAdjustmentInterface(ABC):
    """
    Interface for feature weight adjustment algorithms.

    This interface defines the contract that all weight adjustment
    strategies must implement.
    """

    @abstractmethod
    def adjust_weights(
        self,
        current_weights: Dict[str, float],
        performance_data: Dict[str, Any],
        feature_importance: Dict[str, float],
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Adjust feature weights based on performance data.

        Args:
            current_weights: Current feature weights
            performance_data: Performance metrics and analysis
            feature_importance: Feature importance scores
            market_conditions: Current market conditions (optional)

        Returns:
            Adjusted feature weights
        """
        pass

    @abstractmethod
    def get_adjustment_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the adjustment algorithm.

        Returns:
            Dictionary containing algorithm metadata
        """
        pass

    @abstractmethod
    def validate_weights(self, weights: Dict[str, float]) -> bool:
        """
        Validate that weights are properly normalized and valid.

        Args:
            weights: Feature weights to validate

        Returns:
            True if weights are valid, False otherwise
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the adjustment algorithm to initial state.
        """
        pass