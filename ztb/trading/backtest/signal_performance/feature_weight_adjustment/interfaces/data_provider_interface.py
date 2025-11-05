"""
Data Provider Interface

Defines the contract for data providers that supply SAC learning
and signal performance data to the weight adjustment system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd


class DataProviderInterface(ABC):
    """
    Interface for data providers.

    This interface defines how data providers should supply
    SAC learning data and signal performance metrics.
    """

    @abstractmethod
    def get_sac_learning_data(
        self,
        time_range: Optional[tuple] = None,
        episode_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get SAC learning data for weight adjustment.

        Args:
            time_range: Time range for data (start, end) timestamps
            episode_limit: Maximum number of episodes to retrieve

        Returns:
            SAC learning data including rewards, losses, actions
        """
        pass

    @abstractmethod
    def get_signal_performance_data(
        self,
        time_range: Optional[tuple] = None,
        signal_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get signal performance data for weight adjustment.

        Args:
            time_range: Time range for data (start, end) timestamps
            signal_types: Types of signals to include

        Returns:
            Signal performance metrics and analysis
        """
        pass

    @abstractmethod
    def get_feature_importance_data(
        self,
        features: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get current feature importance scores.

        Args:
            features: Specific features to get importance for

        Returns:
            Feature importance scores
        """
        pass

    @abstractmethod
    def get_market_conditions(self) -> Dict[str, Any]:
        """
        Get current market conditions.

        Returns:
            Market condition indicators
        """
        pass

    @abstractmethod
    def is_data_available(self) -> bool:
        """
        Check if sufficient data is available for adjustment.

        Returns:
            True if data is available, False otherwise
        """
        pass

    @abstractmethod
    def get_data_quality_metrics(self) -> Dict[str, float]:
        """
        Get data quality metrics.

        Returns:
            Data quality scores and statistics
        """
        pass