#!/usr/bin/env python3
"""
Trading Strategy Base Classes

Base classes and protocols for trading strategies in the unified backtest framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Protocol, Union

import pandas as pd


class TradingStrategy(Protocol):
    """
    Protocol for trading strategies in the unified backtest framework.

    All trading strategies should implement this protocol.
    """

    @property
    def name(self) -> str:
        """Strategy name."""
        ...

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_position: int
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        Generate trading signal.

        Args:
            data: Market data with OHLCV and features
            current_position: Current position (-1, 0, 1 for short, flat, long)

        Returns:
            Signal dict with 'action' and optional parameters
        """
        ...

    def update_hyperparameters(self, hyperparameters: Dict[str, float]) -> None:
        """
        Update strategy hyperparameters.

        Args:
            hyperparameters: Dictionary of hyperparameter names and values
        """
        ...


class BaseTradingStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Provides common functionality for all trading strategies.
    """

    def __init__(self, name: str):
        """
        Initialize the trading strategy.

        Args:
            name: Strategy name
        """
        self._name = name
        self.config: Dict[str, Union[str, int, float, bool]] = {}
        self.is_initialized = False

    @property
    def name(self) -> str:
        """Strategy name."""
        return self._name

    @abstractmethod
    def initialize(
        self,
        data: pd.DataFrame,
        backtest_config: 'BacktestConfig',
        **kwargs
    ) -> None:
        """
        Initialize the strategy with data and configuration.

        Args:
            data: Market data for backtesting
            backtest_config: Backtest configuration
            **kwargs: Additional initialization parameters
        """
        pass

    @abstractmethod
    def generate_signal(
        self,
        data: pd.DataFrame,
        current_position: int
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        Generate a trading signal for the current market data.

        Args:
            data: Market data with OHLCV and features
            current_position: Current position (-1, 0, 1 for short, flat, long)

        Returns:
            Signal dict with 'action' and optional parameters
        """
        pass

    def update_hyperparameters(self, hyperparameters: Dict[str, float]) -> None:
        """
        Update strategy hyperparameters.

        Args:
            hyperparameters: Dictionary of hyperparameter names and values
        """
        # Default implementation - override in subclasses if needed
        pass

    def get_config(self) -> Dict[str, Union[str, int, float, bool]]:
        """Get strategy configuration."""
        return self.config.copy()

    def set_config(self, config: Dict[str, Union[str, int, float, bool]]) -> None:
        """Set strategy configuration."""
        self.config.update(config)

    def reset(self) -> None:
        """Reset strategy state."""
        self.is_initialized = False
        self.config = {}


class MLTradingStrategy(BaseTradingStrategy):
    """
    Base class for machine learning-based trading strategies.

    Provides common functionality for ML strategies like model loading,
    feature engineering, and prediction.
    """

    def __init__(self, name: str, model_path: Optional[str] = None):
        """
        Initialize ML trading strategy.

        Args:
            name: Strategy name
            model_path: Path to trained model
        """
        super().__init__(name)
        self.model_path = model_path
        self.model = None
        self.feature_engineer: Optional['FeatureEngineer'] = None

    def load_model(self) -> None:
        """Load the trained model."""
        if self.model_path:
            # Implementation depends on model type (SAC, etc.)
            pass

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess market data for model input.

        Args:
            data: Raw market data

        Returns:
            Preprocessed data
        """
        if self.feature_engineer:
            return self.feature_engineer.engineer_features(data)
        return data

    def predict(self, features: pd.DataFrame) -> Optional[list]:
        """
        Make prediction using the loaded model.

        Args:
            features: Feature data

        Returns:
            Model prediction
        """
        if self.model:
            return self.model.predict(features)
        return None


class SignalBasedStrategy(BaseTradingStrategy):
    """
    Base class for signal-based trading strategies.

    Provides common functionality for strategies that generate signals
    based on technical indicators, patterns, etc.
    """

    def __init__(self, name: str):
        """Initialize signal-based strategy."""
        super().__init__(name)
        self.indicators: Dict[str, 'Indicator'] = {}
        self.signals_history: list = []

    def add_indicator(self, name: str, indicator: 'Indicator') -> None:
        """
        Add a technical indicator.

        Args:
            name: Indicator name
            indicator: Indicator instance
        """
        self.indicators[name] = indicator

    def get_indicator_value(self, name: str, data: pd.DataFrame, index: int) -> Optional[float]:
        """
        Get indicator value for current data point.

        Args:
            name: Indicator name
            data: Market data
            index: Current index

        Returns:
            Indicator value
        """
        if name in self.indicators:
            return self.indicators[name].calculate(data, index)
        return None

    def store_signal(self, signal: Dict[str, Union[str, int, float, bool]]) -> None:
        """Store signal in history."""
        self.signals_history.append(signal)