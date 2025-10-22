from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np

from ztb.config import ConfigManager
from ztb.types.common import Action, AnalysisData, TrainingData
from ztb.types.config_types import EnvironmentConfig, TrainingConfig
from ztb.utils.logging_utils import get_logger

# More specific config type
ComponentConfig = Union[TrainingConfig, EnvironmentConfig, Dict[str, Any]]


class BaseComponent(ABC):
    """Base class for all ZTB components.

    Provides common functionality like logging, configuration access,
    and lifecycle management.
    """

    def __init__(self, name: str, config: Optional[ComponentConfig] = None):
        """Initialize base component.

        Args:
            name: Component name for logging
            config: Optional component-specific configuration
        """
        self.name = name
        self.config = config or {}
        self.logger = get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self.config_manager = ConfigManager.get_instance()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the component. Override in subclasses."""
        self._initialized = True
        self.logger.info(f"Component {self.name} initialized")

    def shutdown(self) -> None:
        """Shutdown the component. Override in subclasses."""
        self._initialized = False
        self.logger.info(f"Component {self.name} shutdown")

    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        elif hasattr(self.config, key):
            return getattr(self.config, key, default)
        return default

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update component configuration."""
        if isinstance(self.config, dict):
            self.config.update(updates)
        else:
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        self.logger.debug(f"Updated config for {self.name}: {updates}")


class BaseTradingAgent(BaseComponent):
    """Base class for trading agents.

    Provides common trading functionality like position management,
    risk control, and market data access.
    """

    def __init__(self, name: str, config: Optional[ComponentConfig] = None):
        super().__init__(name, config)
        self.position_size = 0.0
        self.portfolio_value = 0.0

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> Action:
        """Get trading action based on observation."""
        pass

    @abstractmethod
    def update_position(self, action: Action, price: float) -> None:
        """Update position based on action and current price."""
        pass

    def get_position_info(self) -> Dict[str, Any]:
        """Get current position information."""
        return {
            "position_size": self.position_size,
            "portfolio_value": self.portfolio_value,
        }


class BaseAnalyzer(BaseComponent):
    """Base class for analysis components.

    Provides common analysis functionality like data processing,
    metric calculation, and result formatting.
    """

    def __init__(self, name: str, config: Optional[ComponentConfig] = None):
        super().__init__(name, config)
        self.results: Dict[str, Any] = {}

    @abstractmethod
    def analyze(self, data: AnalysisData) -> Dict[str, Any]:
        """Perform analysis on data."""
        pass

    def save_results(self, path: str) -> None:
        """Save analysis results."""
        import json

        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Results saved to {path}")

    def load_results(self, path: str) -> None:
        """Load analysis results."""
        import json

        with open(path, "r") as f:
            self.results = json.load(f)
        self.logger.info(f"Results loaded from {path}")


class BaseTrainer(BaseComponent):
    """Base class for training components.

    Provides common training functionality like model saving/loading,
    training loop management, and evaluation.
    """

    def __init__(self, name: str, config: Optional[ComponentConfig] = None):
        super().__init__(name, config)
        self.model = None
        self.training_stats: Dict[str, Any] = {}

    @abstractmethod
    def train(self, data: TrainingData) -> Dict[str, Any]:
        """Train the model."""
        pass

    @abstractmethod
    def evaluate(self, data: TrainingData) -> Dict[str, Any]:
        """Evaluate the model."""
        pass

    def save_model(self, path: str) -> None:
        """Save trained model."""
        if self.model:
            # Assume model has save method
            self.model.save(path)
            self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load trained model."""
        # Assume model class has load method
        self.model = self._load_model(path)
        self.logger.info(f"Model loaded from {path}")

    @abstractmethod
    def _load_model(self, path: str) -> Any:
        """Load model implementation."""
        pass
