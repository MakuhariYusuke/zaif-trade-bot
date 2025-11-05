"""
Configuration for Feature Weight Adjustment System

Provides configuration classes and defaults for the weight adjustment system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class AdjustmentStrategyType(Enum):
    """Types of weight adjustment strategies."""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    CORRELATION_BASED = "correlation_based"
    PERFORMANCE_DRIVEN = "performance_driven"
    HYBRID = "hybrid"


class AdjustmentFrequency(Enum):
    """Frequency of weight adjustments."""
    PER_TRADE = "per_trade"
    PER_EPISODE = "per_episode"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    MANUAL = "manual"


@dataclass
class AdjustmentConfig:
    """
    Configuration for feature weight adjustment system.

    This class contains all configurable parameters for the weight
    adjustment system.
    """

    # Basic settings
    enabled: bool = True
    strategy_type: AdjustmentStrategyType = AdjustmentStrategyType.PERFORMANCE_DRIVEN
    adjustment_frequency: AdjustmentFrequency = AdjustmentFrequency.PER_EPISODE

    # Weight constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    weight_normalization: bool = True

    # Performance thresholds
    min_performance_threshold: float = 0.0
    max_adjustment_rate: float = 0.1  # Maximum change per adjustment

    # Learning parameters
    learning_rate: float = 0.01
    momentum: float = 0.9
    regularization_strength: float = 0.001

    # Data requirements
    min_data_points: int = 100
    max_history_length: int = 10000
    data_quality_threshold: float = 0.7

    # Feature settings
    feature_groups: Dict[str, List[str]] = field(default_factory=dict)
    protected_features: List[str] = field(default_factory=list)  # Features that shouldn't be adjusted

    # Strategy-specific parameters
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # Monitoring and logging
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True

    # Safety settings
    enable_safety_checks: bool = True
    max_consecutive_adjustments: int = 10
    rollback_on_failure: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not (0.0 <= self.min_weight <= self.max_weight <= 1.0):
            raise ValueError("Weight constraints must satisfy 0.0 <= min_weight <= max_weight <= 1.0")

        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("Learning rate must be between 0.0 and 1.0")

        if not (0.0 <= self.momentum <= 1.0):
            raise ValueError("Momentum must be between 0.0 and 1.0")

        if self.min_data_points < 1:
            raise ValueError("Minimum data points must be at least 1")

    def get_strategy_config(self, strategy_type: AdjustmentStrategyType) -> Dict[str, Any]:
        """Get configuration specific to a strategy type."""
        base_config = {
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "regularization_strength": self.regularization_strength,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "max_adjustment_rate": self.max_adjustment_rate,
        }

        # Add strategy-specific parameters
        strategy_specific = self.strategy_params.get(strategy_type.value, {})
        base_config.update(strategy_specific)

        return base_config

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Re-validate after update
        self._validate_config()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enabled": self.enabled,
            "strategy_type": self.strategy_type.value,
            "adjustment_frequency": self.adjustment_frequency.value,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "weight_normalization": self.weight_normalization,
            "min_performance_threshold": self.min_performance_threshold,
            "max_adjustment_rate": self.max_adjustment_rate,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "regularization_strength": self.regularization_strength,
            "min_data_points": self.min_data_points,
            "max_history_length": self.max_history_length,
            "data_quality_threshold": self.data_quality_threshold,
            "feature_groups": self.feature_groups.copy(),
            "protected_features": self.protected_features.copy(),
            "strategy_params": self.strategy_params.copy(),
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "enable_metrics": self.enable_metrics,
            "enable_safety_checks": self.enable_safety_checks,
            "max_consecutive_adjustments": self.max_consecutive_adjustments,
            "rollback_on_failure": self.rollback_on_failure,
        }


# Default configurations for different use cases
DEFAULT_BACKTEST_CONFIG = AdjustmentConfig(
    adjustment_frequency=AdjustmentFrequency.PER_EPISODE,
    max_adjustment_rate=0.05,  # Conservative for backtesting
    min_data_points=50,
    enable_safety_checks=True,
)

DEFAULT_LIVE_TRADING_CONFIG = AdjustmentConfig(
    adjustment_frequency=AdjustmentFrequency.PER_HOUR,
    max_adjustment_rate=0.02,  # Very conservative for live trading
    min_data_points=200,
    enable_safety_checks=True,
    rollback_on_failure=True,
)

DEFAULT_AGGRESSIVE_CONFIG = AdjustmentConfig(
    adjustment_frequency=AdjustmentFrequency.PER_TRADE,
    max_adjustment_rate=0.15,
    learning_rate=0.05,
    min_data_points=20,
    enable_safety_checks=False,  # Disable for maximum adaptation
)