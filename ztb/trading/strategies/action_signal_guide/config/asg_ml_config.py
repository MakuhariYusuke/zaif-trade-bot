"""
Machine Learning Configuration for Action Signal Guide.

This module provides configuration management for ML-based components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..interfaces.ml_interfaces import MLPredictionModel, OptimizationTarget


class MLModelType(Enum):
    """Machine learning model types."""

    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    CUSTOM = "custom"


@dataclass
class MLModelConfig:
    """Configuration for individual ML models."""

    model_type: MLPredictionModel
    framework: MLModelType
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_selection: bool = True
    cross_validation_folds: int = 5
    train_test_split: float = 0.2
    random_state: int = 42


@dataclass
class PatternOptimizerConfig:
    """Configuration for pattern optimization."""

    enabled: bool = True
    optimization_target: OptimizationTarget = OptimizationTarget.SIGNAL_ACCURACY
    update_frequency: int = 100  # Update every N signals
    min_training_samples: int = 1000
    max_training_samples: int = 10000
    feature_importance_threshold: float = 0.01
    advanced_features: bool = True  # Enable advanced ML features
    model_types: List[MLPredictionModel] = field(
        default_factory=lambda: [
            MLPredictionModel.RANDOM_FOREST,
            MLPredictionModel.GRADIENT_BOOSTING,
        ]
    )
    # Model-specific parameters
    random_forest_estimators: int = 100
    random_forest_max_depth: int = 10
    gradient_boosting_estimators: int = 100
    gradient_boosting_learning_rate: float = 0.1
    gradient_boosting_max_depth: int = 6
    models: List[MLModelConfig] = field(
        default_factory=lambda: [
            MLModelConfig(
                model_type=MLPredictionModel.RANDOM_FOREST,
                framework=MLModelType.SKLEARN,
                hyperparameters={"n_estimators": 100, "max_depth": 10},
            ),
            MLModelConfig(
                model_type=MLPredictionModel.GRADIENT_BOOSTING,
                framework=MLModelType.SKLEARN,
                hyperparameters={"n_estimators": 100, "learning_rate": 0.1},
            ),
        ]
    )

    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        if self.min_training_samples <= 0:
            return False
        if self.max_training_samples < self.min_training_samples:
            return False
        if self.update_frequency <= 0:
            return False
        if (
            self.feature_importance_threshold < 0
            or self.feature_importance_threshold > 1
        ):
            return False
        return True


@dataclass
class OnlineLearnerConfig:
    """Configuration for online learning."""

    enabled: bool = True
    learning_rate: float = 0.01
    adaptation_rate: float = 0.1
    memory_size: int = 1000
    feedback_weight: float = 0.3
    performance_window: int = 50
    min_adaptation_interval: int = 10
    max_parameter_change: float = 0.2


@dataclass
class EnsemblePredictorConfig:
    """Configuration for ensemble prediction."""

    enabled: bool = True
    ensemble_method: str = "weighted_average"  # weighted_average, stacking, blending
    base_models: List[MLModelConfig] = field(
        default_factory=lambda: [
            MLModelConfig(MLPredictionModel.LINEAR_REGRESSION, MLModelType.SKLEARN),
            MLModelConfig(MLPredictionModel.RANDOM_FOREST, MLModelType.SKLEARN),
            MLModelConfig(MLPredictionModel.GRADIENT_BOOSTING, MLModelType.SKLEARN),
        ]
    )
    meta_model: Optional[MLModelConfig] = None
    weight_update_frequency: int = 50
    min_model_weight: float = 0.1


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""

    enabled: bool = True
    max_features: int = 50
    feature_selection_method: str = (
        "mutual_info"  # mutual_info, f_test, recursive_elimination
    )
    scaling_method: str = "standard"  # standard, minmax, robust
    add_polynomial_features: bool = False
    polynomial_degree: int = 2
    add_time_features: bool = True
    add_statistical_features: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])


@dataclass
class MLIntegrationConfig:
    """Main configuration for ML integration."""

    enabled: bool = True
    pattern_optimizer: PatternOptimizerConfig = field(
        default_factory=PatternOptimizerConfig
    )
    online_learner: OnlineLearnerConfig = field(default_factory=OnlineLearnerConfig)
    ensemble_predictor: EnsemblePredictorConfig = field(
        default_factory=EnsemblePredictorConfig
    )
    feature_engineering: FeatureEngineeringConfig = field(
        default_factory=FeatureEngineeringConfig
    )

    # Global settings
    enable_gpu_acceleration: bool = False
    max_memory_usage: int = 1024  # MB
    model_save_path: str = "models/ml_models"
    log_level: str = "INFO"
    enable_model_persistence: bool = True

    def __post_init__(self):
        """Initialize default configurations."""
        # Ensure meta model is set for stacking
        if (
            self.ensemble_predictor.ensemble_method == "stacking"
            and self.ensemble_predictor.meta_model is None
        ):
            self.ensemble_predictor.meta_model = MLModelConfig(
                model_type=MLPredictionModel.LINEAR_REGRESSION,
                framework=MLModelType.SKLEARN,
            )

    def get_model_configs(self) -> Dict[str, MLModelConfig]:
        """Get all model configurations in a dictionary."""
        configs = {}

        # Pattern optimizer models
        for i, model in enumerate(self.pattern_optimizer.models):
            configs[f"pattern_optimizer_{i}"] = model

        # Ensemble predictor models
        for i, model in enumerate(self.ensemble_predictor.base_models):
            configs[f"ensemble_base_{i}"] = model

        if self.ensemble_predictor.meta_model:
            configs["ensemble_meta"] = self.ensemble_predictor.meta_model

        return configs

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check memory limits
        if self.max_memory_usage < 256:
            issues.append("max_memory_usage should be at least 256 MB")

        # Check feature limits
        if self.feature_engineering.max_features < 10:
            issues.append("max_features should be at least 10")

        # Check training sample requirements
        if self.pattern_optimizer.min_training_samples < 100:
            issues.append("min_training_samples should be at least 100")

        # Check learning rates
        if not 0 < self.online_learner.learning_rate <= 1:
            issues.append("learning_rate should be between 0 and 1")

        if not 0 < self.online_learner.adaptation_rate <= 1:
            issues.append("adaptation_rate should be between 0 and 1")

        return issues
