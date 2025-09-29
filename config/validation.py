"""
Configuration validation using Pydantic models.

This module provides type-safe configuration validation for experiment parameters.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class ExperimentConfigModel(BaseModel):
    """Pydantic model for experiment configuration validation"""

    # Required fields
    steps: int = Field(gt=0, description="Number of training steps")
    strategy: str = Field(description="Training strategy")

    # Optional fields with defaults
    dataset: str = Field(default="coingecko", description="Dataset to use")
    report_interval: int = Field(default=100, gt=0, description="Reporting interval")
    learning_rate: float = Field(default=0.001, gt=0, le=1, description="Learning rate")
    batch_size: int = Field(default=32, gt=0, description="Batch size")
    max_episodes: int = Field(default=1000, gt=0, description="Maximum episodes")
    test_split: float = Field(default=0.2, gt=0, lt=1, description="Test split ratio")
    validation_split: float = Field(
        default=0.1, gt=0, lt=1, description="Validation split ratio"
    )
    random_seed: Optional[int] = Field(default=None, ge=0, description="Random seed")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate strategy is one of allowed values"""
        allowed_strategies = {"generalization", "aggressive", "conservative"}
        if v not in allowed_strategies:
            raise ValueError(f"Strategy must be one of {allowed_strategies}")
        return v

    @field_validator("validation_split", "test_split")
    @classmethod
    def validate_splits(cls, v: float) -> float:
        """Ensure splits are valid"""
        if v >= 1.0:
            raise ValueError("Split ratios must be less than 1.0")
        return v

    class Config:
        """Pydantic configuration"""

        validate_assignment = True
        extra = "allow"  # Allow extra fields for flexibility


class FeatureEvaluationConfig(BaseModel):
    """Configuration for feature evaluation"""

    feature_name: str
    evaluation_method: str = Field(default="correlation")
    threshold: float = Field(default=0.1, ge=0, le=1)
    time_window_days: int = Field(default=30, gt=0)
    min_samples: int = Field(default=100, gt=0)

    @field_validator("evaluation_method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed_methods = {"correlation", "mutual_info", "importance", "stability"}
        if v not in allowed_methods:
            raise ValueError(f"Evaluation method must be one of {allowed_methods}")
        return v


class ParallelExperimentConfig(BaseModel):
    """Configuration for parallel experiment execution"""

    max_workers: int = Field(default=4, gt=0, le=16)
    batch_size: int = Field(default=10, gt=0)
    timeout_seconds: int = Field(default=3600, gt=0)  # 1 hour default
    retry_attempts: int = Field(default=3, ge=0, le=10)
    enable_priority_scheduling: bool = Field(default=True)
    shared_data_cache: bool = Field(default=True)
    resource_limits: Optional[Dict[str, Any]] = Field(default=None)


def validate_experiment_config(config_dict: Dict[str, Any]) -> ExperimentConfigModel:
    """
    Validate and convert a configuration dictionary to ExperimentConfigModel.

    Args:
        config_dict: Raw configuration dictionary

    Returns:
        Validated ExperimentConfigModel instance

    Raises:
        ValidationError: If configuration is invalid
    """
    return ExperimentConfigModel(**config_dict)


def load_and_validate_config(config_path: str) -> ExperimentConfigModel:
    """
    Load configuration from file and validate it.

    Args:
        config_path: Path to configuration file (JSON/YAML)

    Returns:
        Validated configuration model
    """
    import json

    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            config_dict = yaml.safe_load(f)
        else:
            config_dict = json.load(f)

    return validate_experiment_config(config_dict)
