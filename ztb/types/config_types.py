"""
Type definitions for configuration files.

This module provides typed dictionaries and dataclasses for configuration
validation and IDE support.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from ztb.types.generics import ConfigDict


@dataclass
class TrainingConfig:
    """Configuration for training runs."""

    algorithm: str
    learning_rate: float
    batch_size: int
    total_timesteps: int
    checkpoint_interval: int
    eval_freq: int
    n_eval_episodes: int
    reward_scaling: float
    max_grad_norm: float
    ent_coef: float
    vf_coef: float
    clip_range: float
    n_epochs: int
    gae_lambda: float
    gamma: float
    target_kl: Optional[float] = None
    seed: Optional[int] = None


@dataclass
class EnvironmentConfig:
    """Configuration for trading environments."""

    pair: str
    timeframe: str
    initial_balance: float
    max_position_size: float
    transaction_cost: float
    reward_scaling: float
    feature_set: str
    curriculum_stage: str
    stop_loss_threshold: float
    max_consecutive_trades: int
    min_holding_period: int


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    features: List[str]
    feature_storage_dtype: str
    precision_columns: List[str]
    cache_enabled: bool
    parallel_enabled: bool
    seed: int


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    policy: str
    learning_rate: float
    batch_size: int
    n_steps: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float


# Type aliases for backward compatibility
TrainingConfigDict = Dict[str, Any]
EnvironmentConfigDict = Dict[str, Any]
FeatureConfigDict = Dict[str, Any]
ModelConfigDict = Dict[str, Any]