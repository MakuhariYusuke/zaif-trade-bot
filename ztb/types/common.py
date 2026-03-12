"""
Lightweight common type aliases to centralize and reduce direct uses of `Any`.

Add conservative aliases here and prefer importing them instead of `typing.Any` in
new or refactored modules. This module intentionally avoids introducing new
strict types — it provides pragmatic shims that make incremental typing easier.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    TypeGuard,
    Union,
)

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Static type-checking only - avoid importing heavy runtime libs during test
    # collection to prevent unintended imports of `torch` or `stable_baselines3`.
    from gymnasium import spaces  # type: ignore
    from stable_baselines3.common.logger import Logger  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
else:
    Logger = object
    spaces = object
    DataLoader = object

# Action type for trading environments
Action = int | float | np.ndarray

# Data types for analysis and training
AnalysisData = pd.DataFrame | np.ndarray | dict[str, object] | list[dict[str, object]]
TrainingData = pd.DataFrame | np.ndarray | dict[str, object] | list[dict[str, object]]

class SB3ModelProtocol(Protocol):
    """Protocol for Stable Baselines 3 models (SAC, PPO, etc.)."""

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> np.ndarray | tuple[np.ndarray, object]:
        """Predict action from observation."""
        ...

    def save(self, path: str | Path) -> None:
        """Save model to path."""
        ...

    @classmethod
    def load(
        cls, path: str | Path, env: object | None = None
    ) -> "SB3ModelProtocol":
        """Load model from path."""
        ...

    logger: Logger
    device: str
    observation_space: spaces.Space

# Backwards compatibility alias: AlertLevel is defined in types/alert_types.py
try:
    from ztb.types.alert_types import AlertLevel  # type: ignore
except Exception:
    AlertLevel = object  # type: ignore

JSONSerializable = dict[str, "JSONSerializable"] | list["JSONSerializable"] | str | int | float | bool | None

# More specific config types
ConfigValue = str | int | float | bool | list["ConfigValue"] | dict[str, "ConfigValue"] | None

# Refined aliases for common use cases to improve static typing
NumericConfigValue = int | float
PrimitiveConfigValue = str | int | float | bool | None

# Type guard helpers to narrow ConfigValue unions at runtime
def is_numeric_config_value(value: ConfigValue) -> TypeGuard[NumericConfigValue]:
    return isinstance(value, (int, float))

def is_config_dict(value: ConfigValue) -> TypeGuard[dict[str, ConfigValue]]:
    return isinstance(value, dict)

def is_config_list(value: ConfigValue) -> TypeGuard[list[ConfigValue]]:
    return isinstance(value, list)

class BaseConfigDict(TypedDict, total=False):
    """Base configuration dictionary with common fields."""

    # Top-level structure
    version: str
    training: dict[str, ConfigValue]
    environment: dict[str, ConfigValue]
    features: list[str]
    model: dict[str, ConfigValue]
    evaluation: dict[str, ConfigValue]

    # Core settings (flattened for convenience)
    algorithm: str
    learning_rate: ConfigValue
    batch_size: ConfigValue
    total_timesteps: ConfigValue

    # Environment settings
    pair: str
    timeframe: str
    initial_balance: ConfigValue
    max_position_size: ConfigValue
    transaction_cost: ConfigValue
    reward_scaling: ConfigValue

    # Feature settings
    feature_set: str
    cache_enabled: bool

    # Model settings
    policy: str
    n_steps: ConfigValue
    n_epochs: ConfigValue
    gamma: ConfigValue
    gae_lambda: ConfigValue
    clip_range: ConfigValue
    ent_coef: ConfigValue
    vf_coef: ConfigValue
    max_grad_norm: ConfigValue

    # Evaluation settings
    checkpoint_interval: ConfigValue
    eval_freq: ConfigValue
    n_eval_episodes: ConfigValue

    # Advanced settings
    seed: int | None
    target_kl: ConfigValue | None
    buffer_size: ConfigValue | None
    tau: ConfigValue | None
    target_update_interval: ConfigValue | None
    curriculum_stage: str
    stop_loss_threshold: ConfigValue
    max_consecutive_trades: ConfigValue
    min_holding_period: ConfigValue
    feature_storage_dtype: str
    precision_columns: list[str]
    parallel_enabled: bool

    # Dynamic fields for extensibility
    v427_advanced_features: dict[str, ConfigValue]
    v433_adaptive: dict[str, ConfigValue]
    ensemble_system: dict[str, ConfigValue]

ConfigDict = BaseConfigDict | dict[str, ConfigValue]  # Fallback for dynamic or legacy configs
OptConfigDict = ConfigDict | None
PathLike = str | Path

# Transitional aliases for staged `Any` reduction.
# Prefer these aliases over ad-hoc `dict[str, object]` in new code.
ConfigSection = dict[str, ConfigValue]
JSONDict = dict[str, JSONSerializable]
JSONList = list[JSONSerializable]
MetricsDict = dict[str, float]
CountDict = dict[str, int]
StringMap = dict[str, str]
BoolMap = dict[str, bool]
ObjectMap = dict[str, object]
ObjectList = list[object]
ObjectRecords = list[ObjectMap]

# More specific types
class TrainingStats(TypedDict, total=False):
    """Training statistics with proper typing."""

    total_steps: int
    episodes_completed: int
    average_reward: float
    best_reward: float
    loss: float | None
    learning_rate: float
    epsilon: float | None
    training_time_seconds: float | None
    convergence_achieved: bool | None
    total_timesteps: int
    training_time: float
    steps_per_second: float
    model_path: str
    final_reward: float
    action_distribution: dict[str, float]
    curriculum_learning: bool
    stages_completed: int
    status: str
    # Extended fields for advanced training features
    optimization: dict[str, object]
    anomaly_detection: dict[str, object]
    meta_learning: dict[str, object]
    federated_learning: dict[str, object]
    continual_learning: dict[str, object]

class ActionDistribution(TypedDict):
    """Action distribution for discrete actions."""

    action: int
    probability: float
    q_value: float | None

class ModelResult(TypedDict):
    """Model evaluation result."""

    model_path: str
    metrics: dict[str, float]
    predictions: list[object] | None
    confidence: float | None

# Training-related lightweight protocols / aliases
TrainingConfig = ConfigDict  # Keep for backward compatibility

class EnsemblePredictor(Protocol):
    def get_ensemble_stats(self) -> dict[str, object]:
        ...

    def adapt_ensemble(self, market_conditions: dict[str, object]) -> None:
        ...

class AnomalyDetectorProtocol(Protocol):
    def fit_ml_detectors(
        self, training_data: "np.ndarray"
    ) -> bool:  # returns success flag
        ...

    def detect_anomalies(
        self, data: "np.ndarray", feature_names: list[str] | None = None
    ) -> tuple[bool, dict[str, object]]:
        ...

class MetaLearnerProtocol(Protocol):
    meta_learner: object

    def train_on_markets(self, num_epochs: int) -> dict[str, object]:
        ...

class FederatedLearnerProtocol(Protocol):
    def train_all_markets(self, loss_fn: object) -> dict[str, object]:
        ...

    def get_federated_stats(self) -> dict[str, object]:
        ...

class ContinualLearnerProtocol(Protocol):
    def learn_task(self, *args: object, **kwargs: object) -> dict[str, object]:
        ...

class TrainingReporterProtocol(Protocol):
    def generate_ensemble_report(
        self, ensemble_stats: dict[str, object], decision_log: list[dict[str, object]]
    ) -> dict[str, object]:
        ...

    def save_ensemble_report(self, report: dict[str, object]) -> str:
        ...

class TrainingUIProtocol(Protocol):
    def print_warning(self, message: str) -> None:
        ...

    def print_error(self, message: str) -> None:
        ...

    def print_info(self, message: str) -> None:
        ...

# Lightweight TypedDicts for common hyperparameter/config shapes
class SACHyperparams(TypedDict, total=False):
    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    tau: float
    gamma: float
    ent_coef: float
    target_update_interval: int
    target_entropy: float
    train_freq: int

class EnsembleConfigDict(TypedDict, total=False):
    members: int
    voting_mechanism: str
    specializations: list[str]

# Conservative protocol for SAC-like model objects used in trainers/analyzers.
# Keep minimal to avoid over-constraining implementations from SB3 or custom wrappers.
class SACLikeModelProtocol(Protocol):
    """Minimal subset of attributes/methods expected from a SAC model instance.

    This is intentionally small: used to replace raw `Any` where code relies on
    predictable attributes (predict(), logger, device, replay_buffer, observation_space).
    """

    # Runtime prediction API (may return action or (action, state))
    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> np.ndarray | tuple[np.ndarray, object]:  # pragma: no cover - runtime
        ...

    # Optional logger used by SB3 models
    logger: Logger

    # Device string used for torch tensors on the model
    device: str

    # Replay buffer used by SAC implementations (if available)
    replay_buffer: object | None  # Keep as Any for now - complex buffer types

    # Observation space (has .shape)
    observation_space: spaces.Space

# Base classes for components
class BaseComponent:
    """Base class for all system components.

    This base class accepts a flexible constructor to support legacy
    code and tests that pass `config` and other kwargs into
    component constructors. We store provided config for access
    while remaining backward compatible.
    """

    def __init__(
        self, name: str = "", config: dict[str, object] | None = None, **kwargs
    ):
        self.name = name or self.__class__.__name__
        self.config = config or {}
        # Allow additional keyword args passed by callers but ignore them
        # This ensures compatibility with code creating components with
        # extra args like 'config', 'enabled' or similar.
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_name(self) -> str:
        """Get component name."""
        return self.name

    def validate_config(self, config: dict[str, object]) -> bool:
        """Validate component configuration."""
        return True

    def get_status(self) -> dict[str, object]:
        """Get component status."""
        return {"name": self.name, "status": "active"}

# --- Layer 6 optimizer & curriculum types ---
class StageChangeEvent(TypedDict, total=False):
    """Event payload emitted by BalanceCurriculumManager on stage changes.

    Keys:
      - previous_stage: str
      - new_stage: str
      - step: int
      - emergency: bool
    """

    previous_stage: str
    new_stage: str
    step: int
    emergency: bool

class AppliedCandidateTelemetry(TypedDict, total=False):
    """Telemetry object written to reports on applied candidate.

    Keys:
      - candidate_id: str
      - applied_at: float (timestamp)
      - weights: dict[str, float]
      - composite_score: float
      - mean_sharpe: float
      - mean_total_return: float
    """

    candidate_id: str
    applied_at: float
    weights: dict[str, float]
    composite_score: float
    mean_sharpe: float
    mean_total_return: float

@dataclass
class CandidateConfigDataclass:
    config_path: str
    candidate_id: str

@dataclass
class CandidateScoreDataclass:
    candidate_id: str
    mean_sharpe: float = 0.0
    mean_total_return: float = 0.0
    composite_score: float = 0.0
    report_count: int = 0
    run_artifacts: list[str] | None = None

# Backwards-compatible aliases used throughout the repo
CandidateConfig = CandidateConfigDataclass
CandidateScore = CandidateScoreDataclass
