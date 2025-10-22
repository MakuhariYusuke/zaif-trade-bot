"""
Lightweight common type aliases to centralize and reduce direct uses of `Any`.

Add conservative aliases here and prefer importing them instead of `typing.Any` in
new or refactored modules. This module intentionally avoids introducing new
strict types — it provides pragmatic shims that make incremental typing easier.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypedDict, Union

import numpy as np

JSONSerializable = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
ConfigDict = Dict[str, Any]
OptConfigDict = Optional[ConfigDict]
PathLike = Union[str, Path]


from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypedDict, Union

import numpy as np

JSONSerializable = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
ConfigDict = Dict[str, Any]  # Keep for backward compatibility
OptConfigDict = Optional[ConfigDict]
PathLike = Union[str, Path]


# More specific types
class TrainingStats(TypedDict):
    """Training statistics with proper typing."""

    total_steps: int
    episodes_completed: int
    average_reward: float
    best_reward: float
    loss: Optional[float]
    learning_rate: float
    epsilon: Optional[float]


class ActionDistribution(TypedDict):
    """Action distribution for discrete actions."""

    action: int
    probability: float
    q_value: Optional[float]


class ModelResult(TypedDict):
    """Model evaluation result."""

    model_path: str
    metrics: Dict[str, float]
    predictions: Optional[List[Any]]
    confidence: Optional[float]


class BaseAlgorithmTrainer(Protocol):
    """Minimal protocol describing the algorithm trainer used by UnifiedTrainer.

    Keep this intentionally small and conservative; add attributes used by
    UnifiedTrainer to avoid using `object` or raw `Any` in that module.
    """

    model: Any
    dataloader: Any

    def train(self) -> bool:  # returns success flag
        ...

    def get_model_state(self) -> Dict[str, Any]:
        ...

    def set_model_state(self, state: Dict[str, Any]) -> None:
        ...

    def get_training_stats(self) -> Dict[str, Any]:
        ...


class EnsemblePredictor(Protocol):
    def get_ensemble_stats(self) -> Dict[str, Any]:
        ...

    def adapt_ensemble(self, market_conditions: Dict[str, Any]) -> None:
        ...


class AnomalyDetectorProtocol(Protocol):
    def fit_ml_detectors(
        self, training_data: "np.ndarray"
    ) -> bool:  # returns success flag
        ...

    def detect_anomalies(
        self, data: "np.ndarray", feature_names: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        ...


class MetaLearnerProtocol(Protocol):
    meta_learner: Any

    def train_on_markets(self, num_epochs: int) -> Dict[str, Any]:
        ...


class FederatedLearnerProtocol(Protocol):
    def train_all_markets(self, loss_fn: Any) -> Dict[str, Any]:
        ...

    def get_federated_stats(self) -> Dict[str, Any]:
        ...


class ContinualLearnerProtocol(Protocol):
    def learn_task(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        ...


class TrainingReporterProtocol(Protocol):
    def generate_ensemble_report(
        self, ensemble_stats: Dict[str, Any], decision_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        ...

    def save_ensemble_report(self, report: Dict[str, Any]) -> str:
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
    specializations: List[str]


# Conservative protocol for SAC-like model objects used in trainers/analyzers.
# Keep minimal to avoid over-constraining implementations from SB3 or custom wrappers.
class SACLikeModelProtocol(Protocol):
    """Minimal subset of attributes/methods expected from a SAC model instance.

    This is intentionally small: used to replace raw `Any` where code relies on
    predictable attributes (predict(), logger, device, replay_buffer, observation_space).
    """

    # Runtime prediction API (may return action or (action, state))
    def predict(
        self, obs: Any, deterministic: bool = True
    ) -> Any:  # pragma: no cover - runtime
        ...

    # Optional logger used by SB3 models
    logger: Any

    # Device string used for torch tensors on the model
    device: Any

    # Replay buffer used by SAC implementations (if available)
    replay_buffer: Any

    # Observation space (has .shape)
    observation_space: Any


# For code that currently uses Any for runtime-flexible data structures,
# prefer JSONSerializable or ConfigDict where appropriate. Use this file as
# the central place to add more precise aliases over time.
