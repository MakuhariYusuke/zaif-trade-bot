from __future__ import annotations

from typing import Mapping, MutableMapping

from ztb.adaptation.continual_learning import ContinualLearningConfig


def extract_algorithm_model(algorithm_trainer: object | None) -> object | None:
    """Return the trainer model when available, otherwise None."""
    if algorithm_trainer is None or not hasattr(algorithm_trainer, "model"):
        return None
    model = getattr(algorithm_trainer, "model")
    return model if model is not None else None


def build_continual_learning_config(
    config: Mapping[str, object],
) -> ContinualLearningConfig:
    """Build ContinualLearningConfig from trainer config."""
    return ContinualLearningConfig(
        method=str(config.get("continual_method", "ewc")),
        ewc_lambda=float(config.get("continual_ewc_lambda", 0.1)),
        rehearsal_buffer_size=int(config.get("continual_buffer_size", 1000)),
        max_tasks_in_memory=int(config.get("continual_max_tasks", 5)),
        enable_memory_tracking=True,
    )


def resolve_model_input_dim(
    algorithm_trainer: object | None,
    default: int = 10,
) -> int:
    """Resolve model input dimension from an algorithm trainer when possible."""
    model = extract_algorithm_model(algorithm_trainer)
    if model is None:
        return default
    try:
        params_iter = iter(model.parameters())
        first_layer = next(params_iter)
        if first_layer is None:
            return default
        return int(
            first_layer.shape[1]
            if len(first_layer.shape) > 1
            else first_layer.shape[0]
        )
    except Exception:
        return default


def resolve_model_output_dim(
    algorithm_trainer: object | None,
    default: int = 1,
) -> int:
    """Resolve model output dimension from an algorithm trainer when possible."""
    model = extract_algorithm_model(algorithm_trainer)
    if model is None:
        return default
    try:
        params = list(model.parameters())
        if not params:
            return default
        last_layer = params[-1]
        return int(last_layer.shape[0])
    except Exception:
        return default


def collect_meta_learning_history(
    meta_learner: object | None,
    *,
    num_epochs: int = 50,
) -> object | None:
    """Train the meta learner only when task buffers are populated."""
    if meta_learner is None:
        return None
    nested = getattr(meta_learner, "meta_learner", None)
    if nested is not None and len(getattr(nested, "task_buffer", [])) > 0:
        return meta_learner.train_on_markets(num_epochs=num_epochs)
    if len(getattr(meta_learner, "task_buffer", [])) > 0:
        return meta_learner.train_on_markets(num_epochs=num_epochs)
    return None


def resolve_federated_stats(federated_learner: object | None) -> dict[str, object]:
    """Return federated stats when the learner exposes them, otherwise an empty payload."""
    if federated_learner is None or not hasattr(federated_learner, "get_federated_stats"):
        return {}
    stats = getattr(federated_learner, "get_federated_stats")()
    if isinstance(stats, dict):
        return stats
    return {}


def record_training_stat(
    training_stats: MutableMapping[str, object],
    key: str,
    value: object,
) -> None:
    """Persist advanced-feature outputs through a single trainer-owned path."""
    training_stats[key] = value


__all__ = [
    "build_continual_learning_config",
    "collect_meta_learning_history",
    "extract_algorithm_model",
    "record_training_stat",
    "resolve_model_input_dim",
    "resolve_model_output_dim",
    "resolve_federated_stats",
]
