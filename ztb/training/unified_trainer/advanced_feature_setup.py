from __future__ import annotations

from typing import Mapping

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


__all__ = [
    "build_continual_learning_config",
    "extract_algorithm_model",
    "resolve_model_input_dim",
    "resolve_model_output_dim",
]
