from __future__ import annotations

from typing import Mapping

from ztb.adaptation.continual_learning import ContinualLearningConfig


def extract_algorithm_model(algorithm_trainer: object | None) -> object | None:
    """Return the trainer model when available, otherwise None."""
    if algorithm_trainer is None or not hasattr(algorithm_trainer, "model"):
        return None
    return getattr(algorithm_trainer, "model")


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


__all__ = ["extract_algorithm_model", "build_continual_learning_config"]
