from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class TrainerRuntimeFlags:
    ensemble_enabled: bool
    mixed_precision_enabled: bool
    distributed_training_enabled: bool
    federated_learning_enabled: bool
    market_federated_learning_enabled: bool
    continual_learning_enabled: bool


def resolve_ensemble_enabled(config: Mapping[str, object]) -> bool:
    advanced_features = config.get("v427_advanced_features", {})
    if not isinstance(advanced_features, dict):
        return False
    ensemble_system = advanced_features.get("ensemble_system", {})
    if not isinstance(ensemble_system, dict):
        return False
    return bool(ensemble_system.get("enabled", False))


def resolve_trainer_runtime_flags(
    config: Mapping[str, object],
    *,
    enable_distributed: bool,
    world_size: int,
    ensemble_enabled: bool | None = None,
) -> TrainerRuntimeFlags:
    resolved_ensemble_enabled = (
        resolve_ensemble_enabled(config)
        if ensemble_enabled is None
        else ensemble_enabled
    )
    federated_learning_enabled = bool(config.get("enable_federated", False))
    return TrainerRuntimeFlags(
        ensemble_enabled=resolved_ensemble_enabled,
        mixed_precision_enabled=bool(config.get("enable_mixed_precision", False)),
        distributed_training_enabled=enable_distributed and world_size > 1,
        federated_learning_enabled=federated_learning_enabled,
        market_federated_learning_enabled=(
            federated_learning_enabled and bool(config.get("federated_markets", False))
        ),
        continual_learning_enabled=bool(
            config.get("enable_continual_learning", False)
        ),
    )
