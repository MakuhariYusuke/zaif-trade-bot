from __future__ import annotations

from ztb.training.unified_trainer.runtime_flags import (
    resolve_ensemble_enabled,
    resolve_trainer_runtime_flags,
)
from ztb.training.unified_trainer.trainer import UnifiedTrainer


def test_resolve_ensemble_enabled_handles_missing_structure() -> None:
    assert resolve_ensemble_enabled({}) is False
    assert resolve_ensemble_enabled({"v427_advanced_features": None}) is False
    assert (
        resolve_ensemble_enabled(
            {"v427_advanced_features": {"ensemble_system": {"enabled": True}}}
        )
        is True
    )


def test_resolve_trainer_runtime_flags_combines_related_feature_gates() -> None:
    flags = resolve_trainer_runtime_flags(
        {
            "enable_federated": True,
            "federated_markets": True,
            "enable_continual_learning": True,
            "enable_mixed_precision": True,
        },
        enable_distributed=True,
        world_size=2,
        ensemble_enabled=True,
    )

    assert flags.distributed_training_enabled is True
    assert flags.federated_learning_enabled is True
    assert flags.market_federated_learning_enabled is True
    assert flags.continual_learning_enabled is True
    assert flags.mixed_precision_enabled is True
    assert flags.ensemble_enabled is True


def test_unified_trainer_uses_runtime_flag_helper_for_ensemble_init() -> None:
    trainer = UnifiedTrainer(
        {
            "training": {"algorithm": "ppo", "total_timesteps": 1000},
            "v427_advanced_features": {"ensemble_system": {"enabled": True}},
        }
    )

    assert trainer.ensemble_enabled is True
