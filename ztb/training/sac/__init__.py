"""Shared SAC runtime helpers.

Canonical home for reusable SAC evaluation / split / env setup helpers that were
previously hosted under `scripts.v460.lib.sac_common`.
"""

from ztb.training.sac.runtime import (
    SACModelProtocol,
    TrainingEnvProtocol,
    _compute_g3_metrics,
    adjust_buffer_size,
    cleanup_envs,
    cleanup_training_resources,
    create_env_from_config,
    create_sac_model,
    evaluate_model_oos,
    extract_roi_from_env,
    train_val_split,
)

__all__ = [
    "SACModelProtocol",
    "TrainingEnvProtocol",
    "_compute_g3_metrics",
    "adjust_buffer_size",
    "cleanup_envs",
    "cleanup_training_resources",
    "create_env_from_config",
    "create_sac_model",
    "evaluate_model_oos",
    "extract_roi_from_env",
    "train_val_split",
]
