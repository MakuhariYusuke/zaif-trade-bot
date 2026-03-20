"""Compatibility shim for canonical SAC runtime helpers.

Canonical implementations now live in `ztb.training.sac.runtime`.
Keep this module as a thin compatibility layer while `scripts/v460/lib` callers
are migrated gradually and old test import paths remain valid.
"""

from ztb.training.sac.runtime import (  # noqa: F401
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
