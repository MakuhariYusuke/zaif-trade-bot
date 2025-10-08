"""
Training Callbacks - Legacy Compatibility Module.

This module provides backward compatibility for existing imports.
Individual callbacks have been reorganized into the callbacks_lib directory.

DEPRECATED: This file will be removed in a future version.
Use imports from ztb.training.callbacks_lib instead.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from ztb.training.callbacks is deprecated. "
    "Use imports from ztb.training.callbacks_lib instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all classes from the legacy module for full backward compatibility
from ztb.training.callbacks.callbacks_legacy import (  # noqa: F401
    BaseTrainingCallback,
    SimpleTrainingCallback,
    TradingTrainingCallback,
    ProgressTrainingCallback,
    EntropyScheduleCallback,
    CompositeTrainingCallback,
    CheckpointGCCallback,
)

__all__ = [
    "BaseTrainingCallback",
    "SimpleTrainingCallback",
    "TradingTrainingCallback",
    "ProgressTrainingCallback",
    "EntropyScheduleCallback",
    "CompositeTrainingCallback",
    "CheckpointGCCallback",
]
