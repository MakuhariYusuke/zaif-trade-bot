"""Compatibility layer for legacy imports.

The unified trainer implementation now lives in `ztb.training.unified_trainer`.

This module re-exports the public API so existing imports remain functional
without duplicating the implementation.
"""

from __future__ import annotations

from ztb.training.unified_trainer import (
    UnifiedAlgorithm,
    UnifiedTrainer,
    UnifiedTrainerConfig,
    configure_progress_bar,
    load_config,
    main,
)

__all__ = [
    "UnifiedAlgorithm",
    "UnifiedTrainerConfig",
    "UnifiedTrainer",
    "configure_progress_bar",
    "load_config",
    "main",
]
