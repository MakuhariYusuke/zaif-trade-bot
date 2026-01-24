"""
Deprecated loader module.
"""

from __future__ import annotations

import warnings

from ztb.config.loaders.priority_loader import (
    ConfigLoader,
    PriorityConfigLoader,
    initialize_risk_profiles,
    load_config,
)

warnings.warn(
    "ztb.config.loader is deprecated; use ztb.config.loaders.priority_loader",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "PriorityConfigLoader",
    "ConfigLoader",
    "load_config",
    "initialize_risk_profiles",
]
