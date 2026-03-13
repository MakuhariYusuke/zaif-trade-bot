"""
Deprecated base module for configuration loaders.
"""

from __future__ import annotations

import warnings

from ztb.config.core.base import BaseConfigLoader

warnings.warn(
    "ztb.config.base is deprecated; use ztb.config.core.base",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BaseConfigLoader"]
