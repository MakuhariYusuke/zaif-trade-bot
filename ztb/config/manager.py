"""
Deprecated manager module.
"""

from __future__ import annotations

import warnings

from ztb.config.managers.ztb_manager import (
    ZaifTradeBotConfigManager,
    config_manager,
)

warnings.warn(
    "ztb.config.manager is deprecated; use ztb.config.managers.ztb_manager",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ZaifTradeBotConfigManager", "config_manager"]
