"""
Trading features package.
特徴量パッケージ
"""

from __future__ import annotations

from types import ModuleType

from .core.registry import FeatureRegistry
from .feature_set_manager import get_feature_manager
from .microstructure import MICROSTRUCTURE_FEATURES, add_microstructure_features

def get_feature_manager() -> type[FeatureRegistry]:
    """Get the feature manager class"""
    return FeatureRegistry

# Lazy import feature modules - only import when needed
_FEATURE_MODULES: dict[str, ModuleType | None] = {
    "core": None,
    "generators": None,
    "processors": None,
    "models": None,
    "time": None,
    "utils": None,
}

def _ensure_module_loaded(module_name: str) -> None:
    """Ensure a feature module is loaded"""
    if _FEATURE_MODULES[module_name] is None:
        if module_name == "core":
            from . import core

            _FEATURE_MODULES[module_name] = core
        elif module_name == "generators":
            from . import generators

            _FEATURE_MODULES[module_name] = generators
        elif module_name == "processors":
            from . import processors

            _FEATURE_MODULES[module_name] = processors
        elif module_name == "models":
            from . import models

            _FEATURE_MODULES[module_name] = models
        elif module_name == "time":
            from . import time

            _FEATURE_MODULES[module_name] = time
        elif module_name == "utils":
            from . import utils

            _FEATURE_MODULES[module_name] = utils

__all__ = ["FeatureRegistry", "get_feature_manager", "add_microstructure_features", "MICROSTRUCTURE_FEATURES"]
