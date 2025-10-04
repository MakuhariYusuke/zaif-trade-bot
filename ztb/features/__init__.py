"""
Trading features package.
特徴量パッケージ
"""

from __future__ import annotations

from types import ModuleType

from .registry import FeatureRegistry


def get_feature_manager() -> FeatureRegistry:
    """Get the feature manager instance"""
    return FeatureRegistry()


# Lazy import feature modules - only import when needed
_FEATURE_MODULES: dict[str, ModuleType | None] = {
    "momentum": None,
    "scalping": None,
    "time": None,
    "trend": None,
    "utils": None,
    "volatility": None,
    "volume": None,
}


def _ensure_module_loaded(module_name: str) -> None:
    """Ensure a feature module is loaded"""
    if _FEATURE_MODULES[module_name] is None:
        if module_name == "momentum":
            from . import momentum

            _FEATURE_MODULES[module_name] = momentum
        elif module_name == "scalping":
            from . import scalping

            _FEATURE_MODULES[module_name] = scalping
        elif module_name == "time":
            from . import time

            _FEATURE_MODULES[module_name] = time
        elif module_name == "trend":
            from . import trend

            _FEATURE_MODULES[module_name] = trend
        elif module_name == "utils":
            from . import utils

            _FEATURE_MODULES[module_name] = utils
        elif module_name == "volatility":
            from . import volatility

            _FEATURE_MODULES[module_name] = volatility
        elif module_name == "volume":
            from . import volume

            _FEATURE_MODULES[module_name] = volume


__all__ = ["FeatureRegistry", "get_feature_manager"]
