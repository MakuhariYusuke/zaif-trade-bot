"""
ZAIF Trade Bot - Advanced Trading System with Reinforcement Learning

This package provides a comprehensive trading bot framework featuring:
- SAC (Soft Actor-Critic) reinforcement learning algorithms
- Advanced backtesting and analysis tools
- Risk management and portfolio optimization
- Real-time trading execution
- Comprehensive monitoring and logging

Main Components:
- trading: Core trading logic and strategies
- analysis: Backtesting and performance analysis tools
- evaluation: Trading performance evaluation
- config: Configuration management system
- utils: Utility functions and helpers
- data: Data processing and augmentation tools
"""

from typing import TYPE_CHECKING

from .utils.torch_utils import ensure_torch_dll_search_path

__version__ = "4.2.0"
__author__ = "MakuhariYusuke"
__description__ = "Advanced trading bot with reinforcement learning"

# ---------------------------------------------------------------------------
# Windows DLL guard: torch import is optionally performed to avoid importing
# into incompatible NumPy/Torch environments during package import. We lazily
# import torch only if the NumPy major version indicates compatibility.
# ---------------------------------------------------------------------------
_TORCH_IMPORT_ERROR = None
ensure_torch_dll_search_path()
try:
    # Only import torch automatically when NumPy major version < 2, otherwise
    # skip automatic torch import to prevent ABI incompatibilities that may
    # manifest as access violations in tests or during import time.
    import numpy as _np

    np_major = (
        int(_np.__version__.split(".", 1)[0]) if hasattr(_np, "__version__") else 0
    )
except Exception:
    np_major = 0

if np_major < 2:
    try:  # pragma: no cover - platform/env dependent behavior
        import torch  # type: ignore  # noqa: F401
    except Exception as _torch_exc:  # pragma: no cover - diagnostics only
        _TORCH_IMPORT_ERROR = _torch_exc
else:
    # Do not auto-import torch; tests or code that require it should import
    # torch explicitly; this avoids causing segfaults in incompatible envs.
    _TORCH_IMPORT_ERROR = None

if TYPE_CHECKING:
    from .config.schema import GlobalConfig

# Import main components for easy access
# from .analysis import BacktestAnalyzer  # Temporarily disabled
from .config import ConfigManager

# Avoid importing heavy submodules at package import time; use lazy attribute access
__LAZY_MODULE_ATTRIBUTES__ = {
    "BTCDataAugmentor": ("ztb.data", "BTCDataAugmentor"),
    "BTCBiasDetector": ("ztb.data", "BTCBiasDetector"),
}

# Define public API
__all__ = [
    # Core components
    "ConfigManager",
    # "BacktestAnalyzer",  # Temporarily disabled
    "BTCDataAugmentor",
    "BTCBiasDetector",
    # Metadata
    "__version__",
    "__author__",
    "__description__",
]


def __getattr__(name: str) -> object:
    """Lazy import heavy submodules when accessed on the package.

    Implemented per PEP 562 to avoid importing heavy ML modules at package import time.
    """
    if name in __LAZY_MODULE_ATTRIBUTES__:
        module_name, attr_name = __LAZY_MODULE_ATTRIBUTES__[name]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__LAZY_MODULE_ATTRIBUTES__.keys()))


def get_version() -> str:
    """Get the current version of the package."""
    return __version__


def get_config() -> "GlobalConfig":
    """Get the current configuration."""
    return ConfigManager.get_instance().get_config()
