"""Compatibility shims for older import paths under `ztb.optimization`.

These lightweight modules exist to keep tests running when projects import
`ztb.optimization.*`. They re-export or provide minimal classes backed by
existing implementations where available.
"""

from . import model_compression  # re-export module
from . import system_optimizer
from . import reward_function_optimizer

__all__ = ["model_compression", "system_optimizer", "reward_function_optimizer"]
