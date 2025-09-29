"""
Naming shim for backward compatibility.

This module provides backward compatibility for renamed functions, classes,
and modules. It should be imported early in the application lifecycle to
ensure old names are available.

Usage:
    from ztb.utils.naming_shim import apply_shims
    apply_shims()

Or import specific shims:
    from ztb.utils.naming_shim import TradeBot as OldTradeBot
"""

import sys
import warnings
from typing import Any

# Mapping of old names to new names
# Format: "old_module.old_name": "new_module.new_name"
NAME_SHIMS = {
    # Example shims (add actual renamed items here)
    # "ztb.core.trading.TradeEngine": "ztb.core.execution.TradeExecutor",
    # "ztb.utils.helpers.format_price": "ztb.utils.formatting.format_currency",
}

# Classes/functions that have been renamed
CLASS_SHIMS = {
    # "OldClassName": NewClass,
}

FUNCTION_SHIMS = {
    # "old_function": new_function,
}


def apply_shims() -> None:
    """
    Apply all naming shims to make old names available.

    This function should be called early in the application startup
    to ensure backward compatibility.
    """
    for old_path, new_path in NAME_SHIMS.items():
        _apply_module_shim(old_path, new_path)

    # Apply class and function shims
    for old_name, new_obj in {**CLASS_SHIMS, **FUNCTION_SHIMS}.items():
        _apply_object_shim(old_name, new_obj)


def _apply_module_shim(old_path: str, new_path: str) -> None:
    """
    Apply a module-level shim.

    Args:
        old_path: The old module path (e.g., "ztb.core.old_module")
        new_path: The new module path (e.g., "ztb.core.new_module")
    """
    try:
        old_module_parts = old_path.split(".")
        new_module_parts = new_path.split(".")

        if len(old_module_parts) != len(new_module_parts):
            warnings.warn(f"Cannot shim {old_path} -> {new_path}: different depths")
            return

        # Import the new module
        new_module = __import__(new_path, fromlist=[new_module_parts[-1]])

        # Make the old module path point to the new module
        _inject_module(old_module_parts, new_module)

        warnings.warn(
            f"Module {old_path} has been renamed to {new_path}. "
            f"Please update your imports.",
            DeprecationWarning,
            stacklevel=2,
        )

    except ImportError as e:
        warnings.warn(f"Failed to apply shim {old_path} -> {new_path}: {e}")


def _apply_object_shim(old_name: str, new_obj: Any) -> None:
    """
    Apply an object-level shim (class or function).

    Args:
        old_name: The old name
        new_obj: The new object
    """
    # Inject into the global namespace of the calling module
    frame = sys._getframe(2)
    caller_globals = frame.f_globals
    caller_globals[old_name] = new_obj

    warnings.warn(
        f"{old_name} has been renamed. Please use {new_obj.__name__} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def _inject_module(path_parts: list, module: Any) -> None:
    """
    Inject a module into the Python module system.

    Args:
        path_parts: List of module path components
        module: The module to inject
    """
    current = sys.modules
    for part in path_parts[:-1]:
        if part not in current:
            current[part] = type(sys)("module")
        current = getattr(current[part], "__dict__", {})

    current[path_parts[-1]] = module


# Convenience imports for common shims
# Add specific shims here as they are needed

# Example:
# from ztb.core.execution import TradeExecutor as TradeEngine
# TradeEngine = TradeExecutor  # For backward compatibility
