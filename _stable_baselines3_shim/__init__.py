"""Local shim for stable_baselines3 to provide deterministic symbols used by tests.

This module proxies to the real installed `stable_baselines3` when available,
otherwise provides minimal dummy implementations for tests.
"""
import importlib
import types
import sys

# To avoid recursion when the package is imported, do not try to import
# `stable_baselines3` from within its own __init__.py. Instead, provide a
# deterministic, minimal shim that exposes the common algorithm symbols used
# by tests. If a real installation is present, the real package in site-packages
# will typically shadow this repo package; explicit attempts to import the
# real package here can cause confusing re-entry behavior.
_real = None




# Export either the real classes (if available) or the dummy ones.
# errors when lightweight test stubs are present earlier in sys.modules).
try:
    import importlib
    importlib.import_module("stable_baselines3.common.callbacks")
        cb.CheckpointCallback = BaseCallback
    try:
        real_cb = importlib.import_module("stable_baselines3.common.callbacks")
        if getattr(real_cb, "__file__", None):
            _sys.modules["stable_baselines3.common.callbacks"] = real_cb
    except Exception:
        pass
except Exception:
    pass
