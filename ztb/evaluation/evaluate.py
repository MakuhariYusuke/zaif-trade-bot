"""Compatibility shim for `ztb.evaluation.evaluate` used by tests and scripts.

Provides a minimal `PPO` object with a `load` method so tests that patch
`ztb.evaluation.evaluate.PPO.load` can do so without requiring the full
stable-baselines3 package.
"""
from __future__ import annotations

from typing import Any

try:
    from stable_baselines3 import PPO as _SB3_PPO  # type: ignore
except Exception:
    _SB3_PPO = None

class _PPOShim:
    @staticmethod
    def load(path: str) -> Any:
        """Load a model from disk.

        If a real SB3 PPO with `load` exists, delegate to it; otherwise return a
        lightweight MagicMock so tests can inspect calls.
        """
        if _SB3_PPO is not None and hasattr(_SB3_PPO, "load"):
            return _SB3_PPO.load(path)

        # Minimal fallback used in unit tests
        from unittest.mock import MagicMock

        m = MagicMock()
        m._loaded_path = path
        return m

PPO = _PPOShim

__all__ = ["PPO"]
