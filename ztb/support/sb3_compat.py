"""Utilities to make stable_baselines3 imports deterministic for tests.

This module ensures that the `stable_baselines3` module object present in
`sys.modules` exposes the common algorithm symbols (PPO, SAC, etc.) and that
the `stable_baselines3.common.callbacks` module exposes `CallbackList` and
`BaseCallback`. Importing this early in test collection reduces import-time
flakiness caused by lightweight stubs.
"""
from __future__ import annotations

import sys
from types import ModuleType

def ensure_sb3_compat():
    try:
        sb3 = sys.modules.get("stable_baselines3")
        # If missing, try to import the real package first; only fall back to stub
        if sb3 is None:
            try:
                import stable_baselines3 as sb3  # type: ignore[no-redef]
            except ImportError:
                sb3 = ModuleType("stable_baselines3")
                sb3.SAC = type("SAC", (), {"learn": lambda self, *a, **k: self})
                sb3.PPO = type("PPO", (), {"learn": lambda self, *a, **k: self})
                sys.modules["stable_baselines3"] = sb3
        else:
            for algo in ("SAC", "PPO", "A2C", "DQN", "TD3"):
                if not hasattr(sb3, algo):
                    setattr(sb3, algo, type(algo, (), {"learn": lambda self, *a, **k: self}))

        # Ensure callbacks module exposes required symbols
        cbm = sys.modules.get("stable_baselines3.common.callbacks")
        if cbm is None:
            cbm = ModuleType("stable_baselines3.common.callbacks")
            cbm.BaseCallback = type("BaseCallback", (), {"n_calls": 0})
            cbm.CallbackList = list
            sys.modules["stable_baselines3.common.callbacks"] = cbm
        else:
            if not hasattr(cbm, "CallbackList"):
                cbm.CallbackList = list
            if not hasattr(cbm, "BaseCallback"):
                cbm.BaseCallback = type("BaseCallback", (), {"n_calls": 0})
    except Exception:
        # Be conservative: we don't want fixes here to crash collection
        pass

__all__ = ["ensure_sb3_compat"]
