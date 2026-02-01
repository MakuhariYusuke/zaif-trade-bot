"""Local shim for `stable_baselines3`.

This repository includes a minimal placeholder package at `stable_baselines3/`
to keep lightweight environments and type checking happy. For real training and
backtests we must prefer the installed site-packages `stable_baselines3` (SB3).

Behavior:
- If SB3 is installed, load and execute the *real* SB3 package into this module
  namespace (so `import stable_baselines3` behaves normally even when the repo
  root shadows site-packages on `sys.path`).
- If SB3 is not installed, provide tiny dummy algorithm classes and minimal
  submodule stubs to satisfy imports in minimal/test-only environments.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from importlib.machinery import PathFinder
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_THIS_FILE = Path(__file__).resolve()


def _is_project_root_path(entry: str) -> bool:
    if not entry:
        try:
            return Path.cwd().resolve() == _PROJECT_ROOT
        except Exception:
            return False
    try:
        return Path(entry).resolve() == _PROJECT_ROOT
    except Exception:
        return False


def _load_real_sb3_into_current_module() -> bool:
    search_paths = [p for p in sys.path if not _is_project_root_path(p)]
    spec = PathFinder.find_spec(__name__, search_paths)
    if spec is None or spec.origin is None or spec.loader is None:
        return False
    try:
        if Path(spec.origin).resolve() == _THIS_FILE:
            return False
    except Exception:
        if str(spec.origin) == str(_THIS_FILE):
            return False

    module = sys.modules.get(__name__)
    if module is None:
        return False

    module.__spec__ = spec
    module.__file__ = spec.origin
    module.__loader__ = spec.loader
    if spec.submodule_search_locations is not None:
        module.__path__ = list(spec.submodule_search_locations)

    try:
        spec.loader.exec_module(module)  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - defensive fallback
        # If loading the real SB3 package fails for any reason (broken install,
        # missing optional deps, etc.), fall back to the lightweight local
        # dummies below instead of raising during import-time. Tests and
        # minimal tooling should be able to import `stable_baselines3`.
        import warnings

        warnings.warn(
            "Failed to load real 'stable_baselines3' package; using local "
            "dummy shims. Original error: %r" % (exc,),
            RuntimeWarning,
        )
        return False

    # Ensure key exports exist at the package level (some SB3 installs may
    # not re-export algorithm classes at the top-level). Attempt to locate
    # common algorithm classes to preserve `from stable_baselines3 import SAC`.
    try:  # pragma: no cover - best-effort recovery
        if not hasattr(module, "SAC"):
            sac_mod = importlib.import_module("stable_baselines3.sac.sac")
            module.SAC = getattr(sac_mod, "SAC", getattr(sac_mod, "sac", None))
    except Exception:
        try:
            sac_pkg = importlib.import_module("stable_baselines3.sac")
            module.SAC = getattr(sac_pkg, "SAC", None)
        except Exception:
            # Give up silently; downstream code should handle missing classes
            # gracefully (e.g., import-time guarded or try/except around
            # `from stable_baselines3 import SAC`).
            pass

    return True


if not _load_real_sb3_into_current_module():
    class _DummyModel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def learn(self, *args, **kwargs):
            return self

    SAC = _DummyModel
    PPO = _DummyModel
    A2C = _DummyModel
    DQN = _DummyModel
    TD3 = _DummyModel

    __all__ = ["SAC", "PPO", "A2C", "DQN", "TD3"]

    # Minimal stubs for common submodules referenced throughout the codebase.
    common_pkg = types.ModuleType("stable_baselines3.common")
    common_pkg.__path__ = []  # mark as package-like
    sys.modules.setdefault("stable_baselines3.common", common_pkg)

    callbacks_mod = types.ModuleType("stable_baselines3.common.callbacks")
    callbacks_mod.BaseCallback = type("BaseCallback", (), {"n_calls": 0})
    callbacks_mod.CallbackList = list
    callbacks_mod.EvalCallback = callbacks_mod.BaseCallback
    callbacks_mod.CheckpointCallback = callbacks_mod.BaseCallback
    sys.modules.setdefault("stable_baselines3.common.callbacks", callbacks_mod)

    vec_env_mod = types.ModuleType("stable_baselines3.common.vec_env")
    vec_env_mod.DummyVecEnv = type("DummyVecEnv", (), {})
    vec_env_mod.VecFrameStack = type("VecFrameStack", (), {})
    vec_env_mod.VecNormalize = type("VecNormalize", (), {})
    sys.modules.setdefault("stable_baselines3.common.vec_env", vec_env_mod)

    monitor_mod = types.ModuleType("stable_baselines3.common.monitor")
    monitor_mod.Monitor = type("Monitor", (), {})
    sys.modules.setdefault("stable_baselines3.common.monitor", monitor_mod)

    save_util_mod = types.ModuleType("stable_baselines3.common.save_util")
    def _load_from_zip_file(*args, **kwargs):
        raise NotImplementedError(
            "Local stable_baselines3 shim does not implement `load_from_zip_file`."
        )
    save_util_mod.load_from_zip_file = _load_from_zip_file
    sys.modules.setdefault("stable_baselines3.common.save_util", save_util_mod)
