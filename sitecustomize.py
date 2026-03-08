"""Project-level import fixes executed very early on interpreter startup.

This module ensures the file-backed ``stable_baselines3`` package shipped with
the repository is preferred for imports during pytest collection and normal
development, replacing in-memory stubs that some test helpers inject.

It also guarantees expected submodules (common.callbacks, vec_env, base_class,
policies) are importable and that the package object exposes a ``__path__`` so
relative imports resolve properly.

History
-------
This file previously contained ~390 lines with redundant ``_ensure_sb3_compat``
(in-memory SB3 stub builder) and commented-out websockets/importlib monkey-patches.
Since ``stable_baselines3/`` exists on disk as a complete stub package, those
code-paths were dead.  Simplified in 336# to the essential ``_prefer_local_package``
plus a lightweight ``_replace_stub_with_filebacked`` safety-net.
"""

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. Prefer the file-backed stable_baselines3 package under PROJECT_ROOT
# ---------------------------------------------------------------------------
def _prefer_local_package() -> bool:
    """Import file-backed ``stable_baselines3`` and pre-load key sub-modules."""
    try:
        spec = importlib.util.find_spec("stable_baselines3")
        if spec is None or not spec.origin:
            return False
        origin = Path(spec.origin)
        if PROJECT_ROOT not in origin.parents:
            return False

        mod = importlib.import_module("stable_baselines3")
        pkg_dir = str(origin.parent)
        try:
            if not hasattr(mod, "__path__"):
                mod.__path__ = [pkg_dir]
            elif pkg_dir not in mod.__path__:
                mod.__path__.insert(0, pkg_dir)
        except Exception:
            pass

        # Eagerly import common sub-modules so they are in sys.modules
        for sub in (
            "common.callbacks",
            "common.vec_env",
            "common.base_class",
            "common.policies",
        ):
            fullname = f"stable_baselines3.{sub}"
            try:
                importlib.import_module(fullname)
            except Exception:
                if fullname not in sys.modules:
                    m = importlib.util.module_from_spec(
                        importlib.util.spec_from_loader(fullname, loader=None)
                    )
                    try:
                        subpath = Path(*fullname.split(".")[-2:])
                        m.__file__ = str(
                            (Path(PROJECT_ROOT) / "stable_baselines3" / subpath).with_suffix(".py")
                        )
                    except Exception:
                        pass
                    sys.modules[fullname] = m
        return True
    except Exception:
        return False


_prefer_local_package()


# ---------------------------------------------------------------------------
# 2. Safety-net: replace lingering in-memory stubs with file-backed modules
# ---------------------------------------------------------------------------
def _replace_stub_with_filebacked(fullname: str, candidate_path: Path) -> None:
    """If *fullname* is in ``sys.modules`` but lacks ``__file__`` / ``__spec__``,
    load the file-backed module from *candidate_path* instead."""
    try:
        m = sys.modules.get(fullname)
        if m is None:
            return
        if getattr(m, "__file__", None) and getattr(m, "__spec__", None):
            return  # already file-backed
        if not candidate_path.exists():
            return
        spec_fb = importlib.util.spec_from_file_location(fullname, str(candidate_path))
        if spec_fb is None:
            return
        real_mod = importlib.util.module_from_spec(spec_fb)
        if candidate_path.name == "__init__.py":
            real_mod.__path__ = [str(candidate_path.parent)]
        try:
            spec_fb.loader.exec_module(real_mod)  # type: ignore[union-attr]
            sys.modules[fullname] = real_mod
        except Exception:
            try:
                m.__file__ = str(candidate_path)
                m.__spec__ = importlib.util.spec_from_loader(fullname, loader=None)
            except Exception:
                pass
    except Exception:
        pass


try:
    _replace_stub_with_filebacked(
        "stable_baselines3",
        Path(PROJECT_ROOT) / "stable_baselines3" / "__init__.py",
    )
    _replace_stub_with_filebacked(
        "stable_baselines3.common.callbacks",
        Path(PROJECT_ROOT) / "stable_baselines3" / "common" / "callbacks.py",
    )
except Exception:
    pass
