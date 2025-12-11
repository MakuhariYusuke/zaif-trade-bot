"""Torch import helpers with Windows DLL safeguards."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

try:  # pragma: no cover - depends on interpreter build
    import site as _site
except Exception:  # pragma: no cover - site can be missing in embedded builds
    _site = None  # type: ignore

_TORCH_STUB_FLAG = "_zaif_torch_stub"
_DLL_SETUP_DONE = False
_DLL_SUMMARY: Dict[str, Any] | None = None


def _candidate_site_roots() -> List[Path]:
    roots: List[Path] = []
    seen: set[str] = set()

    if _site is not None:
        if hasattr(_site, "getsitepackages"):
            try:
                for entry in _site.getsitepackages():
                    if entry and entry not in seen:
                        seen.add(entry)
                        roots.append(Path(entry))
            except Exception:
                pass
        if hasattr(_site, "getusersitepackages"):
            try:
                user_site = _site.getusersitepackages()
                if user_site and user_site not in seen:
                    seen.add(user_site)
                    roots.append(Path(user_site))
            except Exception:
                pass

    for entry in sys.path:
        if not entry or "site-packages" not in entry:
            continue
        if entry not in seen:
            seen.add(entry)
            roots.append(Path(entry))

    return roots


def ensure_torch_dll_search_path(force: bool = False) -> Dict[str, Any]:
    """Add torch/lib directories to PATH/DLL search path on Windows."""

    global _DLL_SETUP_DONE, _DLL_SUMMARY

    if os.name != "nt":
        summary = {"status": "non-windows"}
        _DLL_SETUP_DONE = True
        _DLL_SUMMARY = summary
        return summary

    if _DLL_SETUP_DONE and not force and _DLL_SUMMARY is not None:
        return _DLL_SUMMARY

    summary: Dict[str, Any] = {
        "candidates": [],
        "added_to_dll_dir": [],
        "added_to_path": [],
    }

    for root in _candidate_site_roots():
        candidate = root / "torch" / "lib"
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        summary["candidates"].append(candidate_str)

        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(candidate_str)
                summary["added_to_dll_dir"].append(candidate_str)
            except Exception as exc:  # pragma: no cover - OS specific
                summary.setdefault("dll_dir_errors", []).append(str(exc))

        try:
            path_val = os.environ.get("PATH", "")
            if candidate_str not in path_val.split(os.pathsep):
                os.environ["PATH"] = candidate_str + os.pathsep + path_val
                summary["added_to_path"].append(candidate_str)
        except Exception as exc:  # pragma: no cover - env specific
            summary.setdefault("path_errors", []).append(str(exc))

    summary["status"] = "ok" if summary["candidates"] else "missing"
    _DLL_SETUP_DONE = True
    _DLL_SUMMARY = summary
    return summary


def _install_torch_stub() -> None:
    if "torch" in sys.modules and getattr(
        sys.modules["torch"], _TORCH_STUB_FLAG, False
    ):
        return

    class _StubOptimizer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.param_groups = []

        def step(self) -> None:  # pragma: no cover - trivial
            return None

        def zero_grad(self) -> None:  # pragma: no cover - trivial
            return None

    stub = types.ModuleType("torch")
    stub.__dict__.update(
        {
            "__version__": "0.0-stub",
            _TORCH_STUB_FLAG: True,
            "device": lambda *_args, **_kwargs: "cpu",
            "cuda": types.SimpleNamespace(
                is_available=lambda: False, device_count=lambda: 0
            ),
            "nn": types.SimpleNamespace(Module=object),
            "optim": types.SimpleNamespace(
                SGD=_StubOptimizer, Optimizer=_StubOptimizer
            ),
        }
    )
    sys.modules["torch"] = stub


def is_torch_available() -> bool:
    ensure_torch_dll_search_path()
    try:
        torch_mod = importlib.import_module("torch")
    except Exception:
        return False
    return not getattr(torch_mod, _TORCH_STUB_FLAG, False)


def get_preferred_device() -> str:
    """Return 'cuda' if available else 'cpu'."""
    ensure_torch_dll_search_path()
    try:
        torch_mod = importlib.import_module("torch")
        if getattr(torch_mod, _TORCH_STUB_FLAG, False):
            return "cpu"
        if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def ensure_cpu_mode() -> None:
    """Force CPU-only operation where possible, install stub if torch missing."""

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    ensure_torch_dll_search_path()

    try:
        torch_mod = importlib.import_module("torch")
    except ModuleNotFoundError:
        _install_torch_stub()
        return
    except Exception:
        _install_torch_stub()
        return

    # Real torch imported successfully; nothing else to do because callers are expected
    # to move tensors/devices explicitly. The environment variable above prevents CUDA
    # from being auto-selected in most setups.
    return


# Apply DLL search path tweaks immediately when the module loads on Windows.
if os.name == "nt":  # pragma: no cover - platform specific
    try:
        ensure_torch_dll_search_path()
    except Exception:
        pass
