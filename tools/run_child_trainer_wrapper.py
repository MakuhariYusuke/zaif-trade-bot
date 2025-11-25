#!/usr/bin/env python3
"""
Child-process wrapper for running Unified Trainer with detailed import diagnostics.

This script is intended to isolate Windows DLL initialization issues (e.g., c10.dll)
by:
  - Capturing environment/sys.path snapshots for the child interpreter
  - Attempting torch/ztb imports with rich error logs
  - Probing torch DLL locations (torch/lib, c10.dll via ctypes)
  - Running the actual trainer only after imports succeed
  - Writing JSONL diagnostics to logs/child_wrapper_debug.jsonl
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_PATH = Path("logs") / "child_wrapper_debug.jsonl"


def _truncate(value: Optional[str], max_len: int = 800) -> Optional[str]:
    """Shorten long strings to keep log lines readable."""
    if value is None:
        return None
    if len(value) <= max_len:
        return value
    return f"{value[:max_len]}...(truncated {len(value) - max_len} chars)"


def log_event(event: str, **data: Any) -> None:
    """Write a JSONL diagnostic record."""
    record = {
        "event": event,
        "time": datetime.now().isoformat(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        **data,
    }
    try:
        LOG_PATH.parent.mkdir(exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Avoid breaking the child even if logging fails
        pass


def snapshot_environment() -> Dict[str, Any]:
    """Collect a concise environment/sys.path snapshot for troubleshooting."""
    env_keys = [
        "PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CUDA_VISIBLE_DEVICES",
        "TORCH_HOME",
        "TORCH_USE_CUDA",
        "KMP_DUPLICATE_LIB_OK",
        "OMP_NUM_THREADS",
    ]
    env = {k: _truncate(os.environ.get(k)) for k in env_keys if os.environ.get(k)}
    return {
        "cwd": str(Path.cwd()),
        "executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "env": env,
        "sys_path": sys.path,
    }


def ensure_torch_dll_search_path() -> Dict[str, Any]:
    """
    On Windows, explicitly add torch/lib to DLL search paths to avoid c10.dll issues.

    Returns a small summary of candidate/added paths for logging.
    """
    summary: Dict[str, Any] = {
        "candidates": [],
        "added_to_dll_dir": [],
        "added_to_path": [],
    }
    if os.name != "nt":
        return summary

    try:
        import site

        site_roots: List[Path] = []
        site_roots.extend(Path(p) for p in site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site:
            site_roots.append(Path(user_site))

        for root in site_roots:
            candidate = root / "torch" / "lib"
            if candidate.exists():
                summary["candidates"].append(str(candidate))
                try:
                    if hasattr(os, "add_dll_directory"):
                        os.add_dll_directory(str(candidate))
                        summary["added_to_dll_dir"].append(str(candidate))
                except Exception as e:
                    summary.setdefault("dll_dir_errors", []).append(str(e))

                try:
                    path_val = os.environ.get("PATH", "")
                    if str(candidate) not in path_val.split(os.pathsep):
                        os.environ["PATH"] = str(candidate) + os.pathsep + path_val
                        summary["added_to_path"].append(str(candidate))
                except Exception as e:
                    summary.setdefault("path_errors", []).append(str(e))
    except Exception as e:
        summary["error"] = str(e)

    return summary


def attempt_torch_import() -> Dict[str, Any]:
    """Try importing torch and probing common DLL locations."""
    result: Dict[str, Any] = {}
    try:
        torch_mod = importlib.import_module("torch")
        result["status"] = "ok"
        result["version"] = getattr(torch_mod, "__version__", None)
        result["file"] = str(getattr(torch_mod, "__file__", ""))
        result["cuda_available"] = bool(
            getattr(torch_mod.cuda, "is_available", lambda: False)()
        )
        result["cuda_version"] = getattr(
            getattr(torch_mod, "version", None), "cuda", None
        )

        torch_dir = Path(torch_mod.__file__).resolve().parent
        lib_dir = torch_dir / "lib"
        result["torch_dir"] = str(torch_dir)
        result["lib_dir_exists"] = lib_dir.exists()

        c10_path = lib_dir / "c10.dll"
        result["c10_path"] = str(c10_path)
        if c10_path.exists():
            try:
                import ctypes

                ctypes.CDLL(str(c10_path))
                result["c10_load"] = "ok"
            except OSError as e:
                result["c10_load"] = "error"
                result["c10_error"] = str(e)
            except Exception as e:
                result["c10_load"] = "error"
                result["c10_error"] = f"Unexpected: {e}"
        else:
            result["c10_load"] = "missing"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    log_event("import_torch", **result)
    return result


def attempt_import(module_name: str) -> Dict[str, Any]:
    """Generic import helper for diagnostics."""
    result: Dict[str, Any] = {"module": module_name}
    try:
        mod = importlib.import_module(module_name)
        result["status"] = "ok"
        result["file"] = str(getattr(mod, "__file__", ""))
    except Exception as exc:
        result["status"] = "error"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    log_event("import_module", **result)
    return result


def run_training(args: argparse.Namespace) -> int:
    """Configure seeds/env and invoke the unified trainer."""
    random.seed(args.seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(args.seed)
    except Exception:
        # Still continue even if numpy seeding fails; log for visibility
        log_event("numpy_seed_error", error=traceback.format_exc())

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    # Force CPU-only behavior by default to avoid CUDA DLL probing on Windows CPU installs
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TORCH_USE_CUDA", "0")

    argv = [
        "main.py",
        "--config",
        args.config,
        "-s",
        str(args.timesteps),
        "-l",
        args.log_level,
    ]
    if args.fast_mode:
        argv.append("--fast-mode")
    if args.ab_tag:
        argv.extend(["--ab-tag", args.ab_tag])
    if args.data_rows_limit is not None:
        argv.extend(["--data-rows-limit", str(args.data_rows_limit)])

    sys.argv = argv
    log_event("training_start", argv=argv)
    print(f"[child-wrapper] launching trainer with argv: {argv}")

    try:
        from ztb.training.unified_trainer import main as trainer_main

        trainer_main.main()
        log_event("training_complete", returncode=0)
        return 0
    except SystemExit as e:
        code = (
            int(e.code)
            if isinstance(e.code, (int, str)) and str(e.code).isdigit()
            else 1
        )
        log_event("training_exit", returncode=code)
        return code
    except Exception:
        log_event("training_error", traceback=traceback.format_exc())
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Child wrapper for AB test training")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--seed", type=int, default=0, help="Seed for RNG setup")
    parser.add_argument(
        "--timesteps", type=int, default=2000, help="Total timesteps override"
    )
    parser.add_argument("--fast-mode", action="store_true", help="Enable fast-mode")
    parser.add_argument(
        "--log-level", default="WARNING", help="Log level passed to trainer"
    )
    parser.add_argument("--ab-tag", type=str, help="AB tag attached to reports")
    parser.add_argument(
        "--data-rows-limit",
        type=int,
        help="Limit dataset rows (forwarded to trainer)",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Run import diagnostics and exit without training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_event("child_start", args=vars(args))
    print(
        f"[child-wrapper] PID {os.getpid()} starting diagnostics for config {args.config}"
    )

    log_event("environment_snapshot", snapshot=snapshot_environment())
    dll_summary = ensure_torch_dll_search_path()
    if dll_summary["candidates"]:
        log_event("dll_search_path", **dll_summary)

    torch_info = attempt_torch_import()
    ztb_info = attempt_import("ztb")
    trainer_info = attempt_import("ztb.training.unified_trainer.trainer")

    diagnostics_ok = (
        torch_info.get("status") == "ok"
        and ztb_info.get("status") == "ok"
        and trainer_info.get("status") == "ok"
    )

    if args.diagnostics_only:
        print(f"[child-wrapper] diagnostics-only requested; status ok={diagnostics_ok}")
        sys.exit(0 if diagnostics_ok else 1)

    if not diagnostics_ok:
        print(
            "[child-wrapper] diagnostics failed; skipping training. Check logs/child_wrapper_debug.jsonl."
        )
        sys.exit(1)

    exit_code = run_training(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
