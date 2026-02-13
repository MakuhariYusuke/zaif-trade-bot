#!/usr/bin/env python3
"""
Lightweight helper to run quick validation of v448 emergency config and child-wrapper diagnostics
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
    Path(__file__).resolve().parent,
)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v448.validation.validate_v448_emergency import (
    load_config,
    validate_emergency_config,
)


def run_quick_config_check(config_path: Path):
    cfg = load_config(config_path)
    res = validate_emergency_config(cfg)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return res


def run_child_wrapper_diagnostics(config_path: Path):
    print("Running child wrapper diagnostics...")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "run_child_trainer_wrapper.py"),
        "--config",
        str(config_path),
        "--diagnostics-only",
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    cfg_path = PROJECT_ROOT / "configs" / "v448" / "sac_v448_emergency_fix.json"
    if not cfg_path.exists():
        cfg_path = PROJECT_ROOT / "config" / "v448" / "sac_v448_emergency_fix.json"
    if not cfg_path.exists():
        print("Config not found:", cfg_path)
        return 2

    res = run_quick_config_check(cfg_path)
    if res["issues"]:
        print("Critical issues found. Exiting.")
        return 1

    diag_ok = run_child_wrapper_diagnostics(cfg_path)
    if not diag_ok:
        print("Child wrapper diagnostics failed. Check logs/child_wrapper_debug.jsonl.")
        return 1

    print("Quick validation completed: Config OK, child wrapper diagnostics OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
