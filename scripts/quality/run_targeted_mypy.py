#!/usr/bin/env python3
"""Run mypy in a way that is usable for incremental local refactors.

The repository currently has a large amount of historical mypy debt, and
plain `mypy --config-file mypy.ini ...` often surfaces unrelated errors from
far outside the changed area. This wrapper keeps the strict config, but
filters diagnostics down to the requested targets so we can answer the
practical question: "Did this change introduce new type problems here?"
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
MYPY_CONFIG = REPO_ROOT / "mypy.ini"
CACHE_DIR = REPO_ROOT / "temp" / ".mypy_targeted_cache"

_DIAGNOSTIC_RE = re.compile(r"^(?P<path>.+?):\d+")


def _default_python() -> str:
    candidates = (
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _target_to_path(target: str) -> tuple[str | None, str | None]:
    candidate = Path(target)
    if candidate.suffix == ".py" or "/" in target or "\\" in target:
        path = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
        rel = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        module = rel[:-3].replace("/", ".")
        return rel, module
    return None, target


def _build_command(
    *,
    python_exe: str,
    targets: Sequence[str],
    follow_imports: str,
) -> tuple[list[str], set[str]]:
    command = [
        python_exe,
        "-m",
        "mypy",
        "--config-file",
        str(MYPY_CONFIG),
        "--cache-dir",
        str(CACHE_DIR),
        "--follow-imports",
        follow_imports,
        "--explicit-package-bases",
    ]
    target_paths: set[str] = set()
    for target in targets:
        path_target, module_target = _target_to_path(target)
        if path_target is not None:
            command.append(path_target)
            target_paths.add(path_target)
        elif module_target is not None:
            command.extend(["-m", module_target])
            target_paths.add(module_target.replace(".", "/") + ".py")
    return command, target_paths


def _normalize_for_match(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def _filter_output(raw_lines: Sequence[str], target_paths: set[str]) -> tuple[list[str], int]:
    normalized_targets = {_normalize_for_match(path) for path in target_paths}
    kept: list[str] = []
    suppressed = 0
    for line in raw_lines:
        match = _DIAGNOSTIC_RE.match(line)
        if not match:
            if line.startswith("Success:"):
                kept.append(line)
            continue
        raw_path = _normalize_for_match(match.group("path"))
        if any(
            raw_path == target or raw_path.endswith("/" + target)
            for target in normalized_targets
        ):
            kept.append(line)
        else:
            suppressed += 1
    return kept, suppressed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run mypy for specific targets while suppressing unrelated repo-wide baseline errors."
    )
    parser.add_argument("targets", nargs="+", help="Python file paths or module names to check.")
    parser.add_argument(
        "--mode",
        choices=("fast", "deep"),
        default="fast",
        help="fast=follow-imports=skip, deep=follow-imports=silent",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=120.0,
        help="Subprocess timeout in seconds.",
    )
    args = parser.parse_args()

    python_exe = _default_python()
    follow_imports = "skip" if args.mode == "fast" else "silent"
    command, target_paths = _build_command(
        python_exe=python_exe,
        targets=args.targets,
        follow_imports=follow_imports,
    )

    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=args.timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(
            f"targeted mypy timed out after {args.timeout_sec:.0f}s: {' '.join(args.targets)}",
            file=sys.stderr,
        )
        return 124

    raw_lines = [
        line
        for line in (result.stdout.splitlines() + result.stderr.splitlines())
        if line.strip()
    ]
    kept, suppressed = _filter_output(raw_lines, target_paths)

    if kept:
        print("\n".join(kept))
        if suppressed:
            print(
                f"\n[targeted-mypy] suppressed {suppressed} unrelated baseline diagnostics.",
                file=sys.stderr,
            )
        return result.returncode if result.returncode != 0 else 0

    if result.returncode != 0:
        print(
            f"[targeted-mypy] no diagnostics in requested targets; suppressed {suppressed} unrelated baseline diagnostics."
        )
        return 0

    print("Success: no targeted mypy issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
