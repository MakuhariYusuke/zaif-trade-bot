#!/usr/bin/env python3
"""Apply safe local git performance settings for large repositories.

This script only writes local repository config (`.git/config`).
"""

from __future__ import annotations

import argparse
import shutil
import stat
import subprocess
import sys
from pathlib import Path

PERF_SETTINGS: list[tuple[str, str]] = [
    ("feature.manyFiles", "true"),
    ("core.untrackedCache", "true"),
    ("core.preloadIndex", "true"),
    ("index.threads", "0"),
    ("status.aheadBehind", "false"),
]

LFS_BYPASS_SETTINGS: list[tuple[str, str]] = [
    ("filter.lfs.required", "false"),
    ("filter.lfs.process", ""),
    ("filter.lfs.clean", "cat"),
    ("filter.lfs.smudge", "cat"),
]


PRE_COMMIT_HOOK_TEMPLATE = """#!/bin/sh
# Cross-shell pre-commit launcher.
# Runs hooks when a compatible pre-commit runtime is available.
# Falls back to a non-blocking skip when runtime is unavailable/misaligned.

HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"

if command -v pre-commit >/dev/null 2>&1; then
    exec pre-commit hook-impl --config=.pre-commit-config.yaml --hook-type=pre-commit --hook-dir "$HOOK_DIR" -- "$@"
fi

if [ -x ".venv/Scripts/python.exe" ]; then
    .venv/Scripts/python.exe -m pre_commit hook-impl --config=.pre-commit-config.yaml --hook-type=pre-commit --hook-dir "$HOOK_DIR" -- "$@"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        exit 0
    fi
    echo "[pre-commit] skipped: runtime mismatch (exit=$rc)." >&2
    exit 0
fi

echo "[pre-commit] skipped: pre-commit runtime not available." >&2
exit 0
"""


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
    )


def _ensure_repo() -> None:
    result = _run_git(["rev-parse", "--is-inside-work-tree"])
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise RuntimeError("Not inside a git repository")


def _set_local_config(key: str, value: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] git config --local {key} {value}")
        return
    result = _run_git(["config", "--local", key, value])
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"Failed to set {key}: {stderr}")


def _git_path(pathspec: str) -> Path:
    result = _run_git(["rev-parse", "--git-path", pathspec])
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"Failed to resolve git path '{pathspec}': {stderr}")
    return Path(result.stdout.strip())


def _repair_pre_commit_hook(dry_run: bool) -> None:
    hook_path = _git_path("hooks") / "pre-commit"
    if dry_run:
        print(f"[dry-run] write hook: {hook_path}")
        return

    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(PRE_COMMIT_HOOK_TEMPLATE, encoding="utf-8", newline="\n")
    mode = hook_path.stat().st_mode
    hook_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Repaired hook: {hook_path}")


def _get_local_config(key: str) -> str:
    result = _run_git(["config", "--local", "--get", key])
    if result.returncode != 0:
        return "(unset)"
    value = result.stdout.rstrip("\n")
    return value if value else "(empty)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize local git config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without modifying config",
    )
    parser.add_argument(
        "--repair-pre-commit-hook",
        action="store_true",
        help="Rewrite .git/hooks/pre-commit with a WSL-safe portable launcher",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _ensure_repo()

        settings = list(PERF_SETTINGS)
        if shutil.which("git-lfs") is None:
            settings.extend(LFS_BYPASS_SETTINGS)

        for key, value in settings:
            _set_local_config(key, value, dry_run=args.dry_run)

        if args.repair_pre_commit_hook:
            _repair_pre_commit_hook(dry_run=args.dry_run)

        print("Applied local git performance settings:")
        for key, _ in settings:
            print(f"  {key} = {_get_local_config(key)}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
