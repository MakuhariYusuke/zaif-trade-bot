#!/usr/bin/env python3
"""Apply safe local git performance settings for large repositories.

This script only writes local repository config (`.git/config`).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

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

        print("Applied local git performance settings:")
        for key, _ in settings:
            print(f"  {key} = {_get_local_config(key)}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
