"""
Small git helper utilities with defensive defaults.

These helpers avoid hard failures when git is unavailable and disable
git-lfs filters to keep metadata collection stable in minimal environments.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_GIT_READONLY_FLAGS = ["--no-optional-locks"]
_GIT_LFS_BYPASS = [
    "-c",
    "filter.lfs.process=",
    "-c",
    "filter.lfs.required=false",
    "-c",
    "filter.lfs.clean=cat",
    "-c",
    "filter.lfs.smudge=cat",
]


def _run_git(
    args: list[str], cwd: Path | None = None, timeout: int = 5
) -> subprocess.CompletedProcess[str] | None:
    try:
        env = dict(os.environ)
        # Read-only metadata commands should not contend on index.lock.
        env.setdefault("GIT_OPTIONAL_LOCKS", "0")
        # 169# subprocess popup 抑制
        extra_kwargs: dict[str, int] = {}
        if sys.platform == "win32":
            extra_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        return subprocess.run(
            ["git", *_GIT_READONLY_FLAGS, *_GIT_LFS_BYPASS, *args],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            env=env,
            **extra_kwargs,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def get_git_output(args: list[str], cwd: Path | None = None, timeout: int = 5) -> str | None:
    result = _run_git(args, cwd=cwd, timeout=timeout)
    if result is None or result.returncode != 0:
        return None
    return result.stdout.strip()


def get_git_sha(cwd: Path | None = None, timeout: int = 5) -> str:
    return get_git_output(["rev-parse", "HEAD"], cwd=cwd, timeout=timeout) or "unknown"


def get_git_branch(cwd: Path | None = None, timeout: int = 5) -> str:
    return (
        get_git_output(["branch", "--show-current"], cwd=cwd, timeout=timeout)
        or "unknown"
    )


def get_git_remote_url(cwd: Path | None = None, timeout: int = 5) -> str:
    return (
        get_git_output(["remote", "get-url", "origin"], cwd=cwd, timeout=timeout)
        or "unknown"
    )


def get_git_status_lines(
    cwd: Path | None = None,
    timeout: int = 8,
    *,
    include_untracked: bool = False,
    max_lines: int | None = 200,
) -> list[str]:
    status_args = ["status", "--porcelain=v1"]
    if not include_untracked:
        status_args.extend(["--untracked-files=no"])
    output = get_git_output(status_args, cwd=cwd, timeout=timeout)
    if not output:
        return []
    lines = [line for line in output.splitlines() if line.strip()]
    if isinstance(max_lines, int) and max_lines > 0:
        return lines[:max_lines]
    return lines


def get_git_dirty_status(
    cwd: Path | None = None, timeout: int = 8, *, include_untracked: bool = False
) -> bool:
    return len(
        get_git_status_lines(
            cwd=cwd,
            timeout=timeout,
            include_untracked=include_untracked,
            max_lines=1,
        )
    ) > 0


def get_git_status_summary(
    cwd: Path | None = None,
    max_chars: int = 200,
    timeout: int = 8,
    *,
    include_untracked: bool = False,
    max_lines: int = 50,
) -> str:
    summary = "\n".join(
        get_git_status_lines(
            cwd=cwd,
            timeout=timeout,
            include_untracked=include_untracked,
            max_lines=max_lines,
        )
    )
    if len(summary) > max_chars:
        return summary[:max_chars]
    return summary
