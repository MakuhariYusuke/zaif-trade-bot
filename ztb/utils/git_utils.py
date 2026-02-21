"""
Small git helper utilities with defensive defaults.

These helpers avoid hard failures when git is unavailable and disable
git-lfs filters to keep metadata collection stable in minimal environments.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

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
        return subprocess.run(
            ["git", *_GIT_LFS_BYPASS, *args],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
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


def get_git_status_lines(cwd: Path | None = None, timeout: int = 8) -> list[str]:
    output = get_git_output(["status", "--porcelain"], cwd=cwd, timeout=timeout)
    if not output:
        return []
    return [line for line in output.splitlines() if line.strip()]


def get_git_dirty_status(cwd: Path | None = None, timeout: int = 8) -> bool:
    return len(get_git_status_lines(cwd=cwd, timeout=timeout)) > 0


def get_git_status_summary(
    cwd: Path | None = None, max_chars: int = 200, timeout: int = 8
) -> str:
    summary = "\n".join(get_git_status_lines(cwd=cwd, timeout=timeout))
    if len(summary) > max_chars:
        return summary[:max_chars]
    return summary

