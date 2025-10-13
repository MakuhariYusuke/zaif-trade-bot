#!/usr/bin/env python3
"""
Cross-platform cleanup utility used by the Makefile.

This script removes common build and cache artifacts without relying on
platform-specific shell utilities, ensuring the clean target works on
Windows, macOS, and Linux alike.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory names that should be deleted recursively.
DIR_PATTERNS: List[str] = [
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
]

# File name patterns to delete.
FILE_PATTERNS: List[str] = [
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.tmp",
]

# Glob patterns for temporary directories.
TEMP_DIR_PATTERNS: List[str] = [
    "tmp*",
    "tmp-*",
    ".tmp-*",
]


def _remove_path(path: Path, *, dry_run: bool) -> None:
    """Delete a path from disk, logging the action."""
    if not path.exists():
        return

    print(f"Removing {path}")
    if dry_run:
        return

    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _iter_paths(patterns: Iterable[str]) -> Iterable[Path]:
    for pattern in patterns:
        yield from PROJECT_ROOT.rglob(pattern)


def clean(*, dry_run: bool) -> None:
    """Remove cached artifacts and temporary files."""
    # Remove temporary directories first.
    for temp_dir in _iter_paths(TEMP_DIR_PATTERNS):
        if temp_dir.is_dir():
            _remove_path(temp_dir, dry_run=dry_run)

    # Remove cache-style directories.
    for cache_dir in _iter_paths(DIR_PATTERNS):
        if cache_dir.is_dir():
            _remove_path(cache_dir, dry_run=dry_run)

    # Remove orphaned compiled files.
    for file_path in _iter_paths(FILE_PATTERNS):
        if file_path.is_file():
            _remove_path(file_path, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove build and cache artifacts.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )
    args = parser.parse_args()

    clean(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
