"""Top-level pytest configuration helpers.

We add a global pytest_ignore_collect hook here so files under `scripts/`
and `archived/` are never collected, even if pytest's discovery walks the
project root directly (which can happen on some platforms/pyproject setups).
"""
from __future__ import annotations

from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config):  # type: ignore[override]
    np = str(collection_path).replace("\\", "/")
    if np.startswith("archived/") or "/archived/" in np:
        return True
    if np.startswith("scripts/") or "/scripts/" in np:
        return True
