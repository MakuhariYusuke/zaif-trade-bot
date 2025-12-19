"""Top-level pytest configuration helpers.

We add a global pytest_ignore_collect hook here so files under `scripts/`
and `archived/` are never collected, even if pytest's discovery walks the
project root directly (which can happen on some platforms/pyproject setups).
"""
from __future__ import annotations

def pytest_ignore_collect(path, config):
    p = str(path)
    np = p.replace("\\", "/")
    if np.startswith("archived/") or "/archived/" in np:
        return True
    if np.startswith("scripts/") or "/scripts/" in np:
        return True
