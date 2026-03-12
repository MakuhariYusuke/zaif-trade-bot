"""Compatibility shims to support older test imports under ztb.core

This package delegates to the newer module layout under ztb.* while
keeping backwards compatibility for tests that import under ztb.core.
"""

from __future__ import annotations

__all__ = ["preprocessing"]

from . import preprocessing  # re-export
