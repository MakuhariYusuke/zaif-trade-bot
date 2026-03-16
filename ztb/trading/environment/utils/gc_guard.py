"""Conditional GC helpers for heavy environment components."""

from __future__ import annotations

import gc


def maybe_collect_garbage(memory_threshold_percent: float = 85.0) -> bool:
    """Collect garbage only when system memory pressure is high.

    Returns True when a collection was triggered.
    """
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return False

    if psutil.virtual_memory().percent > memory_threshold_percent:
        gc.collect()
        return True
    return False
