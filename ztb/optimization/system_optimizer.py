"""Minimal system optimizer shim for tests."""
from typing import Any, Dict


class SystemOptimizer:
    """Stubbed optimizer to allow imports and basic interactions in tests."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def optimize(self) -> Dict[str, Any]:
        """Return a trivial result dict."""
        return {"status": "ok", "config": self.config}


class MemoryOptimizer(SystemOptimizer):
    """Simple memory optimizer stub used in tests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def optimize(self) -> Dict[str, Any]:
        return {"status": "ok", "memory_optimized": True}


class PerformanceOptimizer(SystemOptimizer):
    def optimize(self) -> Dict[str, Any]:
        return {"status": "ok", "performance_optimized": True}


__all__ = ["SystemOptimizer"]
