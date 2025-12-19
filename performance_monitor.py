"""Minimal performance monitor stub used by integration tests."""
from dataclasses import dataclass


@dataclass
class PerformanceMonitor:
    name: str = "performance_monitor"

    def report(self, *args, **kwargs):
        return {}

__all__ = ["PerformanceMonitor"]
