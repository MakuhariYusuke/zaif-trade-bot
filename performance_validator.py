"""Minimal PerformanceValidator stub used in some integration tests."""
from dataclasses import dataclass


@dataclass
class PerformanceValidator:
    def validate(self, *args, **kwargs):
        return True

__all__ = ["PerformanceValidator"]
