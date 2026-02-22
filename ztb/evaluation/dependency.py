"""Minimal compatibility stubs for evaluation dependency utilities used by legacy tests."""
from typing import Any, Dict


class DependencyGraph:
    def __init__(self, *args, **kwargs):
        self.graph = {}


class FeatureDependencyManager:
    def __init__(self, *args, **kwargs):
        pass


__all__ = ["DependencyGraph", "FeatureDependencyManager"]
