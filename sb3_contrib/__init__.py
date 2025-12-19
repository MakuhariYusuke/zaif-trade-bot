"""Lightweight in-repo shim for sb3_contrib used in tests.

This minimal shim provides `MaskablePPO` and a basic package
structure for `sb3_contrib.common.wrappers.ActionMasker` so tests can
import and patch them without requiring the full external package.
"""
from types import SimpleNamespace


class MaskablePPO:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def learn(self, total_timesteps: int, **kwargs):
        return self


__all__ = ["MaskablePPO"]
