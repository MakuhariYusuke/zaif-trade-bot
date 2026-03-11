"""Minimal type aliases for compatibility."""
from __future__ import annotations

from collections.abc import Callable

GymEnv = object
Schedule = Callable[[float], float]
TensorDict = dict[str, object]
