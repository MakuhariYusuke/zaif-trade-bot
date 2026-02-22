"""Minimal type aliases used by tests that import SB3 types."""
from typing import Any, Callable

# GymEnv is usually a Union[gym.Env, VecEnv], tests only need a placeholder
GymEnv = Any

# Schedule is a callable taking a float and returning a float
Schedule = Callable[[float], float]

# TensorDict is used in some codebases as a mapping-like tensor container; tests only
# need a placeholder type.
TensorDict = dict

__all__ = ["GymEnv", "Schedule", "TensorDict"]

