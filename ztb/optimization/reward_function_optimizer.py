"""Compatibility shim for reward_function_optimizer under `ztb.optimization`.

Re-export the core optimizer class from the training module so older imports
continue to work.
"""
from ztb.training.reward_function_optimizer.reward_function_optimizer import (
    RewardFunctionOptimizer,
)

__all__ = ["RewardFunctionOptimizer"]
