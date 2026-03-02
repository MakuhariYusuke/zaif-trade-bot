"""Minimal PPOConfig stub for tests."""
from dataclasses import dataclass
from typing import Any

@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    policy_kwargs: dict[str, Any] = None

__all__ = ["PPOConfig"]
