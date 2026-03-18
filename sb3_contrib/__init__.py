"""Lightweight sb3_contrib compatibility package."""
from __future__ import annotations

from stable_baselines3 import PPO


class MaskablePPO(PPO):
    pass


__all__ = ["MaskablePPO"]
