"""Backward-compatible shim: re-export PPOConfig at ztb.training.ppo_trainer

This module exists to preserve older import paths used in tests and scripts.
"""

from ztb.training.config.ppo_config import PPOConfig

__all__ = ["PPOConfig"]
