"""Compatibility shim for old import path `ztb.training.custom_ppo`.
Re-export CustomPPO from `ztb.training.models.custom_ppo`.
"""
from ztb.training.models.custom_ppo import CustomPPO  # noqa: F401

__all__ = ["CustomPPO"]
