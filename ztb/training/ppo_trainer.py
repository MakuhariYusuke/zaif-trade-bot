"""Compatibility exports for legacy PPO training import paths.

This module keeps older imports working while pointing callers at the current
PPO trainer/config implementation.
"""

from ztb.training.config.ppo_config import PPOConfig
from ztb.training.core.ppo_trainer import PPOTrainer, PPOTrainingConfig
from ztb.training.custom_ppo import CustomPPO
from sb3_contrib import MaskablePPO

TrainingConfig = PPOTrainingConfig

__all__ = [
    "CustomPPO",
    "MaskablePPO",
    "PPOConfig",
    "PPOTrainer",
    "PPOTrainingConfig",
    "TrainingConfig",
]
