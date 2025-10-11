"""
PPO (Proximal Policy Optimization) アルゴリズム実装。

このモジュールは既存のPPOTrainerをBaseRLAlgorithmインターフェースでラップし、
他のアルゴリズム（SAC, TD3等）と統一的に扱えるようにする。

Example:
    >>> from ztb.training.algorithms.ppo import PPOAlgorithm
    >>> ppo = PPOAlgorithm()
    >>> model = ppo.create_model(env, config)
    >>> ppo.train(model, total_timesteps=100000)
"""

from .ppo_algorithm import PPOAlgorithm

__all__ = ["PPOAlgorithm"]
