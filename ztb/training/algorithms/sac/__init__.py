"""SAC (Soft Actor-Critic) algorithm implementation.

An off-policy algorithm that uses entropy regularization to automatically
balance exploration and exploitation.
"""

from ztb.training.algorithms.sac.sac_algorithm import SACAlgorithm

__all__ = ["SACAlgorithm"]
