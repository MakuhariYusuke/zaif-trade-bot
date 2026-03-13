"""Compatibility shim for old path `ztb.training.adv_norm`.
Re-exports from `ztb.training.optimization.adv_norm`.
"""
from ztb.training.optimization.adv_norm import PerActionAdvantageNormalizer  # noqa: F401

__all__ = ["PerActionAdvantageNormalizer"]
