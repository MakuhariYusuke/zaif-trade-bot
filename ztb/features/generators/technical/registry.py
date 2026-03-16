# Compatibility shim to re-export FeatureRegistry from core
from ztb.features.core.registry import FeatureRegistry

__all__ = ["FeatureRegistry"]
