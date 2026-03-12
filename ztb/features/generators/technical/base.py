# Compatibility shim to re-export feature base classes from core.
from ztb.features.core.base import BaseFeature, ParameterizedFeature

__all__ = ["BaseFeature", "ParameterizedFeature"]
