"""
Trading features package.
特徴量パッケージ
"""

from .registry import FeatureRegistry


def get_feature_manager() -> FeatureRegistry:
    """Get the feature manager instance"""
    return FeatureRegistry()


# Import all feature modules to register functions
from . import momentum, time, trend, utils, volatility, volume

__all__ = ["FeatureRegistry", "get_feature_manager"]
