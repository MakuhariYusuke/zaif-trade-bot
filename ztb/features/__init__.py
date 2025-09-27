"""
Trading features package.
特徴量パッケージ
"""

from .registry import FeatureRegistry

def get_feature_manager() -> FeatureRegistry:
    """Get the feature manager instance"""
    return FeatureRegistry()

# Import all feature modules to register functions
from . import time, momentum, volatility, volume, trend, utils

__all__ = ['FeatureRegistry', 'get_feature_manager']