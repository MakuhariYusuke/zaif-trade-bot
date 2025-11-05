"""
Interfaces for Feature Weight Adjustment System
"""

from .adjustment_interface import WeightAdjustmentInterface
from .data_provider_interface import DataProviderInterface

__all__ = [
    'WeightAdjustmentInterface',
    'DataProviderInterface',
]