"""
Dynamic Feature Weight Adjustment System

This module provides dynamic adjustment of feature weights based on SAC learning
and signal performance analysis. The system automatically optimizes feature
importance to improve trading performance.

Key Components:
- Weight Adjuster: Core adjustment logic
- Adjustment Strategies: Different algorithms for weight optimization
- Performance Evaluator: Measures effectiveness of weight adjustments
- Data Providers: Interfaces for SAC learning and signal performance data
"""

from .core.weight_adjuster import DynamicWeightAdjuster
from .core.adjustment_strategies import (
    AdjustmentStrategyRegistry,
    PerformanceDrivenStrategy,
    CorrelationBasedStrategy,
    ReinforcementLearningStrategy,
)
from .core.performance_evaluator import PerformanceEvaluator
from .interfaces.adjustment_interface import WeightAdjustmentInterface
from .interfaces.data_provider_interface import DataProviderInterface
from .config.adjustment_config import AdjustmentConfig, AdjustmentStrategyType

__all__ = [
    'DynamicWeightAdjuster',
    'AdjustmentStrategyRegistry',
    'PerformanceDrivenStrategy',
    'CorrelationBasedStrategy',
    'ReinforcementLearningStrategy',
    'PerformanceEvaluator',
    'WeightAdjustmentInterface',
    'DataProviderInterface',
    'AdjustmentConfig',
    'AdjustmentStrategyType',
]