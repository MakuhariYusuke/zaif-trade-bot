"""
Market Analysis Package for ZAIF Trade Bot.

This package provides advanced market analysis capabilities including:
- Enhanced regime detection with statistical validation
- Technical indicator calculations
- Performance analysis and validation
"""

from ..market_regime_types import MarketRegime, RegimeDetectionResult
from .regime_analyzer import EnhancedRegimeAnalyzer

from .statistical_analyzer import (
    StatisticalAnalyzer,
    StatisticalTestResult,
    RegimeValidationMetrics
)

__all__ = [
    # Regime Analysis
    "EnhancedRegimeAnalyzer",
    "MarketRegime",
    "RegimeDetectionResult",

    # Statistical Analysis
    "StatisticalAnalyzer",
    "StatisticalTestResult",
    "RegimeValidationMetrics",
]
