"""
Market Regime Detection Package.

This package provides various market regime detection capabilities
for adaptive trading strategies.
"""

from .market_regime_types import MarketRegime, RegimeDetectionResult
from .advanced_regime_detector import AdvancedRegimeDetector, TechnicalIndicators
from .basic_regime_detector import MarketRegimeDetector

__all__ = [
    'AdvancedRegimeDetector',
    'MarketRegime',
    'TechnicalIndicators',
    'RegimeDetectionResult',
    'MarketRegimeDetector'
]
