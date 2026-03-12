"""
Market Regime Detection Package.

This package provides various market regime detection capabilities
for adaptive trading strategies.

157# §19: AdvancedRegimeDetector は dead code として archived に移動。
fill_test は mid_price のみ保有、AdvancedRegimeDetector は high/low/close 3入力必要
で入力要件不適合 (152# で不採用判断済み)。
"""

from .market_regime_types import MarketRegime, RegimeDetectionResult
from .basic_regime_detector import MarketRegimeDetector

__all__ = [
    'MarketRegime',
    'RegimeDetectionResult',
    'MarketRegimeDetector'
]
