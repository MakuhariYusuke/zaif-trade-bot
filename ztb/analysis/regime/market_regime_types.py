"""
Common type definitions for market regime analysis.

This module contains shared type definitions used across different
market regime detection implementations.
"""

from dataclasses import dataclass
from enum import Enum

class MarketRegime(Enum):
    """Enumeration of market regimes with standardized definitions."""

    STRONG_BULL_TREND = "strong_bull_trend"
    MODERATE_BULL_TREND = "moderate_bull_trend"
    WEAK_BULL_TREND = "weak_bull_trend"
    STRONG_BEAR_TREND = "strong_bear_trend"
    MODERATE_BEAR_TREND = "moderate_bear_trend"
    WEAK_BEAR_TREND = "weak_bear_trend"
    HIGH_VOLATILITY_RANGING = "high_volatility_ranging"
    MODERATE_VOLATILITY_RANGING = "moderate_volatility_ranging"
    LOW_VOLATILITY_RANGING = "low_volatility_ranging"
    EXTREME_VOLATILITY = "extreme_volatility"
    CONSOLIDATION = "consolidation"
    BREAKOUT_SETUP = "breakout_setup"
    BREAKDOWN_SETUP = "breakdown_setup"

    # New regimes from v444 classifier
    BUY_BREAKOUT = "buy_breakout"
    BUY_DIVERGENCE = "buy_divergence"
    BUY_MOMENTUM_STRONG = "buy_momentum_strong"
    BUY_VOLUME_SURGE = "buy_volume_surge"
    SELL_BREAKDOWN = "sell_breakdown"
    SELL_DIVERGENCE = "sell_divergence"
    SELL_MOMENTUM_WEAK = "sell_momentum_weak"
    SELL_VOLUME_SURGE = "sell_volume_surge"

@dataclass
class RegimeDetectionResult:
    """Standardized result of regime detection with optional classification path."""

    regime: MarketRegime
    confidence: float
    indicators: dict[str, float]
    metadata: dict[str, object]
    classification_path: list[
        str
    ] = None  # Optional: track which conditions led to this regime

    def __post_init__(self):
        """Initialize optional fields."""
        if self.classification_path is None:
            self.classification_path = []

# Backwards compatibility aliases for older enum member names
MarketRegime.STRONG_BULL = MarketRegime.STRONG_BULL_TREND
MarketRegime.MODERATE_BULL = MarketRegime.MODERATE_BULL_TREND
MarketRegime.WEAK_BULL = MarketRegime.WEAK_BULL_TREND
MarketRegime.STRONG_BEAR = MarketRegime.STRONG_BEAR_TREND
MarketRegime.MODERATE_BEAR = MarketRegime.MODERATE_BEAR_TREND
MarketRegime.WEAK_BEAR = MarketRegime.WEAK_BEAR_TREND

# Range compat
MarketRegime.TIGHT_RANGE = MarketRegime.LOW_VOLATILITY_RANGING
MarketRegime.WIDE_RANGE = MarketRegime.MODERATE_VOLATILITY_RANGING
MarketRegime.VOLATILE_RANGE = MarketRegime.HIGH_VOLATILITY_RANGING
MarketRegime.QUIET_RANGE = MarketRegime.LOW_VOLATILITY_RANGING

# Short-name range aliases (without _ING suffix)
MarketRegime.HIGH_VOLATILITY_RANGE = MarketRegime.HIGH_VOLATILITY_RANGING
MarketRegime.MODERATE_VOLATILITY_RANGE = MarketRegime.MODERATE_VOLATILITY_RANGING
MarketRegime.LOW_VOLATILITY_RANGE = MarketRegime.LOW_VOLATILITY_RANGING

# Legacy/common naming compat used across older strategy components
MarketRegime.RANGING = MarketRegime.MODERATE_VOLATILITY_RANGING
MarketRegime.HIGH_VOLATILITY = MarketRegime.HIGH_VOLATILITY_RANGING
MarketRegime.LOW_VOLATILITY = MarketRegime.LOW_VOLATILITY_RANGING
MarketRegime.TRENDING_BULLISH = MarketRegime.MODERATE_BULL_TREND
MarketRegime.TRENDING_BEARISH = MarketRegime.MODERATE_BEAR_TREND
MarketRegime.SIDEWAYS = MarketRegime.MODERATE_VOLATILITY_RANGING

# Pattern compat
MarketRegime.BREAKOUT_UP = MarketRegime.BREAKOUT_SETUP
MarketRegime.BREAKOUT_DOWN = MarketRegime.BREAKDOWN_SETUP
# Reversal maps to consolidation
MarketRegime.REVERSAL_UP = MarketRegime.CONSOLIDATION
MarketRegime.REVERSAL_DOWN = MarketRegime.CONSOLIDATION
