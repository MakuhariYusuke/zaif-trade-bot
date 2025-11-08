"""
Common type definitions for market regime analysis.

This module contains shared type definitions used across different
market regime detection implementations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any


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


@dataclass
class RegimeDetectionResult:
    """Standardized result of regime detection with optional classification path."""
    regime: MarketRegime
    confidence: float
    indicators: Dict[str, float]
    metadata: Dict[str, Any]
    classification_path: List[str] = None  # Optional: track which conditions led to this regime

    def __post_init__(self):
        """Initialize optional fields."""
        if self.classification_path is None:
            self.classification_path = []