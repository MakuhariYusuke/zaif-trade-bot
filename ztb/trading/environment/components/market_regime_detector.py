"""
Market Regime Detector Component.

This component is responsible for detecting market regimes based on price movements.
Follows Single Responsibility Principle by focusing only on regime detection.
"""

import math
from typing import List

import numpy as np

from ztb.utils.logging_utils import get_logger

from .interfaces import IMarketRegimeDetector


class MarketRegimeDetector(IMarketRegimeDetector):
    """
    Detects market regimes based on price movement patterns.

    This class encapsulates all market regime detection logic including:
    - Price history tracking
    - Trend strength calculation
    - Volatility assessment
    - Regime classification (bull, bear, sideways, volatile)
    """

    def __init__(
        self,
        regime_detection_window: int = 20,
        adaptation_frequency: int = 10,
        high_volatility_threshold: float = 0.02,
        low_volatility_threshold: float = 0.005,
        trend_strength_threshold: float = 0.001,
    ):
        """
        Initialize MarketRegimeDetector.

        Args:
            regime_detection_window: Number of prices to keep in history
            adaptation_frequency: How often to update regime (in steps)
            high_volatility_threshold: Threshold for high volatility
            low_volatility_threshold: Threshold for low volatility
            trend_strength_threshold: Threshold for strong trend
        """
        self.regime_detection_window = regime_detection_window
        self.adaptation_frequency = adaptation_frequency
        self.high_volatility_threshold = high_volatility_threshold
        self.low_volatility_threshold = low_volatility_threshold
        self.trend_strength_threshold = trend_strength_threshold

        self.logger = get_logger("ztb.trading.environment.market_regime_detector")

        # Internal state
        self.price_history: List[float] = []
        self.current_regime = "sideways"
        self.regime_step_counter = 0

    def detect_regime(self, current_price: float, step: int) -> str:
        """
        Detect current market regime based on price movement patterns.

        Args:
            current_price: Current market price
            step: Current step number

        Returns:
            Market regime: 'bull', 'bear', 'sideways', 'volatile'
        """
        # Update price history
        self.price_history.append(current_price)
        if len(self.price_history) > self.regime_detection_window:
            self.price_history.pop(0)

        # Need minimum history for regime detection
        if len(self.price_history) < 10:
            return "sideways"

        # Calculate price changes and volatility
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]

        # Trend strength (cumulative return over window)
        if len(returns) > 0:
            trend_strength = np.sum(returns)
        else:
            trend_strength = 0.0

        # Volatility (standard deviation of returns)
        if len(returns) > 1:
            volatility = np.std(returns)
        else:
            volatility = 0.0

        # Detect regime based on trend and volatility
        if volatility > self.high_volatility_threshold:
            regime = "volatile"
        elif abs(trend_strength) > self.trend_strength_threshold:
            regime = "bull" if trend_strength > 0 else "bear"
        else:
            regime = "sideways"

        # Update regime only at adaptation frequency
        if step % self.adaptation_frequency == 0:
            self.current_regime = regime
            self.logger.debug(
                f"Market regime updated to: {regime} "
                f"(trend: {trend_strength:.4f}, vol: {volatility:.4f})"
            )

        return self.current_regime