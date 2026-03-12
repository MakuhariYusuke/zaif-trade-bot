"""
Market Regime Detector Component.

This component is responsible for detecting market regimes based on price movements.
Follows Single Responsibility Principle by focusing only on regime detection.
"""

from collections import defaultdict, deque
from typing import Protocol

import numpy as np

from ztb.utils.logging_utils import get_logger

class IMarketRegimeDetector(Protocol):
    def detect_regime(self, current_price: float, step: int) -> str:
        """Detect the current market regime."""

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
        self.price_history: deque[float] = deque(maxlen=self.regime_detection_window)
        self.current_regime = "sideways"
        self.regime_step_counter = 0

        # Long-term regime statistics
        self.regime_counts: dict[str, int] = defaultdict(int)
        self.total_steps_tracked = 0

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

        # Need minimum history for regime detection. Keep small window minimum
        # to allow earlier detection during tests and in small datasets.
        if len(self.price_history) < 2:
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

            # Update long-term statistics
            self.regime_counts[regime] += 1
            self.total_steps_tracked += 1

            # Log regime distribution summary periodically
            if self.total_steps_tracked % (self.adaptation_frequency * 10) == 0:
                self._log_regime_distribution()

        return self.current_regime

    def _log_regime_distribution(self) -> None:
        """Log the distribution of regimes over time."""
        if self.total_steps_tracked == 0:
            return

        total_regimes = sum(self.regime_counts.values())
        if total_regimes == 0:
            return

        distribution = {}
        for regime, count in self.regime_counts.items():
            percentage = (count / total_regimes) * 100
            distribution[regime] = f"{percentage:.1f}%"

        self.logger.info(
            f"Regime distribution over {self.total_steps_tracked} steps: {distribution}"
        )
