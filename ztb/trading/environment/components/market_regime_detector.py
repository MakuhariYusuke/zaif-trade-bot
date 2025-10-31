"""
Market Regime Detector Component.

This component is responsible for detecting market regimes based on price movements.
Follows Single Responsibility Principle by focusing only on regime detection.
"""

from collections import defaultdict, deque
from typing import Dict, Deque

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
        min_history_for_detection: int = 10,
    ):
        """
        Initialize MarketRegimeDetector.

        Args:
            regime_detection_window: Number of prices to keep in history
            adaptation_frequency: How often to update regime (in steps)
            high_volatility_threshold: Threshold for high volatility
            low_volatility_threshold: Threshold for low volatility
            trend_strength_threshold: Threshold for strong trend
            min_history_for_detection: Minimum price history required for regime detection
        """
        self.regime_detection_window = regime_detection_window
        self.adaptation_frequency = adaptation_frequency
        self.high_volatility_threshold = high_volatility_threshold
        self.low_volatility_threshold = low_volatility_threshold
        self.trend_strength_threshold = trend_strength_threshold
        self.min_history_for_detection = min_history_for_detection

        self.logger = get_logger("ztb.trading.environment.market_regime_detector")

        # Internal state
        self.price_history: Deque[float] = deque(maxlen=self.regime_detection_window)
        self.current_regime = "sideways"
        self.regime_step_counter = 0

        # Long-term regime statistics
        self.regime_counts: Dict[str, int] = defaultdict(int)
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

        # Need minimum history for regime detection
        # Calculate price changes and volatility
        prices = np.array(self.price_history)
        # Avoid division by zero by filtering out zero prices
        safe_prices = prices[:-1]
        price_diffs = np.diff(prices)
        # Replace zero prices with np.nan to avoid division by zero
        safe_prices = np.where(safe_prices == 0, np.nan, safe_prices)
        returns = np.divide(price_diffs, safe_prices)
        # Remove nan values from returns for further calculations
        returns = returns[~np.isnan(returns)]
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
        # Update regime only at adaptation frequency, but not at initialization (step 0)
        if step != 0 and step % self.adaptation_frequency == 0:
            # Determine regime based on trend strength and volatility
            if trend_strength > self.trend_strength_threshold:
                regime = "bull"
            elif trend_strength < -self.trend_strength_threshold:
                regime = "bear"
            elif volatility > self.high_volatility_threshold:
                regime = "volatile"
            else:
                regime = "sideways"
            
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
