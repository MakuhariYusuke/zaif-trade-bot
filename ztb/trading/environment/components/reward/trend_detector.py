"""
Trend Detector for SAC v448.

Detects market trend to inform balance target adjustments.
Uses 5-minute timeframe to filter out 1-minute noise.

Version: 1.0
Created: 2025-11-22
Author: SAC v448 Development Team
"""

from collections import deque
from typing import Optional
import numpy as np
import logging


class TrendDetector:
    """
    Market trend detector using linear regression on price history.
    
    Designed for 1-minute timeframe trading to filter noise:
    - Uses configurable lookback window (default: 20 candles = 20 minutes for 1m)
    - Returns normalized trend signal: -1.0 (strong downtrend) to 1.0 (strong uptrend)
    - Integrates with behavioral penalty calculator for trend-aware balance adjustments
    
    Usage:
        detector = TrendDetector(lookback=20)
        
        # Update with each new price
        detector.update(current_price)
        
        # Get trend signal
        signal = detector.get_trend_signal()
        # signal ∈ [-1.0, 1.0]
        #   > 0.3: uptrend (favor BUY slightly)
        #   < -0.3: downtrend (favor SELL slightly)
        #   else: neutral (balanced targets)
    """
    
    def __init__(
        self,
        lookback: int = 20,
        min_samples: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize trend detector.
        
        Args:
            lookback: Number of candles for trend calculation (default: 20)
                     For 1m timeframe: 20 = 20 minutes
                     For 5m timeframe: 20 = 100 minutes
            min_samples: Minimum samples required (default: lookback)
            logger: Optional logger instance
        """
        self.lookback = lookback
        self.min_samples = min_samples if min_samples is not None else lookback
        self.price_history: deque = deque(maxlen=lookback)
        self.logger = logger or logging.getLogger(__name__)
        
        # Statistics
        self.update_count = 0
        self.last_signal = 0.0
        
        self.logger.info(
            f"TrendDetector initialized: lookback={lookback}, "
            f"min_samples={self.min_samples}"
        )
    
    def update(self, price: float) -> None:
        """
        Add new price observation to history.
        
        Args:
            price: Current market price
        """
        if not np.isfinite(price) or price <= 0:
            self.logger.warning(f"Invalid price: {price}, skipping update")
            return
        
        self.price_history.append(price)
        self.update_count += 1
        
        if self.update_count % 100 == 0:
            signal = self.get_trend_signal()
            self.logger.debug(
                f"Update #{self.update_count}: price={price:.2f}, "
                f"trend_signal={signal:.3f}"
            )
    
    def get_trend_signal(self) -> float:
        """
        Calculate trend signal from price history using linear regression.
        
        Method:
            1. Fit linear regression: price = slope * time + intercept
            2. Normalize slope by price range
            3. Scale to [-1, 1] range
        
        Returns:
            Trend signal ∈ [-1.0, 1.0]
                 1.0: Strong uptrend
                 0.0: Neutral / sideways
                -1.0: Strong downtrend
                
            Returns 0.0 if insufficient data.
        """
        # Check if we have enough data
        if len(self.price_history) < self.min_samples:
            return 0.0
        
        prices = np.array(list(self.price_history))
        n = len(prices)
        
        # Handle edge cases
        if n == 0:
            return 0.0
        
        if n == 1:
            return 0.0  # Can't determine trend from single point
        
        # Calculate linear regression slope
        # y = mx + b, where y = price, x = time
        x = np.arange(n)
        y = prices
        
        # Calculate slope using least squares
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        
        if denominator == 0:
            # All x values are identical (shouldn't happen with arange)
            self.logger.warning("Denominator zero in slope calculation")
            return 0.0
        
        slope = numerator / denominator
        
        # Normalize by price range
        price_range = y.max() - y.min()
        
        if price_range == 0:
            # Prices are all identical (flat market)
            return 0.0
        
        # Normalize: slope / (price_range / n)
        # This gives us "how many price_ranges per n steps"
        normalized_slope = slope / (price_range / n)
        
        # Clip to [-1, 1]
        signal = np.clip(normalized_slope, -1.0, 1.0)
        
        self.last_signal = signal
        
        return signal
    
    def get_trend_strength(self) -> str:
        """
        Get human-readable trend strength.
        
        Returns:
            String description: "Strong Uptrend", "Uptrend", "Neutral", etc.
        """
        signal = self.get_trend_signal()
        
        if signal > 0.6:
            return "Strong Uptrend"
        elif signal > 0.3:
            return "Uptrend"
        elif signal > -0.3:
            return "Neutral"
        elif signal > -0.6:
            return "Downtrend"
        else:
            return "Strong Downtrend"
    
    def reset(self) -> None:
        """Reset all history and statistics."""
        self.price_history.clear()
        self.update_count = 0
        self.last_signal = 0.0
        self.logger.info("TrendDetector reset")
    
    def get_statistics(self) -> dict:
        """
        Get detector statistics.
        
        Returns:
            Dictionary with:
                - samples: Number of price samples in history
                - update_count: Total updates
                - last_signal: Most recent trend signal
                - trend_strength: Human-readable strength
                - price_range: Current price range
        """
        prices = np.array(list(self.price_history)) if self.price_history else np.array([])
        
        return {
            "samples": len(self.price_history),
            "update_count": self.update_count,
            "last_signal": self.last_signal,
            "trend_strength": self.get_trend_strength(),
            "price_range": prices.max() - prices.min() if len(prices) > 0 else 0.0,
            "current_price": prices[-1] if len(prices) > 0 else None,
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"TrendDetector(lookback={self.lookback}, "
            f"samples={stats['samples']}, "
            f"signal={stats['last_signal']:.3f}, "
            f"strength='{stats['trend_strength']}')"
        )
