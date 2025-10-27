"""
Granville's Law pattern recognition for Action Signal Guide.

Joseph Granville's trading rules based on the relationship between price and volume.
This implementation focuses on the core principles while providing configurable parameters.
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ztb.features.volume.obv import compute_obv
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    PatternRecognizer,
    SignalResult,
)


class GranvilleLawRecognizer(PatternRecognizer):
    """
    Recognizes patterns using Granville's Law.

    Based on Joseph Granville's 8 rules for interpreting price and volume relationships.
    This implementation provides the core rules with configurable parameters.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Price change thresholds
        self.price_change_threshold = self.config.get('price_change_threshold', 0.005)  # 0.5%
        # Volume change thresholds
        self.volume_change_threshold = self.config.get('volume_change_threshold', 0.1)  # 10%
        # Trend determination period
        self.trend_period = self.config.get('trend_period', 20)  # 20 periods for trend
        # Minimum volume for valid signals
        self.min_volume = self.config.get('min_volume', 1000)
        # Use OBV for volume analysis
        self.use_obv = self.config.get('use_obv', True)

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """
        Recognize Granville's Law patterns in the data.

        Args:
            data: OHLCV DataFrame
            index: Index to analyze (-1 for latest)

        Returns:
            SignalResult if pattern detected, None otherwise
        """
        if len(data) < self.trend_period + 2:
            return None

        if index == -1:
            index = len(data) - 1

        if index < self.trend_period + 1:
            return None

        # Get required data window
        window_data = data.iloc[index - self.trend_period:index + 1]

        # Calculate price and volume changes
        price_change, volume_change = self._calculate_changes(window_data)

        # Determine market trend
        market_trend = self._determine_market_trend(window_data)

        # Apply Granville's rules
        signal = self._apply_granville_rules(
            price_change, volume_change, market_trend, window_data.iloc[-1]
        )

        if signal:
            return SignalResult(
                signal_type="granville_law",
                strength=abs(signal['strength']),
                direction=signal['direction'],
                description=signal['description'],
                confidence=signal['confidence']
            )

        return None

    def _calculate_changes(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate price and volume changes over the analysis period.

        Returns:
            Tuple of (price_change_ratio, volume_change_ratio)
        """
        if len(data) < 2:
            return 0.0, 0.0

        # Price change (close to close)
        recent_close = data['close'].iloc[-1]
        previous_close = data['close'].iloc[-2]
        price_change = (recent_close - previous_close) / previous_close

        # Volume change - use OBV if enabled, otherwise raw volume
        if self.use_obv and len(data) >= 5:  # Need minimum data for OBV
            try:
                obv_series = compute_obv(data)
                recent_obv = obv_series.iloc[-1]
                previous_obv = obv_series.iloc[-2]
                volume_change = (recent_obv - previous_obv) / abs(previous_obv) if previous_obv != 0 else 0.0
            except Exception:
                # Fallback to raw volume if OBV calculation fails
                recent_volume = data['volume'].iloc[-1]
                previous_volume = data['volume'].iloc[-2]
                volume_change = (recent_volume - previous_volume) / previous_volume if previous_volume != 0 else 0.0
        else:
            # Raw volume change
            recent_volume = data['volume'].iloc[-1]
            previous_volume = data['volume'].iloc[-2]
            volume_change = (recent_volume - previous_volume) / previous_volume if previous_volume != 0 else 0.0

        return price_change, volume_change

    def _determine_market_trend(self, data: pd.DataFrame) -> str:
        """
        Determine the current market trend (bullish/bearish).

        Returns:
            'bullish', 'bearish', or 'sideways'
        """
        if len(data) < self.trend_period:
            return 'sideways'

        # Simple trend determination based on moving averages
        closes = data['close']
        ma_short = closes.rolling(window=min(5, len(closes))).mean()
        ma_long = closes.rolling(window=min(self.trend_period, len(closes))).mean()

        if len(ma_short) < 2 or len(ma_long) < 2:
            return 'sideways'

        # Compare recent short MA vs long MA
        recent_short = ma_short.iloc[-1]
        recent_long = ma_long.iloc[-1]

        if recent_short > recent_long * 1.005:  # 0.5% above
            return 'bullish'
        elif recent_short < recent_long * 0.995:  # 0.5% below
            return 'bearish'
        else:
            return 'sideways'

    def _apply_granville_rules(
        self,
        price_change: float,
        volume_change: float,
        market_trend: str,
        current_data: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """
        Apply Granville's Law rules to generate trading signals.

        Core Rules:
        1. Price up + Volume up = Buy (strong bullish)
        2. Price down + Volume up = Sell (strong bearish)
        3. Price up + Volume down = Buy (in bear market only)
        4. Price down + Volume down = Sell (in bull market only)
        """
        # Check minimum volume
        if current_data.get('volume', 0) < self.min_volume:
            return None

        price_up = price_change > self.price_change_threshold
        price_down = price_change < -self.price_change_threshold
        volume_up = volume_change > self.volume_change_threshold
        volume_down = volume_change < -self.volume_change_threshold

        # Rule 1: Price up + Volume up = Strong Buy
        if price_up and volume_up:
            strength = min(1.0, abs(price_change) * abs(volume_change) * 100)
            return {
                'direction': ACTION_BUY,
                'strength': strength,
                'description': f"Granville Rule 1: Price ↑ + Volume ↑ = Strong Buy (trend: {market_trend})",
                'confidence': min(0.9, strength)
            }

        # Rule 2: Price down + Volume up = Strong Sell
        elif price_down and volume_up:
            strength = min(1.0, abs(price_change) * abs(volume_change) * 100)
            return {
                'direction': ACTION_SELL,
                'strength': strength,
                'description': f"Granville Rule 2: Price ↓ + Volume ↑ = Strong Sell (trend: {market_trend})",
                'confidence': min(0.9, strength)
            }

        # Rule 3: Price up + Volume down = Buy (bear market only)
        elif price_up and volume_down and market_trend == 'bearish':
            strength = min(0.7, abs(price_change) * 50)
            return {
                'direction': ACTION_BUY,
                'strength': strength,
                'description': f"Granville Rule 3: Price ↑ + Volume ↓ = Buy in Bear Market",
                'confidence': min(0.6, strength)
            }

        # Rule 4: Price down + Volume down = Sell (bull market only)
        elif price_down and volume_down and market_trend == 'bullish':
            strength = min(0.7, abs(price_change) * 50)
            return {
                'direction': ACTION_SELL,
                'strength': strength,
                'description': f"Granville Rule 4: Price ↓ + Volume ↓ = Sell in Bull Market",
                'confidence': min(0.6, strength)
            }

        # Additional rules for sideways/invalid signals
        # Rule 5: Price sideways + Volume up = Potential accumulation
        if not price_up and not price_down and volume_up:
            return {
                'direction': ACTION_HOLD,
                'strength': 0.3,
                'description': "Granville Rule 5: Sideways Price + Volume ↑ = Accumulation",
                'confidence': 0.4
            }

        # Rule 6: Price sideways + Volume down = Potential distribution
        elif not price_up and not price_down and volume_down:
            return {
                'direction': ACTION_HOLD,
                'strength': 0.3,
                'description': "Granville Rule 6: Sideways Price + Volume ↓ = Distribution",
                'confidence': 0.4
            }

        return None