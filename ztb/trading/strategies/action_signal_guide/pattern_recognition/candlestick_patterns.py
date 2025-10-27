"""
Candlestick Pattern Recognition Module

This module provides pattern recognition for traditional Japanese candlestick patterns
used in technical analysis for trading signals.
"""

from typing import Optional, cast

import numpy as np
import pandas as pd

from ztb.trading.constants import ACTION_BUY, ACTION_SELL

from .base import CandlestickPatternRecognizer, SignalResult


class SakataFiveMethodsRecognizer(CandlestickPatternRecognizer):
    """Recognizes Sakata's Five Methods pattern.

    This is a complex multi-candle pattern involving trend continuation
    with specific candle arrangements.
    """

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Sakata's Five Methods pattern at the given index."""
        if index < 4:
            return None

        # Check for uptrend
        if not self._is_uptrend(data, index, lookback=5):
            return None

        # Pattern: Small candle, large candle, small candle, large candle, small candle
        candles = [data.iloc[index - i] for i in range(5)]

        # Validate pattern structure
        if not (
            self._is_small_candle(candles[4])
            and self._is_large_candle(candles[3])
            and self._is_small_candle(candles[2])
            and self._is_large_candle(candles[1])
            and self._is_small_candle(candles[0])
        ):
            return None

        # All large candles should be bullish
        if not (
            self.is_bullish_candle(candles[3]) and self.is_bullish_candle(candles[1])
        ):
            return None

        strength = 0.8
        return SignalResult(
            signal_type="sakata_five_methods",
            strength=strength,
            direction=ACTION_BUY,  # Buy signal
            description="Sakata's Five Methods: Complex bullish continuation pattern",
            timestamp=data.index[index],
            metadata={"pattern": "sakata_five_methods", "confidence": strength},
        )

    def _is_small_candle(self, candle: pd.Series) -> bool:
        """Check if candle has small body relative to recent volatility."""
        body_size = abs(candle["close"] - candle["open"])
        total_range = candle["high"] - candle["low"]
        return body_size / total_range < 0.3 if total_range > 0 else False

    def _is_large_candle(self, candle: pd.Series) -> bool:
        """Check if candle has large body relative to recent volatility."""
        body_size = abs(candle["close"] - candle["open"])
        total_range = candle["high"] - candle["low"]
        return body_size / total_range > 0.6 if total_range > 0 else False

    def _is_uptrend(self, data: pd.DataFrame, index: int, lookback: int) -> bool:
        """Check if there's an uptrend over the lookback period."""
        if index < lookback:
            return False
        recent_prices = data.iloc[index - lookback + 1 : index + 1]["close"]
        return cast(bool, recent_prices.iloc[-1] > recent_prices.iloc[0])

    def _is_downtrend(self, data: pd.DataFrame, index: int, lookback: int) -> bool:
        """Check if there's a downtrend over the lookback period."""
        if index < lookback:
            return False
        recent_prices = data.iloc[index - lookback + 1 : index + 1]["close"]
        return cast(bool, recent_prices.iloc[-1] < recent_prices.iloc[0])


class MorningStarRecognizer(CandlestickPatternRecognizer):
    """Recognizes Morning Star pattern.

    A three-candle bullish reversal pattern: large bearish, small, large bullish.
    """

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Morning Star pattern at the given index."""
        if index < 2:
            return None

        # Check for downtrend
        if not self._is_downtrend(data, index, lookback=5):
            return None

        first = data.iloc[index - 2]  # Large bearish
        second = data.iloc[index - 1]  # Small (star)
        third = data.iloc[index]  # Large bullish

        # First candle: large bearish
        avg_body_size = self._get_average_body_size(data, index, 10)
        if avg_body_size == 0:
            return None
        if not (
            self.is_bearish_candle(first)
            and self.calculate_body_size(data, index - 2) > avg_body_size
        ):
            return None
        if not (
            self.is_bearish_candle(first)
            and self.calculate_body_size(data, index - 2) > avg_body_size
        ):
            return None

        # Second candle: small body, can be bullish/bearish/doji
        star_body_ratio = self.get_body_ratio(data, index - 1)
        if star_body_ratio > 0.5:  # Must be relatively small
            return None

        # Third candle: large bullish that closes above midpoint of first candle
        if not self.is_bullish_candle(third):
            return None

        first_midpoint = (first["open"] + first["close"]) / 2
        if third["close"] <= first_midpoint:
            return None

        strength = 0.85
        return SignalResult(
            signal_type="morning_star",
            strength=strength,
            direction=ACTION_BUY,  # Buy signal
            description="Morning Star: Bullish reversal pattern",
            timestamp=data.index[index],
            metadata={"pattern": "morning_star", "confidence": strength},
        )

    def _get_average_body_size(
        self, data: pd.DataFrame, index: int, lookback: int
    ) -> float:
        """Calculate average body size over lookback period."""
        if index < lookback:
            return 0
        bodies = [
            self.calculate_body_size(data, i)
            for i in range(index - lookback + 1, index + 1)
        ]
        return cast(float, np.mean(bodies)) if bodies else 0


class EveningStarRecognizer(CandlestickPatternRecognizer):
    """Recognizes Evening Star pattern.

    A three-candle bearish reversal pattern: large bullish, small, large bearish.
    """

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Evening Star pattern at the given index."""
        if index < 2:
            return None

        # Check for uptrend
        if not self._is_uptrend(data, index, lookback=5):
            return None

        first = data.iloc[index - 2]  # Large bullish
        second = data.iloc[index - 1]  # Small (star)
        third = data.iloc[index]  # Large bearish

        # First candle: large bullish
        if not (
            self.is_bullish_candle(first)
            and self.calculate_body_size(data, index - 2)
            > self._get_average_body_size(data, index, 10)
        ):
            return None

        # Second candle: small body
        star_body_ratio = self.get_body_ratio(data, index - 1)
        if star_body_ratio > 0.5:
            return None

        # Third candle: large bearish that closes below midpoint of first candle
        if not self.is_bearish_candle(third):
            return None

        first_midpoint = (first["open"] + first["close"]) / 2
        if third["close"] >= first_midpoint:
            return None

        strength = 0.85
        return SignalResult(
            signal_type="evening_star",
            strength=strength,
            direction=ACTION_SELL,  # Sell signal
            description="Evening Star: Bearish reversal pattern",
            timestamp=data.index[index],
            metadata={"pattern": "evening_star", "confidence": strength},
        )


class HammerRecognizer(CandlestickPatternRecognizer):
    """Recognizes Hammer pattern.

    A single-candle bullish reversal pattern with long lower shadow.
    """

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Hammer pattern at the given index."""
        if index < 1:
            return None

        # Check for downtrend
        if not self._is_downtrend(data, index, lookback=5):
            return None

        candle = data.iloc[index]

        # Must be bullish or doji
        if self.is_bearish_candle(candle):
            return None

        body_size = self.calculate_body_size(data, index)
        lower_shadow = self.calculate_lower_shadow(data, index)
        upper_shadow = self.calculate_upper_shadow(data, index)
        total_range = candle["high"] - candle["low"]

        if total_range == 0:
            return None

        # Hammer characteristics:
        # - Lower shadow at least 2x body size
        # - Upper shadow small (less than body)
        # - Body in upper third of total range
        if not (
            lower_shadow >= 2 * body_size
            and upper_shadow <= body_size
            and candle["close"] > (total_range * 2 / 3 + candle["low"])
        ):
            return None

        strength = 0.75
        return SignalResult(
            signal_type="hammer",
            strength=strength,
            direction=ACTION_BUY,  # Buy signal
            description="Hammer: Bullish reversal pattern",
            timestamp=data.index[index],
            metadata={"pattern": "hammer", "confidence": strength},
        )


class HangingManRecognizer(CandlestickPatternRecognizer):
    """Recognizes Hanging Man pattern.

    A single-candle bearish reversal pattern with long lower shadow.
    """

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Hanging Man pattern at the given index."""
        if index < 1:
            return None

        # Check for uptrend
        if not self._is_uptrend(data, index, lookback=5):
            return None

        candle = data.iloc[index]

        # Must be bearish or doji
        if self.is_bullish_candle(candle):
            return None

        body_size = self.calculate_body_size(data, index)
        lower_shadow = self.calculate_lower_shadow(data, index)
        upper_shadow = self.calculate_upper_shadow(data, index)
        total_range = candle["high"] - candle["low"]

        if total_range == 0:
            return None

        # Hanging Man characteristics (similar to hammer but bearish):
        # - Lower shadow at least 2x body size
        # - Upper shadow small
        # - Body in upper third of total range
        if not (
            lower_shadow >= 2 * body_size
            and upper_shadow <= body_size
            and candle["open"] > (total_range * 2 / 3 + candle["low"])
        ):
            return None

        strength = 0.75
        return SignalResult(
            signal_type="hanging_man",
            strength=strength,
            direction=ACTION_SELL,  # Sell signal
            description="Hanging Man: Bearish reversal pattern",
            timestamp=data.index[index],
            metadata={"pattern": "hanging_man", "confidence": strength},
        )


class ThreeBlackCrowsRecognizer(CandlestickPatternRecognizer):
    """Recognizes Three Black Crows pattern.

    A three-candle bearish reversal pattern with three consecutive bearish candles.
    """

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Three Black Crows pattern at the given index."""
        if index < 2:
            return None

        # Check for uptrend
        if not self._is_uptrend(data, index, lookback=5):
            return None

        candles = [data.iloc[index - i] for i in range(3)]

        # All three candles must be bearish
        if not all(self.is_bearish_candle(candle) for candle in candles):
            return None

        # Each candle should open near previous close and close near low
        for i in range(1, 3):
            prev_close = candles[i - 1]["close"]
            curr_open = candles[i]["open"]
            curr_close = candles[i]["close"]
            curr_low = candles[i]["low"]

            # Opening near previous close
            if abs(curr_open - prev_close) / prev_close > 0.01:
                return None

            # Closing near low (bearish pressure)
            body_bottom = min(curr_open, curr_close)
            if (body_bottom - curr_low) / (candles[i]["high"] - curr_low) > 0.3:
                return None

        # Progressive lower closes
        if not (candles[0]["close"] > candles[1]["close"] > candles[2]["close"]):
            return None

        strength = 0.8
        return SignalResult(
            signal_type="three_black_crows",
            strength=strength,
            direction=ACTION_SELL,  # Sell signal
            description="Three Black Crows: Bearish reversal pattern",
            timestamp=data.index[index],
            metadata={"pattern": "three_black_crows", "confidence": strength},
        )


class ThreeWhiteSoldiersRecognizer(CandlestickPatternRecognizer):
    """Recognizes Three White Soldiers pattern.

    A three-candle bullish reversal pattern with three consecutive bullish candles.
    """

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Three White Soldiers pattern at the given index."""
        if index < 2:
            return None

        # Check for downtrend
        if not self._is_downtrend(data, index, lookback=5):
            return None

        candles = [data.iloc[index - i] for i in range(3)]

        # All three candles must be bullish
        if not all(self.is_bullish_candle(candle) for candle in candles):
            return None

        # Each candle should open near previous close and close near high
        for i in range(1, 3):
            prev_close = candles[i - 1]["close"]
            curr_open = candles[i]["open"]
            curr_close = candles[i]["close"]
            curr_high = candles[i]["high"]

            # Opening near previous close
            if abs(curr_open - prev_close) / prev_close > 0.01:
                return None

            # Closing near high (bullish pressure)
            body_top = max(curr_open, curr_close)
            if (curr_high - body_top) / (curr_high - candles[i]["low"]) > 0.3:
                return None

        # Progressive higher closes
        if not (candles[0]["close"] < candles[1]["close"] < candles[2]["close"]):
            return None

        strength = 0.8
        return SignalResult(
            signal_type="three_white_soldiers",
            strength=strength,
            direction=ACTION_BUY,  # Buy signal
            description="Three White Soldiers: Bullish reversal pattern",
            timestamp=data.index[index],
            metadata={"pattern": "three_white_soldiers", "confidence": strength},
        )


class RisingThreeMethodsRecognizer(CandlestickPatternRecognizer):
    """Recognizes Rising Three Methods pattern.

    A five-candle bullish continuation pattern.
    """

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Rising Three Methods pattern at the given index."""
        if index < 4:
            return None

        # Check for uptrend
        if not self._is_uptrend(data, index, lookback=5):
            return None

        candles = [data.iloc[index - i] for i in range(5)]

        # Pattern: Large bullish, three small candles, large bullish
        if not (
            self._is_large_candle(candles[4])
            and self._is_small_candle(candles[3])
            and self._is_small_candle(candles[2])
            and self._is_small_candle(candles[1])
            and self._is_large_candle(candles[0])
        ):
            return None

        # First and last candles must be bullish
        if not (
            self.is_bullish_candle(candles[4]) and self.is_bullish_candle(candles[0])
        ):
            return None

        # Three middle candles should be contained within first candle's range
        first_high = candles[4]["high"]
        first_low = candles[4]["low"]

        for i in range(1, 4):
            if candles[i]["high"] > first_high or candles[i]["low"] < first_low:
                return None

        strength = 0.75
        return SignalResult(
            signal_type="rising_three_methods",
            strength=strength,
            direction=ACTION_BUY,  # Buy signal
            description="Rising Three Methods: Bullish continuation pattern",
            timestamp=data.index[index],
            metadata={"pattern": "rising_three_methods", "confidence": strength},
        )


class BullishEngulfingRecognizer(CandlestickPatternRecognizer):
    """Recognizes Bullish Engulfing pattern."""

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Bullish Engulfing pattern at the given index."""
        if index < 1:
            return None

        # Check for downtrend
        if not self._is_downtrend(data, index, lookback=3):
            return None

        current = data.iloc[index]
        previous = data.iloc[index - 1]

        # Previous candle must be bearish
        if not self.is_bearish_candle(previous):
            return None

        # Current candle must be bullish
        if not self.is_bullish_candle(current):
            return None

        # Current candle must engulf previous candle completely
        current_open = current["open"]
        current_close = current["close"]
        prev_open = previous["open"]
        prev_close = previous["close"]

        # Bullish engulfing: current open <= prev_close and current_close >= prev_open
        if not (current_open <= prev_close and current_close >= prev_open):
            return None

        prev_body_size = abs(prev_close - prev_open)
        current_body_size = abs(current_close - current_open)

        if prev_body_size == 0 or current_body_size == 0:
            return None

        engulfing_ratio = current_body_size / prev_body_size
        strength = min(0.9, 0.6 + (engulfing_ratio - 1) * 0.2)

        engulfing_ratio = current_body_size / prev_body_size
        strength = min(0.9, 0.6 + (engulfing_ratio - 1) * 0.2)

        return SignalResult(
            signal_type="bullish_engulfing",
            strength=strength,
            direction=ACTION_BUY,  # Buy signal
            description="Bullish Engulfing: Strong reversal signal in downtrend",
            timestamp=data.index[index],
            metadata={
                "pattern": "bullish_engulfing",
                "confidence": strength,
                "engulfing_ratio": engulfing_ratio,
            },
        )


class BearishEngulfingRecognizer(CandlestickPatternRecognizer):
    """Recognizes Bearish Engulfing pattern."""

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Bearish Engulfing pattern at the given index."""
        if index < 1:
            return None

        # Check for uptrend
        if not self._is_uptrend(data, index, lookback=3):
            return None

        current = data.iloc[index]
        previous = data.iloc[index - 1]

        # Previous candle must be bullish
        if not self.is_bullish_candle(previous):
            return None

        # Current candle must be bearish
        if not self.is_bearish_candle(current):
            return None

        # Current candle must engulf previous candle completely
        current_open = current["open"]
        current_close = current["close"]
        prev_open = previous["open"]
        prev_close = previous["close"]

        # Bearish engulfing: current open >= prev_close and current_close <= prev_open
        if not (current_open >= prev_close and current_close <= prev_open):
            return None

        # Calculate strength based on engulfing ratio
        prev_body_size = abs(prev_close - prev_open)
        current_body_size = abs(current_close - current_open)

        if prev_body_size == 0:
            return None

        engulfing_ratio = current_body_size / prev_body_size
        strength = min(0.9, 0.6 + (engulfing_ratio - 1) * 0.2)

        return SignalResult(
            signal_type="bearish_engulfing",
            strength=strength,
            direction=ACTION_SELL,  # Sell signal
            description="Bearish Engulfing: Strong reversal signal in uptrend",
            timestamp=data.index[index],
            metadata={
                "pattern": "bearish_engulfing",
                "confidence": strength,
                "engulfing_ratio": engulfing_ratio,
            },
        )


class PiercingPatternRecognizer(CandlestickPatternRecognizer):
    """Recognizes Piercing Pattern."""

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Piercing Pattern at the given index."""
        if index < 1:
            return None

        # Check for downtrend
        if not self._is_downtrend(data, index, lookback=3):
            return None

        current = data.iloc[index]
        previous = data.iloc[index - 1]

        # Previous candle must be bearish
        if not self.is_bearish_candle(previous):
            return None

        # Current candle must be bullish
        if not self.is_bullish_candle(current):
            return None

        prev_open = previous["open"]
        prev_close = previous["close"]
        current_open = current["open"]
        current_close = current["close"]

        # Piercing pattern: current open < prev_close and current_close > midpoint of prev body
        prev_midpoint = (prev_open + prev_close) / 2

        if not (current_open < prev_close and current_close > prev_midpoint):
            return None

        # Calculate strength based on penetration depth
        prev_body_size = abs(prev_close - prev_open)
        if prev_body_size == 0:
            return None
        penetration = (current_close - prev_midpoint) / prev_body_size

        strength = min(0.85, 0.5 + penetration * 0.4)

        return SignalResult(
            signal_type="piercing_pattern",
            strength=strength,
            direction=ACTION_BUY,  # Buy signal
            description="Piercing Pattern: Bullish reversal signal in downtrend",
            timestamp=data.index[index],
            metadata={
                "pattern": "piercing_pattern",
                "confidence": strength,
                "penetration": penetration,
            },
        )
