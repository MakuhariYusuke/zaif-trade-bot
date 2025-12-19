"""
Candlestick Pattern Recognition Module

This module provides pattern recognition for traditional Japanese candlestick patterns
used in technical analysis for trading signals.
"""

import logging
from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd

from .base import CandlestickPatternRecognizer, SignalResult

# Standardized constants for candlestick pattern recognition
PATTERN_CONFIDENCE_WEIGHTS = {
    "trend_strength": 0.3,
    "candle_size": 0.25,
    "price_movement": 0.25,
    "pattern_completeness": 0.2,
}

BASE_CONFIDENCE_LEVELS = {
    "sakata_five_methods": 0.7,
    "morning_star": 0.7,
    "three_white_soldiers": 0.8,
    "three_black_crows": 0.8,
    "hammer": 0.6,
    "shooting_star": 0.6,
    "engulfing": 0.7,
    "harami": 0.6,
    "piercing": 0.7,
    "dark_cloud_cover": 0.7,
    "doji": 0.5,
}

CANDLE_SIZE_THRESHOLDS = {
    "min_body_ratio": 0.3,  # Minimum body size relative to total range
    "max_body_ratio": 0.8,  # Maximum body size relative to total range
    "shadow_ratio_threshold": 2.0,  # Shadow to body ratio for hammer/shooting star
}

TREND_STRENGTH_THRESHOLDS = {
    "min_trend_strength": 0.4,  # Minimum trend strength for pattern validity
    "strong_trend_threshold": 0.7,  # Threshold for strong trend
}

VOLUME_THRESHOLDS = {
    "volume_increase_ratio": 1.2,  # Minimum volume increase for confirmation
    "volume_decrease_ratio": 0.8,  # Maximum volume decrease allowed
}


class SakataFiveMethodsRecognizer(CandlestickPatternRecognizer):
    """Recognizes Sakata's Five Methods pattern.
    酒田五法

    This is a complex multi-candle pattern involving trend continuation
    with specific candle arrangements.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "sakata_five_methods"
        self.logger = logging.getLogger(__name__)
        # Override default lookback period for this specific pattern
        if "lookback_period" not in self.config:
            self.config["lookback_period"] = 10

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Sakata's Five Methods pattern at the given index."""
        try:
            # Validate inputs using common method - Sakata's Five Methods requires more data
            index = self.validate_recognition_inputs(data, index, required_length=15)

            # Check for uptrend
            if not self._is_uptrend(data, index, lookback=5):
                return None

            # Pattern: Small candle, large candle, small candle, large candle, small candle
            candles = [data.iloc[index - i] for i in range(5)]

            # Validate pattern structure using common method
            expected_directions = [
                "any",
                "bullish",
                "any",
                "bullish",
                "any",
            ]  # Small, large, small, large, small
            indices_to_check = [index - i for i in range(5)]
            if not self.validate_pattern_structure(
                data, indices_to_check, expected_directions
            ):
                return None

            # Check size characteristics
            size_checks = []
            for i, candle_idx in enumerate(indices_to_check):
                if i % 2 == 0:  # Small candles at even positions (0, 2, 4)
                    size_checks.append(self._is_small_candle(candles[i]))
                else:  # Large candles at odd positions (1, 3)
                    size_checks.append(self._is_large_candle(candles[i]))

            if not all(size_checks):
                return None

            # Analyze candle characteristics using common method
            characteristics = self.analyze_multiple_candle_characteristics(
                data, indices_to_check
            )

            if not (
                characteristics["body_sizes"][3]
                > characteristics["body_sizes"][
                    1
                ]  # Second large (pos 3) > fourth large (pos 1)
                and characteristics["body_sizes"][1]
                > characteristics["body_sizes"][3] * 0.7  # Fourth large substantial
            ):
                return None

            confidence = self._calculate_confidence(data, index, characteristics)

            # Apply multi-timeframe alignment adjustment if data is available
            if multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_alignment(
                    data, index, multi_timeframe_data, self.pattern_type
                )
                confidence = min(1.0, confidence * mtf_confidence)

            return SignalResult(
                signal_type="sakata_five_methods",
                strength=confidence,
                direction=1.0,  # Buy signal
                description="Sakata's Five Methods: Complex bullish continuation pattern",
                timestamp=data.index[index],
                confidence=confidence,
                metadata={
                    "pattern": "sakata_five_methods",
                    "trend": "uptrend",
                    "candle_characteristics": characteristics,
                    "momentum_increase": True,
                },
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Sakata's Five Methods pattern: {e}")
            return None

    def _calculate_confidence(
        self, data: pd.DataFrame, index: int, characteristics: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for Sakata's Five Methods pattern."""
        try:
            # Calculate pattern completeness based on candle characteristics
            small_candle_scores = []
            large_candle_scores = []

            for i in range(5):
                if i % 2 == 0:  # Small candles at even positions (0, 2, 4)
                    is_small = (
                        characteristics["body_sizes"][i]
                        < characteristics["avg_body_size"] * 0.5
                    )
                    small_candle_scores.append(1.0 if is_small else 0.5)
                else:  # Large candles at odd positions (1, 3)
                    is_large = (
                        characteristics["body_sizes"][i]
                        > characteristics["avg_body_size"] * 1.5
                    )
                    large_candle_scores.append(1.0 if is_large else 0.5)

            pattern_completeness = (
                sum(small_candle_scores) / len(small_candle_scores) * 0.6
                + sum(large_candle_scores) / len(large_candle_scores) * 0.4
            )

            # Calculate pattern factors
            pattern_factors = {
                "trend_strength": self._calculate_trend_strength(data, index, 10),
                "candle_size": self._calculate_candle_size_confidence(data, index, 0.7),
                "price_movement": self._calculate_price_movement_confidence(
                    data, index, 0.8
                ),
                "pattern_completeness": pattern_completeness,
                "volume": self._calculate_volume_confidence(
                    data, index, 1.2
                ),  # Expect volume increase for reversal
            }

            confidence = self._calculate_pattern_confidence(
                data,
                index,
                pattern_factors,
                base_confidence=self.get_base_confidence_for_pattern(self.pattern_type),
            )
            return confidence

        except Exception as e:
            self.logger.error(
                f"Error calculating confidence for Sakata's Five Methods: {e}"
            )
            return 0.5

    def _is_small_candle(self, candle: pd.Series) -> bool:
        """Check if candle has small body relative to recent volatility."""
        body_size = abs(candle["close"] - candle["open"])
        total_range = candle["high"] - candle["low"]
        return (
            (body_size / total_range).item() < 0.3 if total_range.item() > 0 else False
        )

    def _is_large_candle(self, candle: pd.Series) -> bool:
        """Check if candle has large body relative to recent volatility."""
        body_size = abs(candle["close"] - candle["open"])
        total_range = candle["high"] - candle["low"]
        return (
            (body_size / total_range).item() >= 0.6 if total_range.item() > 0 else False
        )


    def _is_downtrend(self, data: pd.DataFrame, index: int, lookback: int) -> bool:
        """Check if there's a downtrend over the lookback period."""
        if index < lookback:
            return False
        recent_prices = data.iloc[index - lookback + 1 : index + 1]["close"]
        return cast(bool, recent_prices.iloc[-1] < recent_prices.iloc[0])
    A three-candle bullish reversal pattern: large bearish, small, large bullish.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "morning_star"
        self.logger = logging.getLogger(__name__)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Morning Star pattern at the given index."""
        try:
            # Validate inputs using common method
            index = self.validate_recognition_inputs(data, index, required_length=3)

            # Check for downtrend
            if not self._is_downtrend(data, index, lookback=5):
                return None

            # Pattern: Large bearish, small, large bullish
            indices_to_check = [index - 2, index - 1, index]
            expected_directions = ["bearish", "any", "bullish"]

            # Validate pattern structure using common method
            if not self.validate_pattern_structure(
                data, indices_to_check, expected_directions
            ):
                return None

            # Analyze candle characteristics using common method
            characteristics = self.analyze_multiple_candle_characteristics(
                data, indices_to_check
            )

            # Additional validation: third candle closes above midpoint of first candle
            first = data.iloc[index - 2]
            third = data.iloc[index]
            first_midpoint = (first["open"] + first["close"]) / 2
            if third["close"].item() <= first_midpoint:
                return None

            # Check size characteristics
            if not (
                characteristics["body_sizes"][0] > characteristics["avg_body_size"]
                and characteristics["body_sizes"][2]
                > characteristics["avg_body_size"]  # First large
                and characteristics["body_sizes"][1]  # Third large
                < characteristics["avg_body_size"] * 0.5  # Second small
            ):
                return None

            # Calculate dynamic confidence based on pattern quality
            pattern_factors = {
                "trend_strength": self._calculate_trend_strength(data, index, 5),
                "candle_size": self._calculate_candle_size_confidence(
                    data, index, 0.8
                ),  # Large candles expected
                "price_movement": self._calculate_price_movement_confidence(
                    data, index, 1.0
                ),  # Significant movement
                "pattern_completeness": min(
                    1.0,
                    characteristics["body_sizes"][1]
                    / characteristics["avg_body_size"]
                    * 2,
                ),  # Smaller star body = more complete pattern
                "volume": self._calculate_volume_confidence(
                    data, index, 1.2
                ),  # Expect volume increase for reversal
            }

            confidence = self._calculate_pattern_confidence(
                data,
                index,
                pattern_factors,
                base_confidence=self.get_base_confidence_for_pattern(self.pattern_type),
            )

            # Apply multi-timeframe alignment adjustment if data is available
            if multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_alignment(
                    data, index, multi_timeframe_data, self.pattern_type
                )
                confidence = min(1.0, confidence * mtf_confidence)

            return SignalResult(
                signal_type="morning_star",
                strength=confidence,
                direction=1.0,  # Buy signal
                description="Morning Star: Bullish reversal pattern",
                timestamp=data.index[index],
                confidence=confidence,
                metadata={"pattern": "morning_star", "confidence": confidence},
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Morning Star pattern: {e}")
            return None



class EveningStarRecognizer(CandlestickPatternRecognizer):
    """Recognizes Evening Star pattern.
    宵の明星

    A three-candle bearish reversal pattern: large bullish, small, large bearish.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "evening_star"
        self.logger = logging.getLogger(__name__)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Evening Star pattern at the given index."""
        try:
            # Validate inputs using common method
            index = self.validate_recognition_inputs(data, index, required_length=3)

            # Check for uptrend
            if not self._is_uptrend(data, index, lookback=5):
                return None

            # Pattern: Large bullish, small, large bearish
            indices_to_check = [index - 2, index - 1, index]
            expected_directions = ["bullish", "any", "bearish"]

            # Validate pattern structure using common method
            if not self.validate_pattern_structure(
                data, indices_to_check, expected_directions
            ):
                return None

            # Analyze candle characteristics using common method
            characteristics = self.analyze_multiple_candle_characteristics(
                data, indices_to_check
            )

            # Additional validation: third candle closes below midpoint of first candle
            first = data.iloc[index - 2]
            third = data.iloc[index]
            first_midpoint = (first["open"] + first["close"]) / 2
            if third["close"].item() >= first_midpoint:
                return None

            # Check size characteristics
            if not (
                characteristics["body_sizes"][0] > characteristics["avg_body_size"]
                and characteristics["body_sizes"][2]
                > characteristics["avg_body_size"]  # First large
                and characteristics["body_sizes"][1]  # Third large
                < characteristics["avg_body_size"] * 0.5  # Second small
            ):
                return None

            # Calculate dynamic confidence based on pattern quality
            pattern_factors = {
                "trend_strength": self._calculate_trend_strength(data, index, 5),
                "candle_size": self._calculate_candle_size_confidence(
                    data, index, 0.8
                ),  # Large candles expected
                "price_movement": self._calculate_price_movement_confidence(
                    data, index, 1.0
                ),  # Significant movement
                "pattern_completeness": min(
                    1.0,
                    characteristics["body_sizes"][1]
                    / characteristics["avg_body_size"]
                    * 2,
                ),  # Smaller star body = more complete pattern
                "volume": self._calculate_volume_confidence(
                    data, index, 1.2
                ),  # Expect volume increase for reversal
            }

            confidence = self._calculate_pattern_confidence(
                data,
                index,
                pattern_factors,
                base_confidence=self.get_base_confidence_for_pattern(self.pattern_type),
            )

            # Apply multi-timeframe alignment adjustment if data is available
            if multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_alignment(
                    data, index, multi_timeframe_data, self.pattern_type
                )
                confidence = min(1.0, confidence * mtf_confidence)

            return SignalResult(
                signal_type="evening_star",
                strength=confidence,
                direction=-1.0,  # Sell signal
                description="Evening Star: Bearish reversal pattern",
                timestamp=data.index[index],
                confidence=confidence,
                metadata={"pattern": "evening_star", "confidence": confidence},
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Evening Star pattern: {e}")
            return None


class HammerRecognizer(CandlestickPatternRecognizer):
    """Recognizes Hammer pattern.
    捨て子底

    A single-candle bullish reversal pattern with long lower shadow.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "hammer"
        self.logger = logging.getLogger(__name__)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Hammer pattern at the given index."""
        try:
            # Validate inputs using common method
            index = self.validate_recognition_inputs(data, index, required_length=1)

            # Check for downtrend
            if not self._is_downtrend(data, index, lookback=5):
                return None

            # Analyze candle characteristics using common method
            characteristics = self.analyze_multiple_candle_characteristics(
                data, [index]
            )

            candle = data.iloc[index]

            # Must be bullish or doji
            if self.is_bearish_candle(candle):
                return None

            body_size = characteristics["body_sizes"][0]
            lower_shadow = self.calculate_lower_shadow(data, index)
            upper_shadow = self.calculate_upper_shadow(data, index)
            total_range = candle["high"].item() - candle["low"].item()

            if total_range == 0:
                return None

            # Hammer characteristics:
            # - Lower shadow at least 2x body size
            # - Upper shadow small (less than body)
            # - Body in upper third of total range
            if not (
                lower_shadow >= 2 * body_size
                and upper_shadow <= body_size
                and candle["close"].item()
                > (total_range * 2 / 3 + candle["low"].item())
            ):
                return None

            # Calculate dynamic confidence based on hammer quality
            # Higher confidence for longer lower shadows and smaller upper shadows
            shadow_ratio = lower_shadow / max(
                upper_shadow, 0.001
            )  # Avoid division by zero
            body_position = (
                candle["close"] - candle["low"]
            ) / total_range  # How high in range

            pattern_factors = {
                "trend_strength": self._calculate_trend_strength(data, index, 5),
                "candle_size": self._calculate_candle_size_confidence(
                    data, index, 0.6
                ),  # Medium body expected
                "price_movement": self._calculate_price_movement_confidence(
                    data, index, 0.5
                ),  # Moderate movement
                "pattern_completeness": min(1.0, shadow_ratio / 3.0)
                * min(1.0, body_position),  # Shadow ratio and position
                "volume": self._calculate_volume_confidence(
                    data, index, 1.2
                ),  # Expect volume increase for reversal
            }

            confidence = self._calculate_pattern_confidence(
                data,
                index,
                pattern_factors,
                base_confidence=self.get_base_confidence_for_pattern(self.pattern_type),
            )

            # Apply multi-timeframe alignment adjustment if data is available
            if multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_alignment(
                    data, index, multi_timeframe_data, self.pattern_type
                )
                confidence = min(1.0, confidence * mtf_confidence)

            return SignalResult(
                signal_type="hammer",
                strength=confidence,
                direction=confidence,  # Bullish signal strength
                description="Hammer: Bullish reversal pattern",
                timestamp=data.index[index],
                confidence=confidence,
                metadata={"pattern": "hammer", "confidence": confidence},
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Hammer pattern: {e}")
            return None


class HangingManRecognizer(CandlestickPatternRecognizer):
    """Recognizes Hanging Man pattern.
    首吊り線

    A single-candle bearish reversal pattern with long lower shadow.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "hanging_man"
        self.logger = logging.getLogger(__name__)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Hanging Man pattern at the given index."""
        try:
            # Validate inputs using common method
            index = self.validate_recognition_inputs(data, index, required_length=1)

            # Check for uptrend
            if not self._is_uptrend(data, index, lookback=5):
                return None

            # Analyze candle characteristics using common method
            characteristics = self.analyze_multiple_candle_characteristics(
                data, [index]
            )

            candle = data.iloc[index]

            # Must be bearish or doji
            if self.is_bullish_candle(candle):
                return None

            body_size = characteristics["body_sizes"][0]
            lower_shadow = self.calculate_lower_shadow(data, index)
            upper_shadow = self.calculate_upper_shadow(data, index)
            total_range = candle["high"].item() - candle["low"].item()

            if total_range == 0:
                return None

            # Hanging Man characteristics (similar to hammer but bearish):
            # - Lower shadow at least 2x body size
            # - Upper shadow small
            # - Body in upper third of total range
            if not (
                lower_shadow >= 2 * body_size
                and upper_shadow <= body_size
                and candle["open"].item() > (total_range * 2 / 3 + candle["low"].item())
            ):
                return None

            # Calculate dynamic confidence based on hanging man quality
            # Higher confidence for longer lower shadows and smaller upper shadows
            shadow_ratio = lower_shadow / max(
                upper_shadow, 0.001
            )  # Avoid division by zero
            body_position = (
                candle["open"] - candle["low"]
            ) / total_range  # How high in range

            pattern_factors = {
                "trend_strength": self._calculate_trend_strength(data, index, 5),
                "candle_size": self._calculate_candle_size_confidence(
                    data, index, 0.6
                ),  # Medium body expected
                "price_movement": self._calculate_price_movement_confidence(
                    data, index, 0.5
                ),  # Moderate movement
                "pattern_completeness": min(1.0, shadow_ratio / 3.0)
                * min(1.0, body_position),  # Shadow ratio and position
                "volume": self._calculate_volume_confidence(
                    data, index, 1.2
                ),  # Expect volume increase for reversal
            }

            confidence = self._calculate_pattern_confidence(
                data,
                index,
                pattern_factors,
                base_confidence=self.get_base_confidence_for_pattern(self.pattern_type),
            )

            # Apply multi-timeframe alignment adjustment if data is available
            if multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_alignment(
                    data, index, multi_timeframe_data, self.pattern_type
                )
                confidence = min(1.0, confidence * mtf_confidence)

            return SignalResult(
                signal_type="hanging_man",
                strength=confidence,
                direction=-confidence,  # Bearish signal strength
                description="Hanging Man: Bearish reversal pattern",
                timestamp=data.index[index],
                confidence=confidence,
                metadata={"pattern": "hanging_man", "confidence": confidence},
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Hanging Man pattern: {e}")
            return None


class ThreeBlackCrowsRecognizer(CandlestickPatternRecognizer):
    """Recognizes Three Black Crows pattern.
    # 三羽烏（黒三兵）

    A three-candle bearish reversal pattern with three consecutive bearish candles.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "three_black_crows"
        self.logger = logging.getLogger(__name__)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Three Black Crows pattern at the given index."""
        try:
            # Validate inputs using common method
            index = self.validate_recognition_inputs(data, index, required_length=3)

            # Check for uptrend
            if not self._is_uptrend(data, index, lookback=5):
                return None

            # Pattern: Three consecutive bearish candles
            indices_to_check = [index - i for i in range(3)]
            expected_directions = ["bearish", "bearish", "bearish"]

            # Validate pattern structure using common method
            if not self.validate_pattern_structure(
                data, indices_to_check, expected_directions
            ):
                return None

            # Analyze candle characteristics using common method
            characteristics = self.analyze_multiple_candle_characteristics(
                data, indices_to_check
            )

            candles = [data.iloc[idx] for idx in indices_to_check]

            # Additional validation: each candle should open near previous close and close near low
            for i in range(1, 3):
                prev_close = candles[i - 1]["close"]
                curr_open = candles[i]["open"]
                curr_close = candles[i]["close"]
                curr_low = candles[i]["low"]

                # Opening near previous close
                if abs(curr_open.item() - prev_close.item()) / prev_close.item() > 0.01:
                    return None

                # Closing near low (bearish pressure)
                body_bottom = min(curr_open.item(), curr_close.item())
                if (body_bottom - curr_low.item()) / (
                    candles[i]["high"].item() - curr_low.item()
                ) > 0.3:
                    return None

            # Progressive lower closes
            if not (
                candles[0]["close"].item()
                > candles[1]["close"].item()
                > candles[2]["close"].item()
            ):
                return None

            # Calculate dynamic confidence based on pattern quality
            # Check how well each candle fits the bearish pattern
            bearish_scores = [
                1.0 if self.is_bearish_candle(candles[i]) else 0.5 for i in range(3)
            ]
            progressive_close_scores = [
                1.0
                if candles[i]["close"].item() > candles[i + 1]["close"].item()
                else 0.5
                for i in range(2)
            ]

            pattern_completeness = (
                sum(bearish_scores) / len(bearish_scores) * 0.7
                + sum(progressive_close_scores) / len(progressive_close_scores) * 0.3
            )

            pattern_factors = {
                "trend_strength": self._calculate_trend_strength(data, index, 5),
                "candle_size": self._calculate_candle_size_confidence(
                    data, index, 0.8
                ),  # Large candles expected
                "price_movement": self._calculate_price_movement_confidence(
                    data, index, 1.0
                ),  # Significant movement
                "pattern_completeness": pattern_completeness,  # How well candles fit the pattern
                "volume": self._calculate_volume_confidence(
                    data, index, 1.2
                ),  # Expect volume increase for reversal
            }

            confidence = self._calculate_pattern_confidence(
                data,
                index,
                pattern_factors,
                base_confidence=self.get_base_confidence_for_pattern(self.pattern_type),
            )

            # Apply multi-timeframe alignment adjustment if data is available
            if multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_alignment(
                    data, index, multi_timeframe_data, self.pattern_type
                )
                confidence = min(1.0, confidence * mtf_confidence)

            return SignalResult(
                signal_type="three_black_crows",
                strength=confidence,
                direction=-confidence,  # Bearish signal strength
                description="Three Black Crows: Bearish reversal pattern",
                timestamp=data.index[index],
                confidence=confidence,
                metadata={"pattern": "three_black_crows", "confidence": confidence},
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Three Black Crows pattern: {e}")
            return None


class ThreeWhiteSoldiersRecognizer(CandlestickPatternRecognizer):
    """Recognizes Three White Soldiers pattern.
    赤三兵

    A three-candle bullish reversal pattern with three consecutive bullish candles.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "three_white_soldiers"
        self.logger = logging.getLogger(__name__)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Three White Soldiers pattern at the given index."""
        try:
            # Validate inputs using common method
            index = self.validate_recognition_inputs(data, index, required_length=3)

            # Check for downtrend
            if not self._is_downtrend(data, index, lookback=5):
                return None

            # Pattern: Three consecutive bullish candles
            indices_to_check = [index - i for i in range(3)]
            expected_directions = ["bullish", "bullish", "bullish"]

            # Validate pattern structure using common method
            if not self.validate_pattern_structure(
                data, indices_to_check, expected_directions
            ):
                return None

            # Analyze candle characteristics using common method
            characteristics = self.analyze_multiple_candle_characteristics(
                data, indices_to_check
            )

            candles = [data.iloc[idx] for idx in indices_to_check]

            # Additional validation: each candle should open near previous close and close near high
            for i in range(1, 3):
                prev_close = candles[i - 1]["close"]
                curr_open = candles[i]["open"]
                curr_close = candles[i]["close"]
                curr_high = candles[i]["high"]

                # Opening near previous close
                if abs(curr_open.item() - prev_close.item()) / prev_close.item() > 0.01:
                    return None

                # Closing near high (bullish pressure)
                body_top = max(curr_open.item(), curr_close.item())
                if (curr_high.item() - body_top) / (
                    curr_high.item() - candles[i]["low"].item()
                ) > 0.3:
                    return None

            # Progressive higher closes
            if not (
                candles[0]["close"].item()
                < candles[1]["close"].item()
                < candles[2]["close"].item()
            ):
                return None

            # Calculate dynamic confidence based on pattern quality
            # Check how well each candle fits the bullish pattern
            bullish_scores = [
                1.0 if self.is_bullish_candle(candles[i]) else 0.5 for i in range(3)
            ]
            progressive_close_scores = [
                1.0
                if candles[i]["close"].item() < candles[i + 1]["close"].item()
                else 0.5
                for i in range(2)
            ]

            pattern_completeness = (
                sum(bullish_scores) / len(bullish_scores) * 0.7
                + sum(progressive_close_scores) / len(progressive_close_scores) * 0.3
            )

            pattern_factors = {
                "trend_strength": self._calculate_trend_strength(data, index, 5),
                "candle_size": self._calculate_candle_size_confidence(
                    data, index, 0.8
                ),  # Large candles expected
                "price_movement": self._calculate_price_movement_confidence(
                    data, index, 1.0
                ),  # Significant movement
                "pattern_completeness": pattern_completeness,  # How well candles fit the pattern
                "volume": self._calculate_volume_confidence(
                    data, index, 1.2
                ),  # Expect volume increase for reversal
            }

            confidence = self._calculate_pattern_confidence(
                data,
                index,
                pattern_factors,
                base_confidence=self.get_base_confidence_for_pattern(self.pattern_type),
            )

            # Apply multi-timeframe alignment adjustment if data is available
            if multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_alignment(
                    data, index, multi_timeframe_data, self.pattern_type
                )
                confidence = min(1.0, confidence * mtf_confidence)

            return SignalResult(
                signal_type="three_white_soldiers",
                strength=confidence,
                direction=confidence,  # Bullish signal strength
                description="Three White Soldiers: Bullish reversal pattern",
                timestamp=data.index[index],
                confidence=confidence,
                metadata={"pattern": "three_white_soldiers", "confidence": confidence},
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Three White Soldiers pattern: {e}")
            return None


class RisingThreeMethodsRecognizer(CandlestickPatternRecognizer):
    """Recognizes Rising Three Methods pattern.
    上げ三法

    A five-candle bullish continuation pattern.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "rising_three_methods"
        self.logger = logging.getLogger(__name__)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Rising Three Methods pattern at the given index."""
        try:
            # Validate inputs using common method
            index = self.validate_recognition_inputs(data, index, required_length=5)

            # Check for uptrend
            if not self._is_uptrend(data, index, lookback=5):
                return None

            # Pattern: Large bullish, three small candles, large bullish
            indices_to_check = [index - i for i in range(5)]
            expected_directions = ["bullish", "any", "any", "any", "bullish"]

            # Validate pattern structure using common method
            if not self.validate_pattern_structure(
                data, indices_to_check, expected_directions
            ):
                return None

            # Analyze candle characteristics using common method
            characteristics = self.analyze_multiple_candle_characteristics(
                data, indices_to_check
            )

            candles = [data.iloc[idx] for idx in indices_to_check]

            # Check size characteristics: first and last large, middle three small
            if not (
                characteristics["body_sizes"][4] > characteristics["avg_body_size"]
                and characteristics["body_sizes"][0]
                > characteristics["avg_body_size"]  # First large
                and characteristics["body_sizes"][1]
                < characteristics["avg_body_size"] * 0.5  # Last large
                and characteristics["body_sizes"][2]
                < characteristics["avg_body_size"] * 0.5  # Second small
                and characteristics["body_sizes"][3]  # Third small
                < characteristics["avg_body_size"] * 0.5  # Fourth small
            ):
                return None

            # Three middle candles should be contained within first candle's range
            first_high = candles[4]["high"].item()
            first_low = candles[4]["low"].item()

            for i in range(1, 4):
                if (
                    candles[i]["high"].item() > first_high
                    or candles[i]["low"].item() < first_low
                ):
                    return None

            # Calculate dynamic confidence based on pattern quality
            pattern_factors = {
                "trend_strength": self._calculate_trend_strength(data, index, 5),
                "candle_size": self._calculate_candle_size_confidence(
                    data, index, 0.8
                ),  # Large outer candles expected
                "price_movement": self._calculate_price_movement_confidence(
                    data, index, 1.0
                ),  # Significant movement
                "pattern_completeness": min(
                    1.0,
                    characteristics["body_sizes"][0]
                    / characteristics["avg_body_size"]
                    * 0.5,
                ),  # Large final candle
                "volume": self._calculate_volume_confidence(
                    data, index, 1.2
                ),  # Expect volume increase for reversal
            }

            confidence = self._calculate_pattern_confidence(
                data,
                index,
                pattern_factors,
                base_confidence=self.get_base_confidence_for_pattern(self.pattern_type),
            )

            # Apply multi-timeframe alignment adjustment if data is available
            if multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_alignment(
                    data, index, multi_timeframe_data, self.pattern_type
                )
                confidence = min(1.0, confidence * mtf_confidence)

            return SignalResult(
                signal_type="rising_three_methods",
                strength=confidence,
                direction=confidence,  # Bullish signal strength
                description="Rising Three Methods: Bullish continuation pattern",
                timestamp=data.index[index],
                confidence=confidence,
                metadata={"pattern": "rising_three_methods", "confidence": confidence},
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Rising Three Methods pattern: {e}")
            return None


class BullishEngulfingRecognizer(CandlestickPatternRecognizer):
    """Recognizes Bullish Engulfing pattern.
    陽線はらみ足
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "bullish_engulfing"
        self.logger = logging.getLogger(__name__)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Bullish Engulfing pattern at the given index."""
        try:
            # Validate inputs using common method
            index = self.validate_recognition_inputs(data, index, required_length=2)

            # Check for downtrend
            if not self._is_downtrend(data, index, lookback=3):
                return None

            # Pattern: Bearish followed by bullish engulfing
            indices_to_check = [index - 1, index]
            expected_directions = ["bearish", "bullish"]

            # Validate pattern structure using common method
            if not self.validate_pattern_structure(
                data, indices_to_check, expected_directions
            ):
                return None

            # Analyze candle characteristics using common method
            characteristics = self.analyze_multiple_candle_characteristics(
                data, indices_to_check
            )

            current = data.iloc[index]
            previous = data.iloc[index - 1]

            # Current candle must engulf previous candle completely
            current_open = current["open"]
            current_close = current["close"]
            prev_open = previous["open"]
            prev_close = previous["close"]

            # Bullish engulfing: current open <= prev_close and current_close >= prev_open
            if not (
                current_open.item() <= prev_close.item()
                and current_close.item() >= prev_open.item()
            ):
                return None

            prev_body_size = characteristics["body_sizes"][0]
            current_body_size = characteristics["body_sizes"][1]

            if prev_body_size == 0 or current_body_size == 0:
                return None

            engulfing_ratio = float(current_body_size / prev_body_size)
            base_strength = float(min(0.9, 0.6 + (engulfing_ratio - 1) * 0.2))

            # Calculate dynamic confidence based on engulfing quality and trend
            pattern_factors = {
                "trend_strength": self._calculate_trend_strength(data, index, 5),
                "candle_size": self._calculate_candle_size_confidence(
                    data, index, 0.8
                ),  # Large candles expected
                "price_movement": self._calculate_price_movement_confidence(
                    data, index, 1.2
                ),  # Strong movement expected
                "pattern_completeness": min(
                    1.0, engulfing_ratio / 2.0
                ),  # How complete the engulfing is
                "volume": self._calculate_volume_confidence(
                    data, index, 1.2
                ),  # Expect volume increase for reversal
            }

            confidence = self._calculate_pattern_confidence(
                data,
                index,
                pattern_factors,
                base_confidence=self.get_base_confidence_for_pattern(self.pattern_type),
            )

            # Apply multi-timeframe alignment adjustment if data is available
            if multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_alignment(
                    data, index, multi_timeframe_data, self.pattern_type
                )
                confidence = min(1.0, confidence * mtf_confidence)

            return SignalResult(
                signal_type="bullish_engulfing",
                strength=confidence,
                direction=confidence,  # Bullish signal strength
                description="Bullish Engulfing: Strong reversal signal in downtrend",
                timestamp=data.index[index],
                confidence=confidence,
                metadata={
                    "pattern": "bullish_engulfing",
                    "confidence": confidence,
                    "engulfing_ratio": engulfing_ratio,
                },
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Bullish Engulfing pattern: {e}")
            return None


class BearishEngulfingRecognizer(CandlestickPatternRecognizer):
    """Recognizes Bearish Engulfing pattern.
    陰線はらみ足
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "bearish_engulfing"
        self.logger = logging.getLogger(__name__)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Bearish Engulfing pattern at the given index."""
        try:
            # Validate inputs using common method
            index = self.validate_recognition_inputs(data, index, required_length=2)

            # Check for uptrend
            if not self._is_uptrend(data, index, lookback=3):
                return None

            # Pattern: Bullish followed by bearish engulfing
            indices_to_check = [index - 1, index]
            expected_directions = ["bullish", "bearish"]

            # Validate pattern structure using common method
            if not self.validate_pattern_structure(
                data, indices_to_check, expected_directions
            ):
                return None

            # Analyze candle characteristics using common method
            characteristics = self.analyze_multiple_candle_characteristics(
                data, indices_to_check
            )

            current = data.iloc[index]
            previous = data.iloc[index - 1]

            # Current candle must engulf previous candle completely
            current_open = current["open"]
            current_close = current["close"]
            prev_open = previous["open"]
            prev_close = previous["close"]

            # Bearish engulfing: current open >= prev_close and current_close <= prev_open
            if not (
                current_open.item() >= prev_close.item()
                and current_close.item() <= prev_open.item()
            ):
                return None

            # Calculate strength based on engulfing ratio
            prev_body_size = characteristics["body_sizes"][0]
            current_body_size = characteristics["body_sizes"][1]

            if prev_body_size == 0:
                return None

            engulfing_ratio = current_body_size / prev_body_size
            base_strength = min(0.9, 0.6 + (engulfing_ratio - 1) * 0.2)

            # Calculate dynamic confidence based on engulfing quality and trend
            pattern_factors = {
                "trend_strength": self._calculate_trend_strength(data, index, 5),
                "candle_size": self._calculate_candle_size_confidence(
                    data, index, 0.8
                ),  # Large candles expected
                "price_movement": self._calculate_price_movement_confidence(
                    data, index, 1.2
                ),  # Strong movement expected
                "pattern_completeness": min(
                    1.0, engulfing_ratio / 2.0
                ),  # How complete the engulfing is
                "volume": self._calculate_volume_confidence(
                    data, index, 1.2
                ),  # Expect volume increase for reversal
            }

            confidence = self._calculate_pattern_confidence(
                data,
                index,
                pattern_factors,
                base_confidence=self.get_base_confidence_for_pattern(self.pattern_type),
            )

            # Apply multi-timeframe alignment adjustment if data is available
            if multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_alignment(
                    data, index, multi_timeframe_data, self.pattern_type
                )
                confidence = min(1.0, confidence * mtf_confidence)

            return SignalResult(
                signal_type="bearish_engulfing",
                strength=confidence,
                direction=-confidence,  # Bearish signal strength
                description="Bearish Engulfing: Strong reversal signal in uptrend",
                timestamp=data.index[index],
                confidence=confidence,
                metadata={
                    "pattern": "bearish_engulfing",
                    "confidence": confidence,
                    "engulfing_ratio": engulfing_ratio,
                },
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Bearish Engulfing pattern: {e}")
            return None


class PiercingPatternRecognizer(CandlestickPatternRecognizer):
    """Recognizes Piercing Pattern.
    差し込み線
    A two-candle bullish reversal pattern in a downtrend.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "piercing_pattern"
        self.logger = logging.getLogger(__name__)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Piercing Pattern at the given index."""
        try:
            # Validate inputs using common method
            index = self.validate_recognition_inputs(data, index, required_length=2)

            # Check for downtrend
            if not self._is_downtrend(data, index, lookback=3):
                return None

            # Pattern: Bearish followed by bullish piercing
            indices_to_check = [index - 1, index]
            expected_directions = ["bearish", "bullish"]

            # Validate pattern structure using common method
            if not self.validate_pattern_structure(
                data, indices_to_check, expected_directions
            ):
                return None

            # Analyze candle characteristics using common method
            characteristics = self.analyze_multiple_candle_characteristics(
                data, indices_to_check
            )

            current = data.iloc[index]
            previous = data.iloc[index - 1]

            prev_open = previous["open"]
            prev_close = previous["close"]
            current_open = current["open"]
            current_close = current["close"]

            # Piercing pattern: current open < prev_close and current_close > midpoint of prev body
            prev_midpoint = (prev_open + prev_close) / 2

            if not (
                current_open.item() < prev_close.item()
                and current_close.item() > prev_midpoint.item()
            ):
                return None

            # Calculate strength based on penetration depth
            prev_body_size = characteristics["body_sizes"][0]
            if prev_body_size == 0:
                return None
            penetration = ((current_close - prev_midpoint) / prev_body_size).item()

            base_strength = min(0.85, 0.5 + penetration * 0.4)

            # Calculate dynamic confidence based on penetration and trend
            pattern_factors = {
                "trend_strength": self._calculate_trend_strength(data, index, 5),
                "candle_size": self._calculate_candle_size_confidence(
                    data, index, 0.7
                ),  # Medium-large candles expected
                "price_movement": self._calculate_price_movement_confidence(
                    data, index, 0.8
                ),  # Moderate movement expected
                "pattern_completeness": min(
                    1.0, penetration
                ),  # How deep the penetration is
                "volume": self._calculate_volume_confidence(
                    data, index, 1.2
                ),  # Expect volume increase for reversal
            }

            confidence = self._calculate_pattern_confidence(
                data,
                index,
                pattern_factors,
                base_confidence=self.get_base_confidence_for_pattern(self.pattern_type),
            )

            # Apply multi-timeframe alignment adjustment if data is available
            if multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_alignment(
                    data, index, multi_timeframe_data, self.pattern_type
                )
                confidence = min(1.0, confidence * mtf_confidence)

            return SignalResult(
                signal_type="piercing_pattern",
                strength=confidence,
                direction=confidence,  # Bullish signal strength
                description="Piercing Pattern: Bullish reversal signal in downtrend",
                timestamp=data.index[index],
                confidence=confidence,
                metadata={
                    "pattern": "piercing_pattern",
                    "confidence": confidence,
                    "penetration": penetration,
                },
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Piercing Pattern: {e}")
            return None
