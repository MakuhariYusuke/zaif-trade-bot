"""
Candlestick Pattern Recognition Module

This module provides pattern recognition for traditional Japanese candlestick patterns
used in technical analysis for trading signals.
"""

from __future__ import annotations

import logging
from abc import ABC
from typing import TypedDict, cast

import pandas as pd

from ztb.types.common import ConfigSection

from .base import (
    CandlestickPatternRecognizer,
    MultiTimeframeData,
    SignalMetadata,
    SignalResult,
)


class CandleCharacteristics(TypedDict):
    """Structured characteristics returned by multi-candle analysis."""

    body_sizes: list[float]
    body_ratios: list[float]
    upper_shadow_ratios: list[float]
    lower_shadow_ratios: list[float]
    is_bullish: list[bool]
    is_bearish: list[bool]
    avg_body_size: float


PatternFactors = dict[str, float]


class _CandlestickPatternBase(CandlestickPatternRecognizer, ABC):
    """Shared helpers for candlestick recognizers in this module."""

    pattern_type: str = "candlestick_pattern"

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def _validate_index_with_trend(
        self,
        data: pd.DataFrame,
        index: int,
        required_length: int,
        lookback: int,
        expect_downtrend: bool,
        multi_timeframe_data: MultiTimeframeData | None,
    ) -> int | None:
        validated_index = self.validate_recognition_inputs(
            data,
            index,
            required_length=required_length,
            multi_timeframe_data=multi_timeframe_data,
        )

        trend_ok = (
            self._is_downtrend(data, validated_index, lookback)
            if expect_downtrend
            else self._is_uptrend(data, validated_index, lookback)
        )
        return validated_index if trend_ok else None

    def _analyze_characteristics(
        self, data: pd.DataFrame, indices: list[int]
    ) -> CandleCharacteristics:
        return cast(
            CandleCharacteristics,
            self.analyze_multiple_candle_characteristics(data, indices),
        )

    def _build_pattern_factors(
        self,
        data: pd.DataFrame,
        index: int,
        trend_lookback: int,
        candle_size_expected: float,
        price_movement_expected: float,
        pattern_completeness: float,
        volume_expected: float,
    ) -> PatternFactors:
        return {
            "trend_strength": self._calculate_trend_strength(data, index, trend_lookback),
            "candle_size": self._calculate_candle_size_confidence(
                data, index, candle_size_expected
            ),
            "price_movement": self._calculate_price_movement_confidence(
                data, index, price_movement_expected
            ),
            "pattern_completeness": max(0.0, min(1.0, pattern_completeness)),
            "volume": self._calculate_volume_confidence(data, index, volume_expected),
        }

    def _calculate_confidence(
        self,
        data: pd.DataFrame,
        index: int,
        pattern_factors: PatternFactors,
        multi_timeframe_data: MultiTimeframeData | None,
    ) -> float:
        confidence = self._calculate_pattern_confidence(
            data,
            index,
            pattern_factors,
            base_confidence=self.get_base_confidence_for_pattern(self.pattern_type),
        )

        if multi_timeframe_data:
            mtf_confidence = self._analyze_multi_timeframe_alignment(
                data, index, multi_timeframe_data, self.pattern_type
            )
            confidence = min(1.0, confidence * mtf_confidence)

        return confidence

    def _create_signal_result(
        self,
        data: pd.DataFrame,
        index: int,
        confidence: float,
        direction_sign: float,
        description: str,
        metadata: SignalMetadata | None = None,
        direction_scales_with_confidence: bool = True,
        signal_type: str | None = None,
    ) -> SignalResult:
        direction = (
            direction_sign * confidence
            if direction_scales_with_confidence
            else direction_sign
        )

        result_metadata: SignalMetadata = metadata or cast(
            SignalMetadata,
            {
                "pattern": self.pattern_type,
                "confidence": confidence,
            },
        )

        return SignalResult(
            signal_type=signal_type or self.pattern_type,
            strength=confidence,
            direction=direction,
            description=description,
            timestamp=data.index[index],
            confidence=confidence,
            metadata=result_metadata,
        )


class _ThreeCandleStarBase(_CandlestickPatternBase, ABC):
    """Shared implementation for Morning/Evening Star recognizers."""

    trend_requires_downtrend: bool = True
    expected_directions: tuple[str, str, str] = ("bearish", "any", "bullish")
    midpoint_requires_close_above: bool = True
    signal_direction: float = 1.0
    signal_description: str = ""

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        try:
            validated_index = self._validate_index_with_trend(
                data,
                index,
                required_length=3,
                lookback=5,
                expect_downtrend=self.trend_requires_downtrend,
                multi_timeframe_data=multi_timeframe_data,
            )
            if validated_index is None:
                return None

            indices_to_check = [validated_index - 2, validated_index - 1, validated_index]
            if not self.validate_pattern_structure(
                data, indices_to_check, list(self.expected_directions)
            ):
                return None

            characteristics = self._analyze_characteristics(data, indices_to_check)
            avg_body_size = characteristics["avg_body_size"]
            if avg_body_size <= 0:
                return None

            first = data.iloc[validated_index - 2]
            third = data.iloc[validated_index]
            first_midpoint = (float(first["open"]) + float(first["close"])) / 2.0
            third_close = float(third["close"])

            if self.midpoint_requires_close_above:
                if third_close <= first_midpoint:
                    return None
            elif third_close >= first_midpoint:
                return None

            body_sizes = characteristics["body_sizes"]
            if not (
                body_sizes[0] > avg_body_size
                and body_sizes[2] > avg_body_size
                and body_sizes[1] < avg_body_size * 0.5
            ):
                return None

            pattern_factors = self._build_pattern_factors(
                data,
                validated_index,
                trend_lookback=5,
                candle_size_expected=0.8,
                price_movement_expected=1.0,
                pattern_completeness=min(1.0, (body_sizes[1] / avg_body_size) * 2.0),
                volume_expected=1.2,
            )

            confidence = self._calculate_confidence(
                data,
                validated_index,
                pattern_factors,
                multi_timeframe_data,
            )

            return self._create_signal_result(
                data,
                validated_index,
                confidence,
                direction_sign=self.signal_direction,
                description=self.signal_description,
                metadata=cast(
                    SignalMetadata,
                    {
                        "pattern": self.pattern_type,
                        "confidence": confidence,
                    },
                ),
                direction_scales_with_confidence=False,
            )

        except Exception as e:
            self.logger.error(f"Error recognizing {self.pattern_type} pattern: {e}")
            return None


class _LongShadowReversalBase(_CandlestickPatternBase, ABC):
    """Shared implementation for Hammer/Hanging Man recognizers."""

    trend_requires_downtrend: bool = True
    invalid_candle_direction: str = "bearish"  # "bearish" for hammer, "bullish" for hanging man
    body_anchor_key: str = "close"
    signal_direction: float = 1.0
    signal_description: str = ""

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        try:
            validated_index = self._validate_index_with_trend(
                data,
                index,
                required_length=1,
                lookback=5,
                expect_downtrend=self.trend_requires_downtrend,
                multi_timeframe_data=multi_timeframe_data,
            )
            if validated_index is None:
                return None

            characteristics = self._analyze_characteristics(data, [validated_index])
            candle = data.iloc[validated_index]

            if self.invalid_candle_direction == "bearish" and self.is_bearish_candle(candle):
                return None
            if self.invalid_candle_direction == "bullish" and self.is_bullish_candle(candle):
                return None

            body_size = characteristics["body_sizes"][0]
            lower_shadow = self.calculate_lower_shadow(data, validated_index)
            upper_shadow = self.calculate_upper_shadow(data, validated_index)
            low_price = float(candle["low"])
            total_range = float(candle["high"]) - low_price
            if total_range <= 0:
                return None

            body_anchor = float(candle[self.body_anchor_key])
            if not (
                lower_shadow >= 2.0 * body_size
                and upper_shadow <= body_size
                and body_anchor > (total_range * 2.0 / 3.0 + low_price)
            ):
                return None

            shadow_ratio = lower_shadow / max(upper_shadow, 0.001)
            body_position = (body_anchor - low_price) / total_range

            pattern_factors = self._build_pattern_factors(
                data,
                validated_index,
                trend_lookback=5,
                candle_size_expected=0.6,
                price_movement_expected=0.5,
                pattern_completeness=min(1.0, shadow_ratio / 3.0)
                * min(1.0, body_position),
                volume_expected=1.2,
            )

            confidence = self._calculate_confidence(
                data,
                validated_index,
                pattern_factors,
                multi_timeframe_data,
            )

            return self._create_signal_result(
                data,
                validated_index,
                confidence,
                direction_sign=self.signal_direction,
                description=self.signal_description,
                metadata=cast(
                    SignalMetadata,
                    {
                        "pattern": self.pattern_type,
                        "confidence": confidence,
                    },
                ),
            )

        except Exception as e:
            self.logger.error(f"Error recognizing {self.pattern_type} pattern: {e}")
            return None


class _ThreeConsecutiveReversalBase(_CandlestickPatternBase, ABC):
    """Shared implementation for Three Black Crows / Three White Soldiers."""

    trend_requires_downtrend: bool = False
    expected_direction: str = "bearish"
    closes_should_increase: bool = False
    signal_direction: float = -1.0
    signal_description: str = ""

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        try:
            validated_index = self._validate_index_with_trend(
                data,
                index,
                required_length=3,
                lookback=5,
                expect_downtrend=self.trend_requires_downtrend,
                multi_timeframe_data=multi_timeframe_data,
            )
            if validated_index is None:
                return None

            indices_to_check = [validated_index - 2, validated_index - 1, validated_index]
            expected_directions = [self.expected_direction, self.expected_direction, self.expected_direction]
            if not self.validate_pattern_structure(
                data, indices_to_check, expected_directions
            ):
                return None

            characteristics = self._analyze_characteristics(data, indices_to_check)
            candles = [data.iloc[idx] for idx in indices_to_check]

            for i in range(1, 3):
                prev_close = float(candles[i - 1]["close"])
                curr_open = float(candles[i]["open"])
                curr_close = float(candles[i]["close"])
                curr_low = float(candles[i]["low"])
                curr_high = float(candles[i]["high"])

                open_gap_ratio = abs(curr_open - prev_close) / max(abs(prev_close), 1e-9)
                if open_gap_ratio > 0.01:
                    return None

                total_range = curr_high - curr_low
                if total_range <= 0:
                    return None

                if self.closes_should_increase:
                    body_top = max(curr_open, curr_close)
                    if (curr_high - body_top) / total_range > 0.3:
                        return None
                else:
                    body_bottom = min(curr_open, curr_close)
                    if (body_bottom - curr_low) / total_range > 0.3:
                        return None

            closes = [float(candle["close"]) for candle in candles]
            if self.closes_should_increase:
                if not (closes[0] < closes[1] < closes[2]):
                    return None
            elif not (closes[0] > closes[1] > closes[2]):
                return None

            if self.closes_should_increase:
                direction_scores = [1.0 if flag else 0.5 for flag in characteristics["is_bullish"]]
                progression_scores = [
                    1.0 if closes[i] < closes[i + 1] else 0.5 for i in range(2)
                ]
            else:
                direction_scores = [1.0 if flag else 0.5 for flag in characteristics["is_bearish"]]
                progression_scores = [
                    1.0 if closes[i] > closes[i + 1] else 0.5 for i in range(2)
                ]

            pattern_completeness = (
                (sum(direction_scores) / len(direction_scores)) * 0.7
                + (sum(progression_scores) / len(progression_scores)) * 0.3
            )

            pattern_factors = self._build_pattern_factors(
                data,
                validated_index,
                trend_lookback=5,
                candle_size_expected=0.8,
                price_movement_expected=1.0,
                pattern_completeness=pattern_completeness,
                volume_expected=1.2,
            )

            confidence = self._calculate_confidence(
                data,
                validated_index,
                pattern_factors,
                multi_timeframe_data,
            )

            return self._create_signal_result(
                data,
                validated_index,
                confidence,
                direction_sign=self.signal_direction,
                description=self.signal_description,
                metadata=cast(
                    SignalMetadata,
                    {
                        "pattern": self.pattern_type,
                        "confidence": confidence,
                    },
                ),
            )

        except Exception as e:
            self.logger.error(f"Error recognizing {self.pattern_type} pattern: {e}")
            return None


class _EngulfingPatternBase(_CandlestickPatternBase, ABC):
    """Shared implementation for bullish/bearish engulfing recognizers."""

    trend_requires_downtrend: bool = True
    expected_directions: tuple[str, str] = ("bearish", "bullish")
    is_bullish_pattern: bool = True
    signal_direction: float = 1.0
    signal_description: str = ""

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        try:
            validated_index = self._validate_index_with_trend(
                data,
                index,
                required_length=2,
                lookback=3,
                expect_downtrend=self.trend_requires_downtrend,
                multi_timeframe_data=multi_timeframe_data,
            )
            if validated_index is None:
                return None

            indices_to_check = [validated_index - 1, validated_index]
            if not self.validate_pattern_structure(
                data, indices_to_check, list(self.expected_directions)
            ):
                return None

            characteristics = self._analyze_characteristics(data, indices_to_check)
            previous = data.iloc[validated_index - 1]
            current = data.iloc[validated_index]

            prev_open = float(previous["open"])
            prev_close = float(previous["close"])
            curr_open = float(current["open"])
            curr_close = float(current["close"])

            if self.is_bullish_pattern:
                engulfing_valid = curr_open <= prev_close and curr_close >= prev_open
            else:
                engulfing_valid = curr_open >= prev_close and curr_close <= prev_open
            if not engulfing_valid:
                return None

            prev_body_size = characteristics["body_sizes"][0]
            curr_body_size = characteristics["body_sizes"][1]
            if prev_body_size <= 0 or curr_body_size <= 0:
                return None

            engulfing_ratio = curr_body_size / prev_body_size
            pattern_factors = self._build_pattern_factors(
                data,
                validated_index,
                trend_lookback=5,
                candle_size_expected=0.8,
                price_movement_expected=1.2,
                pattern_completeness=min(1.0, engulfing_ratio / 2.0),
                volume_expected=1.2,
            )

            confidence = self._calculate_confidence(
                data,
                validated_index,
                pattern_factors,
                multi_timeframe_data,
            )

            return self._create_signal_result(
                data,
                validated_index,
                confidence,
                direction_sign=self.signal_direction,
                description=self.signal_description,
                metadata=cast(
                    SignalMetadata,
                    {
                        "pattern": self.pattern_type,
                        "confidence": confidence,
                        "engulfing_ratio": engulfing_ratio,
                    },
                ),
            )

        except Exception as e:
            self.logger.error(f"Error recognizing {self.pattern_type} pattern: {e}")
            return None


class SakataFiveMethodsRecognizer(_CandlestickPatternBase):
    """Recognizes Sakata's Five Methods pattern.
    酒田五法

    This is a complex multi-candle pattern involving trend continuation
    with specific candle arrangements.
    """

    pattern_type = "sakata_five_methods"

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(config)
        if "lookback_period" not in self.config:
            self.config["lookback_period"] = 10

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize Sakata's Five Methods pattern at the given index."""
        try:
            validated_index = self._validate_index_with_trend(
                data,
                index,
                required_length=15,
                lookback=5,
                expect_downtrend=False,
                multi_timeframe_data=multi_timeframe_data,
            )
            if validated_index is None:
                return None

            candles = [data.iloc[validated_index - i] for i in range(5)]
            expected_directions = ["any", "bullish", "any", "bullish", "any"]
            indices_to_check = [validated_index - i for i in range(5)]
            if not self.validate_pattern_structure(
                data, indices_to_check, expected_directions
            ):
                return None

            size_checks: list[bool] = []
            for i, candle in enumerate(candles):
                if i % 2 == 0:
                    size_checks.append(self._is_small_candle(candle))
                else:
                    size_checks.append(self._is_large_candle(candle))
            if not all(size_checks):
                return None

            characteristics = self._analyze_characteristics(data, indices_to_check)
            body_sizes = characteristics["body_sizes"]
            if not (
                body_sizes[3] > body_sizes[1] and body_sizes[1] > body_sizes[3] * 0.7
            ):
                return None

            confidence = self._calculate_sakata_confidence(
                data,
                validated_index,
                characteristics,
                multi_timeframe_data,
            )

            return self._create_signal_result(
                data,
                validated_index,
                confidence,
                direction_sign=1.0,
                description="Sakata's Five Methods: Complex bullish continuation pattern",
                metadata=cast(
                    SignalMetadata,
                    {
                        "pattern": self.pattern_type,
                        "trend": "uptrend",
                        "avg_body_size": characteristics.get("avg_body_size", 0.0),
                        "body_sizes": list(characteristics.get("body_sizes", [])),
                        "momentum_increase": True,
                    },
                ),
                direction_scales_with_confidence=False,
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Sakata's Five Methods pattern: {e}")
            return None

    def _calculate_sakata_confidence(
        self,
        data: pd.DataFrame,
        index: int,
        characteristics: CandleCharacteristics,
        multi_timeframe_data: MultiTimeframeData | None,
    ) -> float:
        """Calculate confidence score for Sakata's Five Methods pattern."""
        try:
            small_candle_scores: list[float] = []
            large_candle_scores: list[float] = []
            avg_body_size = characteristics["avg_body_size"]

            for i in range(5):
                if i % 2 == 0:
                    is_small = characteristics["body_sizes"][i] < avg_body_size * 0.5
                    small_candle_scores.append(1.0 if is_small else 0.5)
                else:
                    is_large = characteristics["body_sizes"][i] > avg_body_size * 1.5
                    large_candle_scores.append(1.0 if is_large else 0.5)

            pattern_completeness = (
                (sum(small_candle_scores) / len(small_candle_scores)) * 0.6
                + (sum(large_candle_scores) / len(large_candle_scores)) * 0.4
            )

            pattern_factors = self._build_pattern_factors(
                data,
                index,
                trend_lookback=10,
                candle_size_expected=0.7,
                price_movement_expected=0.8,
                pattern_completeness=pattern_completeness,
                volume_expected=1.2,
            )

            return self._calculate_confidence(
                data,
                index,
                pattern_factors,
                multi_timeframe_data,
            )

        except Exception as e:
            self.logger.error(
                f"Error calculating confidence for Sakata's Five Methods: {e}"
            )
            return 0.5

    def _is_small_candle(self, candle: pd.Series) -> bool:
        """Check if candle has small body relative to recent volatility."""
        body_size = float(abs(candle["close"] - candle["open"]))
        total_range = float(candle["high"] - candle["low"])
        return (body_size / total_range) < 0.3 if total_range > 0 else False

    def _is_large_candle(self, candle: pd.Series) -> bool:
        """Check if candle has large body relative to recent volatility."""
        body_size = float(abs(candle["close"] - candle["open"]))
        total_range = float(candle["high"] - candle["low"])
        return (body_size / total_range) >= 0.6 if total_range > 0 else False


class MorningStarRecognizer(_ThreeCandleStarBase):
    """Recognizes Morning Star pattern.
    A three-candle bullish reversal pattern: large bearish, small, large bullish.
    """

    pattern_type = "morning_star"
    trend_requires_downtrend = True
    expected_directions = ("bearish", "any", "bullish")
    midpoint_requires_close_above = True
    signal_direction = 1.0
    signal_description = "Morning Star: Bullish reversal pattern"


class EveningStarRecognizer(_ThreeCandleStarBase):
    """Recognizes Evening Star pattern.
    宵の明星

    A three-candle bearish reversal pattern: large bullish, small, large bearish.
    """

    pattern_type = "evening_star"
    trend_requires_downtrend = False
    expected_directions = ("bullish", "any", "bearish")
    midpoint_requires_close_above = False
    signal_direction = -1.0
    signal_description = "Evening Star: Bearish reversal pattern"


class HammerRecognizer(_LongShadowReversalBase):
    """Recognizes Hammer pattern.
    捨て子底

    A single-candle bullish reversal pattern with long lower shadow.
    """

    pattern_type = "hammer"
    trend_requires_downtrend = True
    invalid_candle_direction = "bearish"
    body_anchor_key = "close"
    signal_direction = 1.0
    signal_description = "Hammer: Bullish reversal pattern"


class HangingManRecognizer(_LongShadowReversalBase):
    """Recognizes Hanging Man pattern.
    首吊り線

    A single-candle bearish reversal pattern with long lower shadow.
    """

    pattern_type = "hanging_man"
    trend_requires_downtrend = False
    invalid_candle_direction = "bullish"
    body_anchor_key = "open"
    signal_direction = -1.0
    signal_description = "Hanging Man: Bearish reversal pattern"


class ThreeBlackCrowsRecognizer(_ThreeConsecutiveReversalBase):
    """Recognizes Three Black Crows pattern.
    三羽烏（黒三兵）

    A three-candle bearish reversal pattern with three consecutive bearish candles.
    """

    pattern_type = "three_black_crows"
    trend_requires_downtrend = False
    expected_direction = "bearish"
    closes_should_increase = False
    signal_direction = -1.0
    signal_description = "Three Black Crows: Bearish reversal pattern"


class ThreeWhiteSoldiersRecognizer(_ThreeConsecutiveReversalBase):
    """Recognizes Three White Soldiers pattern.
    赤三兵

    A three-candle bullish reversal pattern with three consecutive bullish candles.
    """

    pattern_type = "three_white_soldiers"
    trend_requires_downtrend = True
    expected_direction = "bullish"
    closes_should_increase = True
    signal_direction = 1.0
    signal_description = "Three White Soldiers: Bullish reversal pattern"


class RisingThreeMethodsRecognizer(_CandlestickPatternBase):
    """Recognizes Rising Three Methods pattern.
    上げ三法

    A five-candle bullish continuation pattern.
    """

    pattern_type = "rising_three_methods"

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize Rising Three Methods pattern at the given index."""
        try:
            validated_index = self._validate_index_with_trend(
                data,
                index,
                required_length=5,
                lookback=5,
                expect_downtrend=False,
                multi_timeframe_data=multi_timeframe_data,
            )
            if validated_index is None:
                return None

            indices_to_check = [validated_index - i for i in range(5)]
            expected_directions = ["bullish", "any", "any", "any", "bullish"]
            if not self.validate_pattern_structure(
                data, indices_to_check, expected_directions
            ):
                return None

            characteristics = self._analyze_characteristics(data, indices_to_check)
            candles = [data.iloc[idx] for idx in indices_to_check]
            avg_body_size = characteristics["avg_body_size"]
            if avg_body_size <= 0:
                return None

            body_sizes = characteristics["body_sizes"]
            if not (
                body_sizes[4] > avg_body_size
                and body_sizes[0] > avg_body_size
                and body_sizes[1] < avg_body_size * 0.5
                and body_sizes[2] < avg_body_size * 0.5
                and body_sizes[3] < avg_body_size * 0.5
            ):
                return None

            first_high = float(candles[4]["high"])
            first_low = float(candles[4]["low"])
            for i in range(1, 4):
                if float(candles[i]["high"]) > first_high or float(candles[i]["low"]) < first_low:
                    return None

            pattern_factors = self._build_pattern_factors(
                data,
                validated_index,
                trend_lookback=5,
                candle_size_expected=0.8,
                price_movement_expected=1.0,
                pattern_completeness=min(1.0, (body_sizes[0] / avg_body_size) * 0.5),
                volume_expected=1.2,
            )

            confidence = self._calculate_confidence(
                data,
                validated_index,
                pattern_factors,
                multi_timeframe_data,
            )

            return self._create_signal_result(
                data,
                validated_index,
                confidence,
                direction_sign=1.0,
                description="Rising Three Methods: Bullish continuation pattern",
                metadata=cast(
                    SignalMetadata,
                    {
                        "pattern": "rising_three_methods",
                        "confidence": confidence,
                    },
                ),
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Rising Three Methods pattern: {e}")
            return None


class BullishEngulfingRecognizer(_EngulfingPatternBase):
    """Recognizes Bullish Engulfing pattern.
    陽線はらみ足
    """

    pattern_type = "bullish_engulfing"
    trend_requires_downtrend = True
    expected_directions = ("bearish", "bullish")
    is_bullish_pattern = True
    signal_direction = 1.0
    signal_description = "Bullish Engulfing: Strong reversal signal in downtrend"


class BearishEngulfingRecognizer(_EngulfingPatternBase):
    """Recognizes Bearish Engulfing pattern.
    陰線はらみ足
    """

    pattern_type = "bearish_engulfing"
    trend_requires_downtrend = False
    expected_directions = ("bullish", "bearish")
    is_bullish_pattern = False
    signal_direction = -1.0
    signal_description = "Bearish Engulfing: Strong reversal signal in uptrend"


class PiercingPatternRecognizer(_CandlestickPatternBase):
    """Recognizes Piercing Pattern.
    差し込み線
    A two-candle bullish reversal pattern in a downtrend.
    """

    pattern_type = "piercing_pattern"

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize Piercing Pattern at the given index."""
        try:
            validated_index = self._validate_index_with_trend(
                data,
                index,
                required_length=2,
                lookback=3,
                expect_downtrend=True,
                multi_timeframe_data=multi_timeframe_data,
            )
            if validated_index is None:
                return None

            indices_to_check = [validated_index - 1, validated_index]
            expected_directions = ["bearish", "bullish"]
            if not self.validate_pattern_structure(
                data, indices_to_check, expected_directions
            ):
                return None

            characteristics = self._analyze_characteristics(data, indices_to_check)
            previous = data.iloc[validated_index - 1]
            current = data.iloc[validated_index]

            prev_open = float(previous["open"])
            prev_close = float(previous["close"])
            curr_open = float(current["open"])
            curr_close = float(current["close"])
            prev_midpoint = (prev_open + prev_close) / 2.0

            if not (curr_open < prev_close and curr_close > prev_midpoint):
                return None

            prev_body_size = characteristics["body_sizes"][0]
            if prev_body_size <= 0:
                return None

            penetration = (curr_close - prev_midpoint) / prev_body_size
            pattern_factors = self._build_pattern_factors(
                data,
                validated_index,
                trend_lookback=5,
                candle_size_expected=0.7,
                price_movement_expected=0.8,
                pattern_completeness=min(1.0, penetration),
                volume_expected=1.2,
            )

            confidence = self._calculate_confidence(
                data,
                validated_index,
                pattern_factors,
                multi_timeframe_data,
            )

            return self._create_signal_result(
                data,
                validated_index,
                confidence,
                direction_sign=1.0,
                description="Piercing Pattern: Bullish reversal signal in downtrend",
                metadata=cast(
                    SignalMetadata,
                    {
                        "pattern": "piercing_pattern",
                        "confidence": confidence,
                        "penetration": penetration,
                    },
                ),
            )

        except Exception as e:
            self.logger.error(f"Error recognizing Piercing Pattern: {e}")
            return None
