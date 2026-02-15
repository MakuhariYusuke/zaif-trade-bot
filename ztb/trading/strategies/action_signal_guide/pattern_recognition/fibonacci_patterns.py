"""
Fibonacci Pattern Recognition Module

This module provides pattern recognition for Fibonacci-based technical analysis,
including retracements, extensions, projections, and Fibonacci-based patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, cast

import pandas as pd

from ztb.types.common import ConfigSection, ObjectMap

from .base import CandlestickPatternRecognizer, MultiTimeframeData, SignalMetadata, SignalResult


class FibonacciRetracementMatch(TypedDict):
    """Detected retracement information for a swing."""

    level: float
    actual_ratio: float
    swing_high: float
    swing_low: float
    current_price: float
    start_idx: int
    end_idx: int


@dataclass(frozen=True)
class FibonacciLevelConfig:
    """Per-level configuration for retracement interpretation."""

    strength: float
    direction_factor: float


class FibonacciAnalyzer:
    """Utility class for Fibonacci calculations and analysis."""

    # Standard Fibonacci ratios
    RETRACEMENT_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
    EXTENSION_LEVELS = [0.618, 1.0, 1.236, 1.382, 1.618, 2.0, 2.618]
    PROJECTION_LEVELS = [0.618, 1.0, 1.236, 1.382, 1.618, 2.0, 2.618]

    # Class-level cache for retracement calculations
    _retracement_cache: dict[str, FibonacciRetracementMatch | None] = {}
    _max_cache_size = 2048

    @staticmethod
    def calculate_retracement_levels(high: float, low: float) -> dict[float, float]:
        """Calculate Fibonacci retracement levels between high and low."""
        diff = high - low
        return {ratio: low + diff * ratio for ratio in FibonacciAnalyzer.RETRACEMENT_LEVELS}

    @staticmethod
    def calculate_extension_levels(
        high: float, low: float, direction: int = 1
    ) -> dict[float, float]:
        """Calculate Fibonacci extension levels."""
        diff = high - low
        levels: dict[float, float] = {}
        for ratio in FibonacciAnalyzer.EXTENSION_LEVELS:
            if direction == 1:  # Bullish extension from low
                levels[ratio] = low + diff * ratio
            else:  # Bearish extension from high
                levels[ratio] = high - diff * ratio
        return levels

    @staticmethod
    def calculate_deviation_from_ideal(actual_ratio: float, target_level: float) -> float:
        """Calculate deviation from ideal Fibonacci level."""
        deviation = abs(actual_ratio - target_level)
        tolerance = 0.02  # 2% tolerance for major levels

        if deviation <= tolerance:
            return 0.0
        if deviation <= tolerance * 2:
            return 0.3
        if deviation <= tolerance * 4:
            return 0.6
        return 1.0

    @staticmethod
    def validate_with_multi_timeframe(
        actual_ratio: float,
        target_level: float,
        multi_timeframe_data: MultiTimeframeData | ObjectMap | None = None,
    ) -> float:
        """Validate Fibonacci level using multi-timeframe confirmation."""
        if not multi_timeframe_data:
            return 0.5

        try:
            higher_tf_trend = 0.0

            if isinstance(multi_timeframe_data, dict):
                direct_value = multi_timeframe_data.get("higher_timeframe_trend")
                if isinstance(direct_value, (int, float)):
                    higher_tf_trend = float(direct_value)
                else:
                    # Fallback: infer from first timeframe payload containing dataframe.
                    for payload in multi_timeframe_data.values():
                        if isinstance(payload, dict) and "data" in payload:
                            tf_df = payload["data"]
                            if isinstance(tf_df, pd.DataFrame) and len(tf_df) > 1:
                                prev_close = float(tf_df.iloc[-2]["close"])
                                curr_close = float(tf_df.iloc[-1]["close"])
                                higher_tf_trend = curr_close - prev_close
                                break

            if abs(higher_tf_trend) > 0.5:
                deviation = abs(actual_ratio - target_level)
                if deviation < 0.03:
                    return 0.8
                if deviation < 0.06:
                    return 0.6

            return 0.4
        except Exception:
            return 0.5

    @staticmethod
    def calculate_fibonacci_strength(
        actual_ratio: float, target_level: float, level_significance: float
    ) -> float:
        """Calculate overall Fibonacci pattern strength."""
        deviation_score = FibonacciAnalyzer.calculate_deviation_from_ideal(
            actual_ratio, target_level
        )
        strength = level_significance * (1.0 - deviation_score * 0.5)
        return max(0.0, min(1.0, strength))

    @staticmethod
    def find_fibonacci_retracement(
        data: pd.DataFrame, start_idx: int, end_idx: int
    ) -> FibonacciRetracementMatch | None:
        """Find retracement alignment for a swing range."""
        if start_idx >= end_idx or end_idx >= len(data) or start_idx < 0:
            return None

        # Include context in key to avoid cross-dataset collisions.
        key = (
            f"{len(data)}_{start_idx}_{end_idx}_"
            f"{float(data.iloc[start_idx]['high']):.8f}_"
            f"{float(data.iloc[start_idx]['low']):.8f}_"
            f"{float(data.iloc[end_idx]['close']):.8f}"
        )
        if key in FibonacciAnalyzer._retracement_cache:
            return FibonacciAnalyzer._retracement_cache[key]

        swing_high = float(data.iloc[start_idx : end_idx + 1]["high"].max())
        swing_low = float(data.iloc[start_idx : end_idx + 1]["low"].min())
        current_close = float(data.iloc[end_idx]["close"])

        total_range = swing_high - swing_low
        result: FibonacciRetracementMatch | None
        if total_range == 0:
            result = None
        else:
            retracement_ratio = (swing_high - current_close) / total_range
            closest_level = min(
                FibonacciAnalyzer.RETRACEMENT_LEVELS,
                key=lambda x: abs(x - retracement_ratio),
            )

            tolerance = 0.02  # 2% tolerance
            if abs(retracement_ratio - closest_level) <= tolerance:
                result = {
                    "level": float(closest_level),
                    "actual_ratio": float(retracement_ratio),
                    "swing_high": swing_high,
                    "swing_low": swing_low,
                    "current_price": current_close,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            else:
                result = None

        # Bounded cache to avoid unbounded growth in long sessions.
        FibonacciAnalyzer._retracement_cache[key] = result
        while len(FibonacciAnalyzer._retracement_cache) > FibonacciAnalyzer._max_cache_size:
            oldest_key = next(iter(FibonacciAnalyzer._retracement_cache))
            FibonacciAnalyzer._retracement_cache.pop(oldest_key, None)

        return result

    @staticmethod
    def find_support_resistance_levels(prices: pd.Series | list[float]) -> dict[str, list[float]]:
        """Compatibility helper for tests and legacy callers."""
        if isinstance(prices, pd.Series):
            series = prices.astype(float)
        else:
            series = pd.Series(prices, dtype=float)

        if series.empty:
            return {"support": [], "resistance": []}

        high = float(series.max())
        low = float(series.min())
        retracements = FibonacciAnalyzer.calculate_retracement_levels(high, low)

        pivot = float(series.iloc[-1])
        support = sorted([level for level in retracements.values() if level <= pivot])
        resistance = sorted([level for level in retracements.values() if level > pivot])
        return {"support": support, "resistance": resistance}


class _FibonacciPatternBase(CandlestickPatternRecognizer):
    """Shared behavior for Fibonacci recognizers."""

    pattern_type = "fibonacci"

    def __init__(
        self,
        config: ConfigSection | None,
        *,
        pattern_type: str,
        default_min_swing_length: int,
        lookback_key: str,
        default_lookback: int,
    ) -> None:
        super().__init__(config)
        self.pattern_type = pattern_type
        self.fib_analyzer = FibonacciAnalyzer()

        self.min_swing_length = int(
            self.config.get("min_swing_length", default_min_swing_length)
        )
        self.lookback_window = int(self.config.get(lookback_key, default_lookback))
        self.confidence_cap = float(self.config.get("confidence_cap", 0.0001))
        self.pattern_completeness_threshold = float(
            self.config.get("pattern_completeness_threshold", 0.0)
        )

        # Compatibility for legacy tests/consumers.
        self.thresholds: dict[str, float] = {
            "retracement_threshold": float(self.config.get("retracement_threshold", 0.02)),
            "pattern_completeness_threshold": self.pattern_completeness_threshold,
            "confidence_cap": self.confidence_cap,
        }

    def _resolve_index(self, data: pd.DataFrame, index: int) -> int | None:
        try:
            resolved = self.validate_recognition_inputs(
                data,
                index,
                required_length=max(self.lookback_window + 1, self.min_swing_length + 2),
            )
        except Exception:
            return None

        if resolved < self.lookback_window:
            return None
        return resolved

    def _calculate_capped_confidence(
        self,
        data: pd.DataFrame,
        index: int,
        *,
        base_confidence: float,
        trend_lookback: int,
        candle_size_expected: float,
        price_movement_expected: float,
        pattern_completeness: float,
    ) -> float:
        pattern_factors = {
            "trend_strength": self._calculate_trend_strength(data, index, trend_lookback),
            "candle_size": self._calculate_candle_size_confidence(
                data, index, candle_size_expected
            ),
            "price_movement": self._calculate_price_movement_confidence(
                data, index, price_movement_expected
            ),
            "pattern_completeness": max(0.0, min(1.0, pattern_completeness)),
        }
        confidence = self._calculate_pattern_confidence(
            data, index, pattern_factors, base_confidence=base_confidence
        )
        return min(confidence, self.confidence_cap)

    def _passes_pattern_threshold(self, pattern_completeness: float) -> bool:
        return pattern_completeness >= self.pattern_completeness_threshold


class FibonacciRetracementRecognizer(_FibonacciPatternBase):
    """Recognizes Fibonacci retracement levels in price action."""

    _LEVEL_CONFIG: dict[float, FibonacciLevelConfig] = {
        0.236: FibonacciLevelConfig(strength=0.4, direction_factor=0.6),
        0.382: FibonacciLevelConfig(strength=0.6, direction_factor=0.7),
        0.5: FibonacciLevelConfig(strength=0.8, direction_factor=0.8),
        0.618: FibonacciLevelConfig(strength=0.9, direction_factor=0.9),
        0.786: FibonacciLevelConfig(strength=0.7, direction_factor=0.8),
    }

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="fibonacci_retracement",
            default_min_swing_length=5,
            lookback_key="max_swing_length",
            default_lookback=50,
        )
        self.max_swing_length = self.lookback_window

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize Fibonacci retracement at the given index."""
        resolved_index = self._resolve_index(data, index)
        if resolved_index is None:
            return None

        for swing_length in range(
            self.min_swing_length, min(self.max_swing_length, resolved_index + 1)
        ):
            start_idx = resolved_index - swing_length
            fib_retracement = self.fib_analyzer.find_fibonacci_retracement(
                data, start_idx, resolved_index
            )
            if fib_retracement is None:
                continue

            level = fib_retracement["level"]
            level_cfg = self._LEVEL_CONFIG.get(
                level, FibonacciLevelConfig(strength=0.5, direction_factor=0.6)
            )

            start_close = float(data.iloc[start_idx]["close"])
            current_close = float(data.iloc[resolved_index]["close"])
            price_movement = current_close - start_close

            prior_idx = start_idx - swing_length
            if prior_idx >= 0:
                prior_close = float(data.iloc[prior_idx]["close"])
                swing_size = abs(start_close - prior_close)
            else:
                swing_size = abs(price_movement)

            if swing_size > 0:
                retracement_ratio = abs(price_movement) / swing_size
                direction_scale = max(0.0, 1.0 - retracement_ratio * 0.5)
                direction = (
                    level_cfg.direction_factor * direction_scale
                    if price_movement > 0
                    else -level_cfg.direction_factor * direction_scale
                )
            else:
                direction = 0.0

            actual_ratio = fib_retracement["actual_ratio"]
            strength = self.fib_analyzer.calculate_fibonacci_strength(
                actual_ratio, level, level_cfg.strength
            )
            mtf_boost = self.fib_analyzer.validate_with_multi_timeframe(
                actual_ratio, level, multi_timeframe_data
            )
            base_strength = min(1.0, strength * (1.0 + mtf_boost * 0.2))

            deviation_score = self.fib_analyzer.calculate_deviation_from_ideal(
                actual_ratio, level
            )
            pattern_completeness = 1.0 - deviation_score
            if not self._passes_pattern_threshold(pattern_completeness):
                continue

            confidence = self._calculate_capped_confidence(
                data,
                resolved_index,
                base_confidence=base_strength,
                trend_lookback=15,
                candle_size_expected=0.6,
                price_movement_expected=0.7,
                pattern_completeness=pattern_completeness,
            )

            signal_type = (
                "fib_retracement_support" if direction > 0 else "fib_retracement_resistance"
            )

            return SignalResult(
                signal_type=signal_type,
                strength=confidence,
                direction=max(-1.0, min(1.0, direction)),
                description=(
                    f"Fibonacci Retracement at {level:.3f} level "
                    f"(deviation: {deviation_score:.2f})"
                ),
                timestamp=data.index[resolved_index],
                confidence=confidence,
                metadata=cast(
                    SignalMetadata,
                    {
                        "pattern": "fibonacci_retracement",
                        "level": level,
                        "actual_ratio": actual_ratio,
                        "deviation_score": deviation_score,
                        "multi_timeframe_boost": mtf_boost,
                        "swing_length": swing_length,
                        "confidence": confidence,
                        "pattern_completeness": pattern_completeness,
                    },
                ),
            )

        return None


class FibonacciExtensionRecognizer(_FibonacciPatternBase):
    """Recognizes Fibonacci extension targets."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="fibonacci_extension",
            default_min_swing_length=5,
            lookback_key="max_swing_length",
            default_lookback=50,
        )
        self.max_swing_length = self.lookback_window

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize Fibonacci extension targets at the given index."""
        _ = multi_timeframe_data  # Extension currently does not use MTF enhancement.
        resolved_index = self._resolve_index(data, index)
        if resolved_index is None:
            return None

        current_price = float(data.iloc[resolved_index]["close"])

        for swing_length in range(
            self.min_swing_length, min(self.max_swing_length, resolved_index + 1)
        ):
            start_idx = resolved_index - swing_length

            swing_high = float(data.iloc[start_idx : resolved_index + 1]["high"].max())
            swing_low = float(data.iloc[start_idx : resolved_index + 1]["low"].min())
            swing_range = abs(swing_high - swing_low)
            if swing_range <= 0:
                continue

            trend_direction = 1 if current_price > (swing_high + swing_low) / 2 else -1
            extension_levels = self.fib_analyzer.calculate_extension_levels(
                swing_high, swing_low, trend_direction
            )

            for ratio, level in extension_levels.items():
                tolerance = swing_range * 0.02
                if tolerance <= 0:
                    continue

                if abs(current_price - level) > tolerance:
                    continue

                price_deviation = abs(current_price - level) / tolerance
                pattern_completeness = max(0.0, 1.0 - price_deviation)
                if not self._passes_pattern_threshold(pattern_completeness):
                    continue

                base_confidence = min(0.9, 0.7 + (ratio - 1.0) * 0.1)
                confidence = self._calculate_capped_confidence(
                    data,
                    resolved_index,
                    base_confidence=base_confidence,
                    trend_lookback=20,
                    candle_size_expected=0.6,
                    price_movement_expected=0.8,
                    pattern_completeness=pattern_completeness,
                )

                signal_type = (
                    "fib_extension_target"
                    if trend_direction == 1
                    else "fib_extension_target_bearish"
                )

                return SignalResult(
                    signal_type=signal_type,
                    strength=confidence,
                    direction=float(trend_direction),
                    description=f"Fibonacci Extension target at {ratio:.3f} level",
                    timestamp=data.index[resolved_index],
                    confidence=confidence,
                    metadata=cast(
                        SignalMetadata,
                        {
                            "pattern": "fibonacci_extension",
                            "level": ratio,
                            "target_price": level,
                            "swing_length": swing_length,
                            "confidence": confidence,
                            "pattern_completeness": pattern_completeness,
                        },
                    ),
                )

        return None


class FibonacciProjectionRecognizer(_FibonacciPatternBase):
    """Recognizes Fibonacci price projections from multiple swings."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="fibonacci_projection",
            default_min_swing_length=3,
            lookback_key="max_lookback",
            default_lookback=20,
        )
        self.max_lookback = self.lookback_window

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize Fibonacci projections at the given index."""
        _ = multi_timeframe_data
        resolved_index = self._resolve_index(data, index)
        if resolved_index is None:
            return None

        current_price = float(data.iloc[resolved_index]["close"])

        for first_swing_end in range(
            resolved_index - self.min_swing_length,
            resolved_index - 2 * self.min_swing_length,
            -1,
        ):
            if first_swing_end < 0:
                break

            first_swing_start = first_swing_end - self.min_swing_length
            if first_swing_start < 0:
                continue

            first_high = float(
                data.iloc[first_swing_start : first_swing_end + 1]["high"].max()
            )
            first_low = float(
                data.iloc[first_swing_start : first_swing_end + 1]["low"].min()
            )
            first_range = abs(first_high - first_low)
            if first_range <= 0:
                continue

            second_high = float(data.iloc[first_swing_end : resolved_index + 1]["high"].max())
            second_low = float(data.iloc[first_swing_end : resolved_index + 1]["low"].min())

            if second_high > first_high:  # Bullish projection
                projection_base = first_low
                projection_range = second_high - first_low
            elif second_low < first_low:  # Bearish projection
                projection_base = first_high
                projection_range = first_high - second_low
            else:
                continue

            for ratio in FibonacciAnalyzer.PROJECTION_LEVELS:
                projected_price = projection_base + projection_range * ratio
                tolerance = first_range * 0.03
                if tolerance <= 0:
                    continue

                if abs(current_price - projected_price) > tolerance:
                    continue

                direction = 1.0 if projected_price > projection_base else -1.0
                price_deviation = abs(current_price - projected_price) / tolerance
                pattern_completeness = max(0.0, 1.0 - price_deviation)
                if not self._passes_pattern_threshold(pattern_completeness):
                    continue

                base_confidence = min(0.85, 0.6 + (ratio - 1.0) * 0.15)
                confidence = self._calculate_capped_confidence(
                    data,
                    resolved_index,
                    base_confidence=base_confidence,
                    trend_lookback=25,
                    candle_size_expected=0.6,
                    price_movement_expected=0.8,
                    pattern_completeness=pattern_completeness,
                )

                return SignalResult(
                    signal_type="fib_projection_target",
                    strength=confidence,
                    direction=direction,
                    description=f"Fibonacci Projection at {ratio:.3f} level",
                    timestamp=data.index[resolved_index],
                    confidence=confidence,
                    metadata=cast(
                        SignalMetadata,
                        {
                            "pattern": "fibonacci_projection",
                            "level": ratio,
                            "projected_price": projected_price,
                            "confidence": confidence,
                            "pattern_completeness": pattern_completeness,
                        },
                    ),
                )

        return None
