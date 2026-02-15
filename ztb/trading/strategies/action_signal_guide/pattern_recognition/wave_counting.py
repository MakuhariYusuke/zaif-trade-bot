"""
Wave Counting Module

This module provides pattern recognition for wave counting analysis,
primarily based on Elliott Wave Theory including impulse waves, corrective waves,
and various wave patterns.
"""

from enum import Enum
from typing import List, NamedTuple, TypedDict

import pandas as pd

from ztb.trading.environment.constants import EPSILON
from ztb.types.common import ConfigSection

from .base import CandlestickPatternRecognizer, MultiTimeframeData, SignalResult


class WaveType(Enum):
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"
    DIAGONAL = "diagonal"
    TRIANGLE = "triangle"


class WaveLabel(Enum):
    """Specific Elliott Wave labels."""

    I = "I"  # Initial impulse wave
    II = "II"  # First corrective wave
    III = "III"  # Third impulse wave (usually strongest)
    IV = "IV"  # Second corrective wave
    V = "V"  # Final impulse wave
    A = "A"  # Start of correction
    B = "B"  # Partial retracement in correction
    C = "C"  # Final leg of correction
    W = "W"  # First leg of double correction
    X = "X"  # Connecting wave in complex correction
    Y = "Y"  # Terminal wave in complex corrections
    P = "P"  # Irregular correction wave
    N = "N"  # Complex correction wave
    S = "S"  # Secondary correction wave
    PH = "PH"  # Pivot High
    PL = "PL"  # Pivot Low


class WaveDegree(Enum):
    GRAND_SUPERCYCLE = "grand_supercycle"
    SUPERCYCLE = "supercycle"
    CYCLE = "cycle"
    PRIMARY = "primary"
    INTERMEDIATE = "intermediate"
    MINOR = "minor"
    MINUTE = "minute"
    MINUETTE = "minuette"
    SUBMINUETTE = "subminuette"


class WavePoint(NamedTuple):
    position: int
    price: float
    wave_label: WaveLabel
    degree: WaveDegree


class WaveStructure(TypedDict):
    """Typed structure returned by wave structure identification."""

    type: WaveType
    degree: WaveDegree
    waves: list[WavePoint]
    direction: int
    strength: float
    completion_index: int


class WaveAnalyzer:
    """
    Utility class for wave counting and analysis.

    Provides methods to:
    - Find significant pivot points in price data for wave identification.
    - Identify Elliott Wave structures such as impulse and corrective patterns.
    """

    @staticmethod
    def find_pivot_points(
        data: pd.DataFrame, lookback: int = 20, min_distance: int = 3
    ) -> List[WavePoint]:
        """Find significant pivot points in the data."""
        if len(data) < lookback:
            return []

        highs = data["high"]
        lows = data["low"]

        pivot_highs: List[WavePoint] = []
        pivot_lows: List[WavePoint] = []

        for i in range(lookback // 2, len(data) - lookback // 2):
            # Check for pivot high
            is_pivot_high = True
            for j in range(1, lookback // 2 + 1):
                if (
                    highs.iloc[i] <= highs.iloc[i - j]
                    or highs.iloc[i] <= highs.iloc[i + j]
                ):
                    is_pivot_high = False
                    break

            if is_pivot_high:
                # Check minimum distance from previous pivot
                if not pivot_highs or (i - pivot_highs[-1].position) >= min_distance:
                    pivot_highs.append(
                        WavePoint(i, highs.iloc[i], WaveLabel.PH, WaveDegree.MINOR)
                    )

            # Check for pivot low
            is_pivot_low = True
            for j in range(1, lookback // 2 + 1):
                if lows.iloc[i] >= lows.iloc[i - j] or lows.iloc[i] >= lows.iloc[i + j]:
                    is_pivot_low = False
                    break

            if is_pivot_low:
                # Check minimum distance from previous pivot
                if not pivot_lows or (i - pivot_lows[-1].position) >= min_distance:
                    pivot_lows.append(
                        WavePoint(i, lows.iloc[i], WaveLabel.PL, WaveDegree.MINOR)
                    )

        # Combine and sort by index
        all_pivots = pivot_highs + pivot_lows
        all_pivots.sort(key=lambda x: x.position)

        return all_pivots

    @staticmethod
    def identify_wave_structure(pivots: List[WavePoint]) -> WaveStructure | None:
        """Identify wave structure from pivot points."""
        if len(pivots) < 5:
            return None

        # Look for 5-wave impulse pattern
        # Wave 1: up, 2: down (correction), 3: up, 4: down (correction), 5: up

        # Find potential wave 1-5 sequence
        for i in range(len(pivots) - 4):
            w1, w2, w3, w4, w5 = pivots[i : i + 5]

            # Basic impulse wave rules
            if (
                w1.price < w3.price > w5.price
                and w2.price < w1.price  # Waves 1,3,5 trending up
                and w4.price < w3.price
                and w3.price > w1.price  # Corrections
                and w5.price > w3.price
            ):  # Progression
                # Check wave ratios (Fibonacci relationships)
                wave1_length = w1.price - min(w1.price, w2.price)
                wave3_length = w3.price - w2.price
                wave5_length = w5.price - w4.price

                # Wave 3 should be the longest
                if wave3_length > wave1_length and wave3_length > wave5_length:
                    # Check Fibonacci extensions
                    total_length = w5.price - w1.price
                    wave3_ratio = wave3_length / total_length

                    if 0.5 < wave3_ratio < 0.8:  # Wave 3 typically 50-80% of total
                        return {
                            "type": WaveType.IMPULSE,
                            "degree": WaveDegree.MINOR,
                            "waves": [w1, w2, w3, w4, w5],
                            "direction": 1,  # Bullish impulse
                            "strength": 0.8,
                            "completion_index": w5.position,
                        }

        # Look for corrective ABC pattern
        for i in range(len(pivots) - 2):
            a, b, c = pivots[i : i + 3]

            # ABC correction: A down, B up (partial retracement), C down (beyond A)
            if (
                a.price > b.price
                and b.price < c.price
                and c.price < a.price  # B is lower than both A and C
            ):  # C goes below A
                # Check Fibonacci retracement of B from A
                ab_range = a.price - b.price
                if ab_range == 0:
                    continue  # Skip this pattern, avoid division by zero
                bc_retracement = (b.price - c.price) / ab_range

                if 0.5 < bc_retracement < 0.8:  # 50-80% retracement
                    return {
                        "type": WaveType.CORRECTIVE,
                        "degree": WaveDegree.MINOR,
                        "waves": [a, b, c],
                        "direction": -1,  # Bearish correction
                        "strength": 0.7,
                        "completion_index": c.position,
                    }

        return None


class _WavePatternBase(CandlestickPatternRecognizer):
    """Shared utilities for wave recognizers."""

    pattern_type: str = "wave_pattern"

    def __init__(
        self,
        config: ConfigSection | None,
        *,
        pattern_type: str,
        default_lookback: int,
        default_min_pivot_distance: int = 3,
    ) -> None:
        super().__init__(config)
        self.pattern_type = pattern_type

        lookback_period = self.config.get("lookback_period", default_lookback)
        min_pivot_distance = self.config.get(
            "min_pivot_distance", default_min_pivot_distance
        )
        self.lookback_period = int(lookback_period)
        self.min_pivot_distance = int(min_pivot_distance)

        # Keep base accessor (`get_lookback_period`) aligned with runtime value.
        self.config.setdefault("lookback_period", self.lookback_period)
        self.wave_analyzer = WaveAnalyzer()

    def _resolve_index(
        self,
        data: pd.DataFrame,
        index: int,
        multi_timeframe_data: MultiTimeframeData | None,
    ) -> int | None:
        try:
            validated_index = self.validate_recognition_inputs(
                data,
                index,
                required_length=self.lookback_period + 1,
                multi_timeframe_data=multi_timeframe_data,
            )
        except Exception:
            return None

        if validated_index < self.lookback_period:
            return None
        return validated_index

    def _extract_global_pivots(
        self, data: pd.DataFrame, index: int, lookback_divisor: int = 2
    ) -> list[WavePoint]:
        lookback_start = max(0, index - self.lookback_period)
        lookback_data = data.iloc[lookback_start : index + 1]
        pivot_lookback = max(2, self.lookback_period // max(1, lookback_divisor))

        pivots = self.wave_analyzer.find_pivot_points(
            lookback_data,
            lookback=pivot_lookback,
            min_distance=self.min_pivot_distance,
        )
        return [
            WavePoint(
                position=p.position + lookback_start,
                price=float(p.price),
                wave_label=p.wave_label,
                degree=p.degree,
            )
            for p in pivots
        ]

    def _calculate_wave_confidence(
        self,
        data: pd.DataFrame,
        index: int,
        *,
        base_confidence: float,
        pattern_completeness: float,
        trend_lookback: int,
        candle_size_expected: float,
        price_movement_expected: float,
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
        return self._calculate_pattern_confidence(
            data, index, pattern_factors, base_confidence=base_confidence
        )


class ImpulseWaveRecognizer(_WavePatternBase):
    """Recognizes Elliott Wave impulse patterns (5-wave structures)."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="impulse_wave",
            default_lookback=50,
            default_min_pivot_distance=3,
        )

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize impulse wave patterns at the given index."""
        validated_index = self._resolve_index(data, index, multi_timeframe_data)
        if validated_index is None:
            return None
        index = validated_index

        pivots = self._extract_global_pivots(data, index, lookback_divisor=2)
        wave_structure = self.wave_analyzer.identify_wave_structure(pivots)

        if wave_structure and wave_structure["type"] == WaveType.IMPULSE:
            # Check if we're at or near the completion of wave 5
            if abs(index - wave_structure["completion_index"]) <= 2:  # Within 2 bars
                # Calculate dynamic confidence based on wave structure quality
                waves = wave_structure["waves"]
                if len(waves) >= 5:
                    w1, w2, w3, w4, w5 = waves[-5:]

                    # Calculate wave ratios for pattern completeness
                    wave1_length = abs(w1.price - w2.price)
                    wave3_length = abs(w3.price - w2.price)
                    wave5_length = abs(w5.price - w4.price)
                    total_length = abs(w5.price - w1.price)

                    # Wave 3 should be the strongest (longest)
                    wave3_ratio = (
                        wave3_length / total_length if total_length > 0 else 0.5
                    )
                    pattern_completeness = min(
                        1.0, wave3_ratio * 1.5
                    )  # Boost confidence for strong wave 3

                    # Base confidence from wave structure quality
                    base_confidence = min(0.9, 0.7 + pattern_completeness * 0.2)

                    confidence = self._calculate_wave_confidence(
                        data,
                        index,
                        base_confidence=base_confidence,
                        pattern_completeness=pattern_completeness,
                        trend_lookback=20,
                        candle_size_expected=0.6,
                        price_movement_expected=0.7,
                    )
                    direction = float(wave_structure["direction"]) * confidence

                    signal_type = "impulse_wave_completion"

                    return SignalResult(
                        signal_type=signal_type,
                        strength=confidence,
                        direction=direction,
                        description="Elliott Wave Impulse Pattern (5-wave structure)",
                        timestamp=data.index[index],
                        confidence=confidence,
                        metadata={
                            "pattern": "impulse_wave",
                            "wave_type": wave_structure["type"].value,
                            "degree": wave_structure["degree"].value,
                            "wave_count": len(wave_structure["waves"]),
                            "confidence": confidence,
                            "pattern_completeness": pattern_completeness,
                        },
                    )

        return None


class CorrectiveWaveRecognizer(_WavePatternBase):
    """Recognizes Elliott Wave corrective patterns (ABC structures)."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="corrective_wave",
            default_lookback=40,
            default_min_pivot_distance=3,
        )

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize corrective wave patterns at the given index."""
        validated_index = self._resolve_index(data, index, multi_timeframe_data)
        if validated_index is None:
            return None
        index = validated_index

        pivots = self._extract_global_pivots(data, index, lookback_divisor=2)
        wave_structure = self.wave_analyzer.identify_wave_structure(pivots)

        if wave_structure and wave_structure["type"] == WaveType.CORRECTIVE:
            # Check if we're at or near the completion of wave C
            if abs(index - wave_structure["completion_index"]) <= 2:  # Within 2 bars
                # Calculate dynamic confidence based on corrective structure quality
                waves = wave_structure["waves"]
                if len(waves) >= 3:
                    a, b, c = waves[-3:]

                    # Calculate corrective pattern ratios
                    ab_range = abs(a.price - b.price)
                    bc_range = abs(b.price - c.price)
                    total_correction = abs(a.price - c.price)

                    # B should be partial retracement (typically 0.382-0.786 of A)
                    b_retracement = (
                        ab_range / (a.price - min(a.price, c.price))
                        if (a.price - min(a.price, c.price)) > 0
                        else 0.5
                    )
                    pattern_completeness = 1.0 - abs(
                        b_retracement - 0.618
                    )  # Closer to 0.618 Fibonacci is better

                    # C should extend beyond A (for zigzag corrections)
                    c_extension = bc_range / ab_range if ab_range > 0 else 1.0
                    if c_extension > 1.0:  # C extends beyond A
                        pattern_completeness *= min(1.0, c_extension * 0.5)

                    # Base confidence from pattern structure quality
                    base_confidence = min(0.85, 0.6 + pattern_completeness * 0.25)

                    confidence = self._calculate_wave_confidence(
                        data,
                        index,
                        base_confidence=base_confidence,
                        pattern_completeness=pattern_completeness,
                        trend_lookback=15,
                        candle_size_expected=0.5,
                        price_movement_expected=0.6,
                    )
                    direction = float(wave_structure["direction"]) * confidence

                    signal_type = "corrective_wave_completion"

                    return SignalResult(
                        signal_type=signal_type,
                        strength=confidence,
                        direction=direction,
                        description="Elliott Wave Corrective Pattern (ABC structure)",
                        timestamp=data.index[index],
                        confidence=confidence,
                        metadata={
                            "pattern": "corrective_wave",
                            "wave_type": wave_structure["type"].value,
                            "degree": wave_structure["degree"].value,
                            "wave_count": len(wave_structure["waves"]),
                            "confidence": confidence,
                            "pattern_completeness": pattern_completeness,
                        },
                    )

        return None


class WaveExtensionRecognizer(_WavePatternBase):
    """Recognizes wave extensions and truncations."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="wave_extension",
            default_lookback=60,
            default_min_pivot_distance=3,
        )

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize wave extensions or truncations at the given index."""
        validated_index = self._resolve_index(data, index, multi_timeframe_data)
        if validated_index is None:
            return None
        index = validated_index

        pivots = self._extract_global_pivots(data, index, lookback_divisor=3)

        if len(pivots) < 3:
            return None

        # Look for extended waves (wave 3 much longer than others)
        recent_pivots = pivots[-5:] if len(pivots) >= 5 else pivots

        if len(recent_pivots) >= 3:
            # Calculate wave lengths
            wave_lengths = []
            for i in range(len(recent_pivots) - 1):
                length = abs(recent_pivots[i + 1].price - recent_pivots[i].price)
                wave_lengths.append(length)

            if len(wave_lengths) >= 3:
                # Check for wave 3 extension (much longer than waves 1 and 5)
                if (
                    wave_lengths[1] > wave_lengths[0] * 1.5
                    and wave_lengths[1] > wave_lengths[2] * 1.5
                ):
                    strength = min(
                        0.85,
                        0.6
                        + (
                            wave_lengths[1]
                            / max(wave_lengths[0], wave_lengths[2], EPSILON)
                            - 1.5
                        )
                        * 0.1,
                    )
                    direction = (
                        1 if recent_pivots[-1].price > recent_pivots[0].price else -1
                    ) * strength

                    dominant_base = max(wave_lengths[0], wave_lengths[2], EPSILON)
                    return SignalResult(
                        signal_type="wave_extension",
                        strength=strength,
                        direction=direction,
                        description="Elliott Wave Extension (extended wave 3)",
                        timestamp=data.index[index],
                        metadata={
                            "pattern": "wave_extension",
                            "extension_ratio": wave_lengths[1] / dominant_base,
                            "confidence": strength,
                        },
                    )

        # Look for truncated waves (wave 5 fails to exceed wave 3)
        if len(recent_pivots) >= 5:
            w1, w2, w3, w4, w5 = recent_pivots[-5:]

            # Check for truncation: wave 5 doesn't exceed wave 3 high (in bullish case)
            if w3.price > w1.price and w5.price < w3.price:
                # Calculate truncation severity
                truncation_ratio = (
                    (w3.price - w5.price) / (w3.price - w1.price)
                    if (w3.price - w1.price) > 0
                    else 0.5
                )
                base_confidence = min(0.8, 0.6 + truncation_ratio * 0.3)

                confidence = self._calculate_wave_confidence(
                    data,
                    index,
                    base_confidence=base_confidence,
                    pattern_completeness=truncation_ratio,
                    trend_lookback=10,
                    candle_size_expected=0.5,
                    price_movement_expected=0.8,
                )

                return SignalResult(
                    signal_type="wave_truncation",
                    strength=confidence,
                    direction=-confidence,  # Bearish signal (failure)
                    description="Elliott Wave Truncation (wave 5 fails to exceed wave 3)",
                    timestamp=data.index[index],
                    confidence=confidence,
                    metadata={
                        "pattern": "wave_truncation",
                        "wave3_high": w3.price,
                        "wave5_high": w5.price,
                        "truncation_ratio": truncation_ratio,
                        "confidence": confidence,
                    },
                )

        return None


class WaveIRecognizer(_WavePatternBase):
    """Recognizes Wave I (Initial impulse wave) - start of a new trend."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="wave_i",
            default_lookback=30,
            default_min_pivot_distance=3,
        )

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize Wave I patterns at the given index."""
        validated_index = self._resolve_index(data, index, multi_timeframe_data)
        if validated_index is None:
            return None
        index = validated_index

        pivots = self._extract_global_pivots(data, index, lookback_divisor=2)

        if len(pivots) < 2:
            return None

        # Look for the start of an impulse wave (Wave I)
        recent_pivots = pivots[-2:]  # Last two pivots

        if len(recent_pivots) >= 2:
            p1, p2 = recent_pivots[-2], recent_pivots[-1]

            # Wave I should be a strong directional move
            if (
                abs(p2.price - p1.price)
                > data["close"].iloc[index - self.lookback_period : index].std() * 2
            ):
                direction: float = 1.0 if p2.price > p1.price else -1.0

                # Calculate wave strength based on size relative to recent volatility
                wave_size = abs(p2.price - p1.price)
                recent_volatility = (
                    data["close"].iloc[index - self.lookback_period : index].std()
                )
                wave_strength_ratio = (
                    wave_size / recent_volatility if recent_volatility > 0 else 1.0
                )

                # Pattern completeness based on how strong the initial move is
                pattern_completeness = min(
                    1.0, wave_strength_ratio / 3.0
                )  # Strong moves get higher confidence

                # Base confidence from wave strength
                base_confidence = min(0.85, 0.6 + pattern_completeness * 0.25)

                confidence = self._calculate_wave_confidence(
                    data,
                    index,
                    base_confidence=base_confidence,
                    pattern_completeness=pattern_completeness,
                    trend_lookback=15,
                    candle_size_expected=0.7,
                    price_movement_expected=0.8,
                )
                direction = direction * confidence

                return SignalResult(
                    signal_type="wave_i",
                    strength=confidence,
                    direction=direction,
                    description="Elliott Wave I (Initial impulse wave)",
                    timestamp=data.index[index],
                    confidence=confidence,
                    metadata={
                        "pattern": "wave_i",
                        "wave_label": WaveLabel.I.value,
                        "start_price": p1.price,
                        "end_price": p2.price,
                        "confidence": confidence,
                        "pattern_completeness": pattern_completeness,
                    },
                )

        return None


class WaveVRecognizer(_WavePatternBase):
    """Recognizes Wave V (Final impulse wave) - completion of impulse sequence."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="wave_v",
            default_lookback=50,
            default_min_pivot_distance=3,
        )

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize Wave V patterns at the given index."""
        validated_index = self._resolve_index(data, index, multi_timeframe_data)
        if validated_index is None:
            return None
        index = validated_index

        pivots = self._extract_global_pivots(data, index, lookback_divisor=2)

        if len(pivots) < 5:
            return None

        # Look for 5-wave impulse completion (Wave V)
        recent_pivots = pivots[-5:]  # Last 5 pivots

        if len(recent_pivots) >= 5:
            w1, w2, w3, w4, w5 = recent_pivots

            # Check for impulse wave structure
            if (
                w1.price < w3.price > w5.price
                and w2.price < w1.price  # Waves 1,3,5 trending up
                and w4.price < w3.price
                and w3.price > w1.price  # Corrections
                and w5.price > w3.price
            ):  # Progression
                # Wave V should be at the end
                if abs(index - w5.position) <= 2:  # Near completion
                    # Calculate dynamic confidence based on wave structure quality
                    wave1_length = abs(w1.price - w2.price)
                    wave3_length = abs(w3.price - w2.price)
                    wave5_length = abs(w5.price - w4.price)
                    total_length = abs(w5.price - w1.price)

                    # Wave 3 should be the strongest (longest)
                    wave3_ratio = (
                        wave3_length / total_length if total_length > 0 else 0.5
                    )
                    pattern_completeness = min(
                        1.0, wave3_ratio * 1.5
                    )  # Boost confidence for strong wave 3

                    # Base confidence from wave structure quality (slightly higher for final wave)
                    base_confidence = min(0.9, 0.75 + pattern_completeness * 0.15)

                    confidence = self._calculate_wave_confidence(
                        data,
                        index,
                        base_confidence=base_confidence,
                        pattern_completeness=pattern_completeness,
                        trend_lookback=20,
                        candle_size_expected=0.6,
                        price_movement_expected=0.8,
                    )
                    direction = confidence  # Bullish completion

                    return SignalResult(
                        signal_type="wave_v",
                        strength=confidence,
                        direction=direction,
                        description="Elliott Wave V (Final impulse wave completion)",
                        timestamp=data.index[index],
                        confidence=confidence,
                        metadata={
                            "pattern": "wave_v",
                            "wave_label": WaveLabel.V.value,
                            "wave_structure": [w.price for w in recent_pivots],
                            "confidence": confidence,
                            "pattern_completeness": pattern_completeness,
                        },
                    )

        return None


class WaveYRecognizer(_WavePatternBase):
    """Recognizes Wave Y (Terminal wave in complex corrections)."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="wave_y",
            default_lookback=60,
            default_min_pivot_distance=3,
        )

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize Wave Y patterns at the given index."""
        validated_index = self._resolve_index(data, index, multi_timeframe_data)
        if validated_index is None:
            return None
        index = validated_index

        pivots = self._extract_global_pivots(data, index, lookback_divisor=2)

        if len(pivots) < 7:
            return None

        # Look for complex correction with W-X-Y structure
        recent_pivots = pivots[-7:]  # Last 7 pivots for W-X-Y

        if len(recent_pivots) >= 7:
            # W-X-Y structure: W (first correction), X (connecting), Y (terminal)
            w_start, w_end, x_start, x_end, y_start, y_end, current = recent_pivots

            # Check for complex correction pattern
            if abs(w_end.price - w_start.price) > abs(
                x_end.price - x_start.price
            ) and abs(  # W > X
                y_end.price - y_start.price
            ) > abs(x_end.price - x_start.price):  # Y > X
                # Wave Y should be completing
                if abs(index - y_end.position) <= 2:
                    direction = -1 if y_end.price < y_start.price else 1

                    # Calculate dynamic confidence based on complex correction structure
                    w_length = abs(w_end.price - w_start.price)
                    x_length = abs(x_end.price - x_start.price)
                    y_length = abs(y_end.price - y_start.price)

                    # In complex corrections, Y should be larger than X (connecting wave)
                    y_x_ratio = y_length / x_length if x_length > 0 else 1.0
                    pattern_completeness = min(
                        1.0, y_x_ratio * 0.5
                    )  # Higher confidence for dominant Y wave

                    # Base confidence from complex correction structure
                    base_confidence = min(0.85, 0.7 + pattern_completeness * 0.15)

                    confidence = self._calculate_wave_confidence(
                        data,
                        index,
                        base_confidence=base_confidence,
                        pattern_completeness=pattern_completeness,
                        trend_lookback=25,
                        candle_size_expected=0.5,
                        price_movement_expected=0.6,
                    )
                    direction = direction * confidence

                    return SignalResult(
                        signal_type="wave_y",
                        strength=confidence,
                        direction=direction,
                        description="Elliott Wave Y (Terminal wave in complex correction)",
                        timestamp=data.index[index],
                        metadata={
                            "pattern": "wave_y",
                            "wave_label": WaveLabel.Y.value,
                            "w_length": abs(w_end.price - w_start.price),
                            "x_length": abs(x_end.price - x_start.price),
                            "y_length": abs(y_end.price - y_start.price),
                            "confidence": confidence,
                            "pattern_completeness": pattern_completeness,
                        },
                    )

        return None


class WavePRecognizer(_WavePatternBase):
    """Recognizes Wave P (Irregular correction wave)."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="wave_p",
            default_lookback=40,
            default_min_pivot_distance=3,
        )

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize Wave P (irregular correction) patterns at the given index."""
        validated_index = self._resolve_index(data, index, multi_timeframe_data)
        if validated_index is None:
            return None
        index = validated_index

        pivots = self._extract_global_pivots(data, index, lookback_divisor=2)

        if len(pivots) < 3:
            return None

        # Look for irregular correction (Wave P)
        recent_pivots = pivots[-3:]  # Last 3 pivots

        if len(recent_pivots) >= 3:
            start, middle, end = recent_pivots

            # Irregular correction: middle point exceeds start (overshoots)
            if (
                start.price < middle.price > end.price and end.price > start.price
            ) or (  # Bullish irregular
                start.price > middle.price < end.price and end.price < start.price
            ):  # Bearish irregular
                # Check if middle exceeds start significantly (irregular characteristic)
                overshoot_ratio = abs(middle.price - start.price) / abs(
                    end.price - start.price
                )
                if overshoot_ratio > 1.2:  # More than 20% overshoot
                    direction: float = -1.0 if end.price < start.price else 1.0

                    # Calculate dynamic confidence based on overshoot severity
                    pattern_completeness = min(
                        1.0, (overshoot_ratio - 1.2) * 2.0
                    )  # Higher confidence for more severe overshoots

                    # Base confidence from overshoot severity
                    base_confidence = min(0.8, 0.6 + pattern_completeness * 0.2)

                    confidence = self._calculate_wave_confidence(
                        data,
                        index,
                        base_confidence=base_confidence,
                        pattern_completeness=pattern_completeness,
                        trend_lookback=15,
                        candle_size_expected=0.5,
                        price_movement_expected=0.7,
                    )
                    direction = direction * confidence

                    return SignalResult(
                        signal_type="wave_p",
                        strength=confidence,
                        direction=direction,
                        description="Elliott Wave P (Irregular correction wave)",
                        timestamp=data.index[index],
                        confidence=confidence,
                        metadata={
                            "pattern": "wave_p",
                            "wave_label": WaveLabel.P.value,
                            "overshoot_ratio": overshoot_ratio,
                            "start_price": start.price,
                            "middle_price": middle.price,
                            "end_price": end.price,
                            "confidence": confidence,
                            "pattern_completeness": pattern_completeness,
                        },
                    )

        return None


class WaveNRecognizer(_WavePatternBase):
    """Recognizes Wave N (Complex correction wave)."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="wave_n",
            default_lookback=50,
            default_min_pivot_distance=3,
        )

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize Wave N (complex correction) patterns at the given index."""
        validated_index = self._resolve_index(data, index, multi_timeframe_data)
        if validated_index is None:
            return None
        index = validated_index

        pivots = self._extract_global_pivots(data, index, lookback_divisor=2)

        if len(pivots) < 5:
            return None

        # Look for complex correction pattern (Wave N)
        recent_pivots = pivots[-5:]  # Last 5 pivots

        if len(recent_pivots) >= 5:
            p1, p2, p3, p4, p5 = recent_pivots

            # Complex correction: multiple swings, final move exceeds initial
            if (
                p1.price < p3.price < p5.price and p2.price > p4.price
            ) or (  # Bullish complex
                p1.price > p3.price > p5.price and p2.price < p4.price
            ):  # Bearish complex
                # Check for complexity (multiple direction changes)
                direction_changes = 0
                for i in range(1, len(recent_pivots)):
                    if (recent_pivots[i].price - recent_pivots[i - 1].price) * (
                        recent_pivots[i + 1].price - recent_pivots[i].price
                        if i + 1 < len(recent_pivots)
                        else 1
                    ) < 0:
                        direction_changes += 1

                if direction_changes >= 2:  # At least 2 direction changes
                    direction: float = 1.0 if p5.price > p1.price else -1.0

                    # Calculate dynamic confidence based on complexity
                    pattern_completeness = min(
                        1.0, direction_changes * 0.3
                    )  # Higher confidence for more complex patterns

                    # Base confidence from complexity level
                    base_confidence = min(0.85, 0.65 + pattern_completeness * 0.2)

                    confidence = self._calculate_wave_confidence(
                        data,
                        index,
                        base_confidence=base_confidence,
                        pattern_completeness=pattern_completeness,
                        trend_lookback=20,
                        candle_size_expected=0.5,
                        price_movement_expected=0.6,
                    )
                    direction = direction * confidence

                    return SignalResult(
                        signal_type="wave_n",
                        strength=confidence,
                        direction=direction,
                        description="Elliott Wave N (Complex correction wave)",
                        timestamp=data.index[index],
                        confidence=confidence,
                        metadata={
                            "pattern": "wave_n",
                            "wave_label": WaveLabel.N.value,
                            "direction_changes": direction_changes,
                            "start_price": p1.price,
                            "end_price": p5.price,
                            "confidence": confidence,
                            "pattern_completeness": pattern_completeness,
                        },
                    )

        return None


class WaveSRecognizer(_WavePatternBase):
    """Recognizes Wave S (Secondary correction wave)."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="wave_s",
            default_lookback=35,
            default_min_pivot_distance=3,
        )

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize Wave S (secondary correction) patterns at the given index."""
        validated_index = self._resolve_index(data, index, multi_timeframe_data)
        if validated_index is None:
            return None
        index = validated_index

        pivots = self._extract_global_pivots(data, index, lookback_divisor=2)

        if len(pivots) < 3:
            return None

        # Look for secondary correction (Wave S)
        recent_pivots = pivots[-3:]  # Last 3 pivots

        if len(recent_pivots) >= 3:
            start, middle, end = recent_pivots

            # Secondary correction: deeper than typical correction
            total_range = abs(start.price - end.price)
            correction_depth = (
                abs(middle.price - start.price) / total_range if total_range > 0 else 0
            )

            # Secondary corrections are typically 50-80% of the prior move
            if 0.5 < correction_depth < 0.8:
                # Check if it's a secondary correction (following a primary move)
                prior_trend: float = 1.0 if start.price < end.price else -1.0
                correction_direction = 1 if middle.price > start.price else -1

                if (
                    prior_trend != correction_direction
                ):  # Correction opposes prior trend
                    direction = -prior_trend  # Signal continuation of prior trend

                    # Calculate dynamic confidence based on correction depth
                    # Ideal correction depth is around 0.618 Fibonacci retracement
                    ideal_depth = 0.618
                    pattern_completeness = (
                        1.0 - abs(correction_depth - ideal_depth) * 2
                    )  # Closer to ideal is better
                    pattern_completeness = max(
                        0.0, pattern_completeness
                    )  # Ensure non-negative

                    # Base confidence from correction depth quality
                    base_confidence = min(0.8, 0.6 + pattern_completeness * 0.2)

                    confidence = self._calculate_wave_confidence(
                        data,
                        index,
                        base_confidence=base_confidence,
                        pattern_completeness=pattern_completeness,
                        trend_lookback=15,
                        candle_size_expected=0.5,
                        price_movement_expected=0.6,
                    )
                    direction = direction * confidence

                    return SignalResult(
                        signal_type="wave_s",
                        strength=confidence,
                        direction=direction,
                        description="Elliott Wave S (Secondary correction wave)",
                        timestamp=data.index[index],
                        confidence=confidence,
                        metadata={
                            "pattern": "wave_s",
                            "wave_label": WaveLabel.S.value,
                            "correction_depth": correction_depth,
                            "prior_trend": prior_trend,
                            "start_price": start.price,
                            "middle_price": middle.price,
                            "end_price": end.price,
                            "confidence": confidence,
                            "pattern_completeness": pattern_completeness,
                        },
                    )

        return None
