"""
Wave Counting Module

This module provides pattern recognition for wave counting analysis,
primarily based on Elliott Wave Theory including impulse waves, corrective waves,
and various wave patterns.
"""

from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Any

import pandas as pd

from ztb.trading.environment.constants import EPSILON

from .base import PatternRecognizer, SignalResult


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
    def identify_wave_structure(pivots: List[WavePoint]) -> Optional[Dict]:
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


class ImpulseWaveRecognizer(PatternRecognizer):
    """Recognizes Elliott Wave impulse patterns (5-wave structures)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.lookback_period = config.get('lookback_period', 50) if config else 50
        self.min_pivot_distance = config.get('min_pivot_distance', 3) if config else 3
        self.wave_analyzer = WaveAnalyzer()

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize impulse wave patterns at the given index."""
        if index < self.lookback_period:
            return None

        # Get pivot points in the lookback period
        lookback_data = data.iloc[max(0, index - self.lookback_period) : index + 1]
        pivots = self.wave_analyzer.find_pivot_points(
            lookback_data,
            lookback=self.lookback_period // 2,
            min_distance=self.min_pivot_distance,
        )

        # Adjust pivot indices to global index
        offset = max(0, index - self.lookback_period)
        adjusted_pivots = [
            WavePoint(p.position + offset, p.price, p.wave_label, p.degree)
            for p in pivots
        ]

        wave_structure = self.wave_analyzer.identify_wave_structure(adjusted_pivots)

        if wave_structure and wave_structure["type"] == WaveType.IMPULSE:
            # Check if we're at or near the completion of wave 5
            if abs(index - wave_structure["completion_index"]) <= 2:  # Within 2 bars
                strength = wave_structure["strength"]
                direction = wave_structure["direction"]

                signal_type = "impulse_wave_completion"

                return SignalResult(
                    signal_type=signal_type,
                    strength=strength,
                    direction=direction,
                    description="Elliott Wave Impulse Pattern (5-wave structure)",
                    timestamp=data.index[index],
                    metadata={
                        "pattern": "impulse_wave",
                        "wave_type": wave_structure["type"].value,
                        "degree": wave_structure["degree"].value,
                        "wave_count": len(wave_structure["waves"]),
                        "confidence": strength,
                    },
                )

        return None


class CorrectiveWaveRecognizer(PatternRecognizer):
    """Recognizes Elliott Wave corrective patterns (ABC structures)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.lookback_period = config.get('lookback_period', 40) if config else 40
        self.min_pivot_distance = config.get('min_pivot_distance', 3) if config else 3
        self.wave_analyzer = WaveAnalyzer()

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize corrective wave patterns at the given index."""
        if index < self.lookback_period:
            return None

        # Get pivot points in the lookback period
        lookback_data = data.iloc[max(0, index - self.lookback_period) : index + 1]
        pivots = self.wave_analyzer.find_pivot_points(
            lookback_data,
            lookback=self.lookback_period // 2,
            min_distance=self.min_pivot_distance,
        )

        # Adjust pivot indices to global index
        offset = max(0, index - self.lookback_period)
        adjusted_pivots = [
            WavePoint(p.position + offset, p.price, p.wave_label, p.degree)
            for p in pivots
        ]

        wave_structure = self.wave_analyzer.identify_wave_structure(adjusted_pivots)

        if wave_structure and wave_structure["type"] == WaveType.CORRECTIVE:
            # Check if we're at or near the completion of wave C
            if abs(index - wave_structure["completion_index"]) <= 2:  # Within 2 bars
                strength = wave_structure["strength"]
                direction = wave_structure["direction"]

                signal_type = "corrective_wave_completion"

                return SignalResult(
                    signal_type=signal_type,
                    strength=strength,
                    direction=direction,
                    description="Elliott Wave Corrective Pattern (ABC structure)",
                    timestamp=data.index[index],
                    metadata={
                        "pattern": "corrective_wave",
                        "wave_type": wave_structure["type"].value,
                        "degree": wave_structure["degree"].value,
                        "wave_count": len(wave_structure["waves"]),
                        "confidence": strength,
                    },
                )

        return None


class WaveExtensionRecognizer(PatternRecognizer):
    """Recognizes wave extensions and truncations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.lookback_period = config.get('lookback_period', 60) if config else 60
        self.wave_analyzer = WaveAnalyzer()

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize wave extensions or truncations at the given index."""
        if index < self.lookback_period:
            return None

        lookback_data = data.iloc[max(0, index - self.lookback_period) : index + 1]
        pivots = self.wave_analyzer.find_pivot_points(
            lookback_data, lookback=self.lookback_period // 3
        )

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
                    )
                    strength = min(
                        0.85,
                        0.6
                        + (
                            wave_lengths[1] / max(wave_lengths[0], wave_lengths[2])
                            - 1.5
                        )
                        * 0.1,
                    )

                    return SignalResult(
                        signal_type="wave_extension",
                        strength=strength,
                        direction=direction,
                        description="Elliott Wave Extension (extended wave 3)",
                        timestamp=data.index[index],
                        metadata={
                            "pattern": "wave_extension",
                            "extension_ratio": wave_lengths[1]
                            / max(wave_lengths[0], wave_lengths[2]),
                            "confidence": strength,
                        },
                    )

        # Look for truncated waves (wave 5 fails to exceed wave 3)
        if len(recent_pivots) >= 5:
            w1, w2, w3, w4, w5 = recent_pivots[-5:]

            # Check for truncation: wave 5 doesn't exceed wave 3 high (in bullish case)
            if w3.price > w1.price and w5.price < w3.price:
                strength = 0.7

                return SignalResult(
                    signal_type="wave_truncation",
                    strength=strength,
                    direction=-1,  # Bearish signal (failure)
                    description="Elliott Wave Truncation (wave 5 fails to exceed wave 3)",
                    timestamp=data.index[index],
                    metadata={
                        "pattern": "wave_truncation",
                        "wave3_high": w3.price,
                        "wave5_high": w5.price,
                        "confidence": strength,
                    },
                )

        return None


class WaveIRecognizer(PatternRecognizer):
    """Recognizes Wave I (Initial impulse wave) - start of a new trend."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.lookback_period = config.get('lookback_period', 30) if config else 30
        self.min_pivot_distance = config.get('min_pivot_distance', 3) if config else 3
        self.wave_analyzer = WaveAnalyzer()

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Wave I patterns at the given index."""
        if index < self.lookback_period:
            return None

        # Get recent pivot points
        lookback_data = data.iloc[
            max(0, index - self.lookback_period) : index + 1
        ].copy()
        pivots = self.wave_analyzer.find_pivot_points(
            lookback_data,
            lookback=self.lookback_period // 2,
            min_distance=self.min_pivot_distance,
        )

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
                direction = 1 if p2.price > p1.price else -1
                strength = min(
                    0.8,
                    abs(p2.price - p1.price)
                    / data["close"].iloc[index - self.lookback_period : index].mean()
                    * 10,
                )

                return SignalResult(
                    signal_type="wave_i",
                    strength=strength,
                    direction=direction,
                    description="Elliott Wave I (Initial impulse wave)",
                    timestamp=data.index[index],
                    metadata={
                        "pattern": "wave_i",
                        "wave_label": WaveLabel.I.value,
                        "start_price": p1.price,
                        "end_price": p2.price,
                        "confidence": strength,
                    },
                )

        return None


class WaveVRecognizer(PatternRecognizer):
    """Recognizes Wave V (Final impulse wave) - completion of impulse sequence."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.lookback_period = config.get('lookback_period', 50) if config else 50
        self.min_pivot_distance = config.get('min_pivot_distance', 3) if config else 3
        self.wave_analyzer = WaveAnalyzer()

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Wave V patterns at the given index."""
        if index < self.lookback_period:
            return None

        # Get pivot points in the lookback period
        lookback_data = data.iloc[
            max(0, index - self.lookback_period) : index + 1
        ].copy()
        pivots = self.wave_analyzer.find_pivot_points(
            lookback_data,
            lookback=self.lookback_period // 2,
            min_distance=self.min_pivot_distance,
        )

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
                    strength = 0.85
                    direction = 1  # Bullish completion

                    return SignalResult(
                        signal_type="wave_v",
                        strength=strength,
                        direction=direction,
                        description="Elliott Wave V (Final impulse wave completion)",
                        timestamp=data.index[index],
                        metadata={
                            "pattern": "wave_v",
                            "wave_label": WaveLabel.V.value,
                            "wave_structure": [w.price for w in recent_pivots],
                            "confidence": strength,
                        },
                    )

        return None


class WaveYRecognizer(PatternRecognizer):
    """Recognizes Wave Y (Terminal wave in complex corrections)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.lookback_period = config.get('lookback_period', 60) if config else 60
        self.min_pivot_distance = config.get('min_pivot_distance', 3) if config else 3
        self.wave_analyzer = WaveAnalyzer()

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Wave Y patterns at the given index."""
        if index < self.lookback_period:
            return None

        # Get pivot points in the lookback period
        lookback_data = data.iloc[
            max(0, index - self.lookback_period) : index + 1
        ].copy()
        pivots = self.wave_analyzer.find_pivot_points(
            lookback_data,
            lookback=self.lookback_period // 2,
            min_distance=self.min_pivot_distance,
        )

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
                    strength = 0.8

                    return SignalResult(
                        signal_type="wave_y",
                        strength=strength,
                        direction=direction,
                        description="Elliott Wave Y (Terminal wave in complex correction)",
                        timestamp=data.index[index],
                        metadata={
                            "pattern": "wave_y",
                            "wave_label": WaveLabel.Y.value,
                            "w_length": abs(w_end.price - w_start.price),
                            "x_length": abs(x_end.price - x_start.price),
                            "y_length": abs(y_end.price - y_start.price),
                            "confidence": strength,
                        },
                    )

        return None


class WavePRecognizer(PatternRecognizer):
    """Recognizes Wave P (Irregular correction wave)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.lookback_period = config.get('lookback_period', 40) if config else 40
        self.min_pivot_distance = config.get('min_pivot_distance', 3) if config else 3
        self.wave_analyzer = WaveAnalyzer()

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Wave P (irregular correction) patterns at the given index."""
        if index < self.lookback_period:
            return None

        # Get pivot points in the lookback period
        lookback_data = data.iloc[
            max(0, index - self.lookback_period) : index + 1
        ].copy()
        pivots = self.wave_analyzer.find_pivot_points(
            lookback_data,
            lookback=self.lookback_period // 2,
            min_distance=self.min_pivot_distance,
        )

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
                    direction = -1 if end.price < start.price else 1
                    strength = min(0.75, 0.5 + overshoot_ratio * 0.1)

                    return SignalResult(
                        signal_type="wave_p",
                        strength=strength,
                        direction=direction,
                        description="Elliott Wave P (Irregular correction wave)",
                        timestamp=data.index[index],
                        metadata={
                            "pattern": "wave_p",
                            "wave_label": WaveLabel.P.value,
                            "overshoot_ratio": overshoot_ratio,
                            "start_price": start.price,
                            "middle_price": middle.price,
                            "end_price": end.price,
                            "confidence": strength,
                        },
                    )

        return None


class WaveNRecognizer(PatternRecognizer):
    """Recognizes Wave N (Complex correction wave)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.lookback_period = config.get('lookback_period', 50) if config else 50
        self.min_pivot_distance = config.get('min_pivot_distance', 3) if config else 3
        self.wave_analyzer = WaveAnalyzer()

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Wave N (complex correction) patterns at the given index."""
        if index < self.lookback_period:
            return None

        # Get pivot points in the lookback period
        lookback_data = data.iloc[
            max(0, index - self.lookback_period) : index + 1
        ].copy()
        pivots = self.wave_analyzer.find_pivot_points(
            lookback_data,
            lookback=self.lookback_period // 2,
            min_distance=self.min_pivot_distance,
        )

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
                    direction = 1 if p5.price > p1.price else -1
                    strength = min(0.8, 0.6 + direction_changes * 0.1)

                    return SignalResult(
                        signal_type="wave_n",
                        strength=strength,
                        direction=direction,
                        description="Elliott Wave N (Complex correction wave)",
                        timestamp=data.index[index],
                        metadata={
                            "pattern": "wave_n",
                            "wave_label": WaveLabel.N.value,
                            "direction_changes": direction_changes,
                            "start_price": p1.price,
                            "end_price": p5.price,
                            "confidence": strength,
                        },
                    )

        return None


class WaveSRecognizer(PatternRecognizer):
    """Recognizes Wave S (Secondary correction wave)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.lookback_period = config.get('lookback_period', 35) if config else 35
        self.min_pivot_distance = config.get('min_pivot_distance', 3) if config else 3
        self.wave_analyzer = WaveAnalyzer()

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Wave S (secondary correction) patterns at the given index."""
        if index < self.lookback_period:
            return None

        # Get pivot points in the lookback period
        lookback_data = data.iloc[
            max(0, index - self.lookback_period) : index + 1
        ].copy()
        pivots = self.wave_analyzer.find_pivot_points(
            lookback_data,
            lookback=self.lookback_period // 2,
            min_distance=self.min_pivot_distance,
        )

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
                prior_trend = 1 if start.price < end.price else -1
                correction_direction = 1 if middle.price > start.price else -1

                if (
                    prior_trend != correction_direction
                ):  # Correction opposes prior trend
                    direction = -prior_trend  # Signal continuation of prior trend
                    strength = min(0.75, 0.5 + correction_depth * 0.5)

                    return SignalResult(
                        signal_type="wave_s",
                        strength=strength,
                        direction=direction,
                        description="Elliott Wave S (Secondary correction wave)",
                        timestamp=data.index[index],
                        metadata={
                            "pattern": "wave_s",
                            "wave_label": WaveLabel.S.value,
                            "correction_depth": correction_depth,
                            "prior_trend": prior_trend,
                            "start_price": start.price,
                            "middle_price": middle.price,
                            "end_price": end.price,
                            "confidence": strength,
                        },
                    )

        return None
