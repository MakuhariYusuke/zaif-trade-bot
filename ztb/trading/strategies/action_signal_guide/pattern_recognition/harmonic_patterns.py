"""
Harmonic Patterns Module

This module provides pattern recognition for harmonic patterns including
Gartley, Butterfly, Bat, Crab, and other geometric price patterns based
on Fibonacci ratios and symmetry.
"""

from typing import Any, Dict, List, NamedTuple, Optional

import pandas as pd

from .base import CandlestickPatternRecognizer, SignalResult


class HarmonicPoint(NamedTuple):
    position: int
    price: float
    label: str


class HarmonicAnalyzer:
    """Utility class for harmonic pattern calculations and validation."""

    def __init__(self) -> None:
        # Cache for pivot points to avoid recalculation
        self._pivot_cache: Dict[str, List[HarmonicPoint]] = {}
        self._cache_max_size = 10  # Limit cache size to prevent memory issues

    # Fibonacci ratios for harmonic patterns
    GARTLEY_RATIOS = {
        "XA": 1.0,  # Base move
        "AB": 0.618,  # 61.8% retracement of XA
        "BC": 0.382,  # 38.2% to 88.6% retracement of AB
        "CD": 1.272,  # 127.2% extension of BC
        "AD": 0.786,  # 78.6% retracement of XA
    }

    BUTTERFLY_RATIOS = {"XA": 1.0, "AB": 0.786, "BC": 0.382, "CD": 1.618, "AD": 1.272}

    BAT_RATIOS = {"XA": 1.0, "AB": 0.382, "BC": 0.382, "CD": 1.618, "AD": 0.886}

    CRAB_RATIOS = {"XA": 1.0, "AB": 0.382, "BC": 0.382, "CD": 2.618, "AD": 1.618}

    @staticmethod
    def calculate_fibonacci_ratio(
        point1: float, point2: float, target_ratio: float
    ) -> float:
        """
        Calculate the price at a specific Fibonacci ratio from point1 toward point2.

        The result is point1 plus (point2 - point1) multiplied by target_ratio.
        """
        return point1 + (point2 - point1) * target_ratio

    @staticmethod
    def validate_ratio(actual: float, target: float, tolerance: float = 0.05) -> bool:
        """Check if actual ratio is within tolerance of target ratio."""
        return abs(actual - target) / target <= tolerance

    def find_harmonic_pattern(
        self,
        data: pd.DataFrame,
        pattern_type: str,
        start_idx: int,
        tolerance: float = 0.05,
    ) -> Optional[Dict]:
        """Find a specific harmonic pattern starting from start_idx."""
        """Find a specific harmonic pattern starting from start_idx."""
        if start_idx >= len(data) - 4:
            return None

        ratios = getattr(HarmonicAnalyzer, f"{pattern_type.upper()}_RATIOS")

        # Get potential pivot points
        search_window = min(50, len(data) - start_idx)
        window_data = data.iloc[start_idx : start_idx + search_window]

        # Find X, A, B, C, D points using zigzag or pivot method
        pivots = self._get_pivot_points(window_data)

        if len(pivots) < 5:
            return None

        # Test different combinations of 5 pivots
        for i in range(len(pivots) - 4):
            x, a, b, c, d = pivots[i : i + 5]

            # Validate the pattern ratios
            if HarmonicAnalyzer._validate_harmonic_ratios(
                x.price, a.price, b.price, c.price, d.price, ratios, tolerance
            ):
                # Calculate pattern completion and target
                completion_price = d.price
                target_price = HarmonicAnalyzer._calculate_pattern_target(
                    x.price, a.price, b.price, c.price, d.price, pattern_type
                )

                # Determine direction (bullish or bearish pattern)
                direction = 1 if d.price > c.price else -1

                return {
                    "pattern_type": pattern_type,
                    "points": [x, a, b, c, d],
                    "completion_index": window_data.index[d.position],
                    "completion_price": completion_price,
                    "target_price": target_price,
                    "direction": direction,
                    "strength": HarmonicAnalyzer._calculate_pattern_strength(
                        ratios, tolerance, pattern_type
                    ),
                    "ratios": ratios,
                }

        return None

    def _get_pivot_points(
        self, data: pd.DataFrame, min_distance: int = 3
    ) -> List[HarmonicPoint]:
        """Find pivot points in the data for harmonic pattern detection with caching."""
        # Create cache key based on data range and parameters
        start_idx = data.index[0] if hasattr(data.index, "__getitem__") else 0
        end_idx = (
            data.index[-1] if hasattr(data.index, "__getitem__") else len(data) - 1
        )
        cache_key = f"{start_idx}_{end_idx}_{min_distance}"

        # Check cache first
        if cache_key in self._pivot_cache:
            return self._pivot_cache[cache_key]

        # Calculate pivots
        highs = data["high"]
        lows = data["low"]

        pivots: List[HarmonicPoint] = []

        for i in range(2, len(data) - 2):
            # Pivot high
            if (
                highs.iloc[i] > highs.iloc[i - 1]
                and highs.iloc[i] > highs.iloc[i - 2]
                and highs.iloc[i] > highs.iloc[i + 1]
                and highs.iloc[i] > highs.iloc[i + 2]
            ):
                if not pivots or (i - pivots[-1].position) >= min_distance:
                    pivots.append(HarmonicPoint(i, highs.iloc[i], "H"))

            # Pivot low
            elif (
                lows.iloc[i] < lows.iloc[i - 1]
                and lows.iloc[i] < lows.iloc[i - 2]
                and lows.iloc[i] < lows.iloc[i + 1]
                and lows.iloc[i] < lows.iloc[i + 2]
            ):
                if not pivots or (i - pivots[-1].position) >= min_distance:
                    pivots.append(HarmonicPoint(i, lows.iloc[i], "L"))

        # Cache the result (limit cache size)
        if len(self._pivot_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._pivot_cache))
            del self._pivot_cache[oldest_key]

        self._pivot_cache[cache_key] = pivots
        return pivots

    @staticmethod
    def _validate_harmonic_ratios(
        x: float,
        a: float,
        b: float,
        c: float,
        d: float,
        ratios: Dict[str, float],
        tolerance: float,
    ) -> bool:
        """Validate if the points form a valid harmonic pattern."""
        # Calculate actual ratios
        xa_range = abs(a - x)
        if xa_range == 0:
            return False

        ab_ratio = abs(b - a) / xa_range
        bc_ratio = abs(c - b) / abs(b - a) if abs(b - a) > 0 else 0
        cd_ratio = abs(d - c) / abs(c - b) if abs(c - b) > 0 else 0
        ad_ratio = abs(d - a) / xa_range

        # Validate each ratio
        validations = [
            HarmonicAnalyzer.validate_ratio(ab_ratio, ratios["AB"], tolerance),
            HarmonicAnalyzer.validate_ratio(bc_ratio, ratios["BC"], tolerance),
            # The CD leg in harmonic patterns often exhibits greater volatility and extension variability in real market data,
            # making it less likely to match the ideal Fibonacci ratio precisely. Therefore, we allow a wider tolerance for CD
            # to improve pattern detection robustness and reduce false negatives.
            HarmonicAnalyzer.validate_ratio(cd_ratio, ratios["CD"], tolerance * 2),
            HarmonicAnalyzer.validate_ratio(ad_ratio, ratios["AD"], tolerance),
        ]

        return all(validations)

    @staticmethod
    def _calculate_pattern_target(
        x: float, a: float, b: float, c: float, d: float, pattern_type: str
    ) -> float:
        """Calculate the price target for the harmonic pattern."""
        # Common target calculation: projection from D
        cd_move = d - c
        target = d + cd_move  # Simple projection

        # Pattern-specific adjustments
        if pattern_type.upper() == "GARTLEY":
            # Gartley target is often at 161.8% of CD from D
            target = d + cd_move * 0.618
        elif pattern_type.upper() == "BUTTERFLY":
            target = d + cd_move * 0.786
        elif pattern_type.upper() == "BAT":
            target = d + cd_move * 0.382
        elif pattern_type.upper() == "CRAB":
            target = d + cd_move * 0.382

        return target

    @staticmethod
    def _calculate_pattern_strength(
        ratios: Dict[str, float], tolerance: float, pattern_type: str = ""
    ) -> float:
        """Calculate the strength/confidence of the harmonic pattern."""
        # Base strength on how strict the ratios are
        base_strength = 0.7

        # Bonus for patterns with more precise ratios
        strict_patterns = ["CRAB", "BAT"]
        if pattern_type.upper() in strict_patterns:
            base_strength += 0.1

        # Adjust for tolerance (tighter tolerance = higher strength)
        tolerance_bonus = (0.1 - tolerance) * 2
        base_strength += max(0, tolerance_bonus)

        return min(0.9, base_strength)


class GartleyRecognizer(CandlestickPatternRecognizer):
    """Recognizes Gartley harmonic patterns."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.lookback_period = config.get("lookback_period", 60) if config else 60
        self.tolerance = config.get("tolerance", 0.05) if config else 0.05
        self.harmonic_analyzer = HarmonicAnalyzer()

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Gartley pattern at the given index."""
        if index < self.lookback_period:
            return None

        # Search for Gartley pattern in recent data (reverse order for better performance)
        for start_idx in range(
            min(len(data) - 5, index - 4), max(0, index - self.lookback_period) - 1, -1
        ):
            pattern = self.harmonic_analyzer.find_harmonic_pattern(
                data, "GARTLEY", start_idx, self.tolerance
            )

            if pattern and abs(pattern["completion_index"] - index) <= 1:
                # Calculate pattern completeness based on how close price is to completion
                price_deviation = (
                    abs(data.iloc[index]["close"] - pattern["completion_price"])
                    / pattern["completion_price"]
                )
                pattern_completeness = 1.0 - min(
                    1.0, price_deviation * 10
                )  # Closer to completion = higher completeness

                # Use pattern confidence calculation
                pattern_factors = {
                    "trend_strength": self._calculate_trend_strength(data, index, 20),
                    "candle_size": self._calculate_candle_size_confidence(
                        data, index, 0.6
                    ),  # Harmonic patterns are structural
                    "price_movement": self._calculate_price_movement_confidence(
                        data, index, 0.7
                    ),  # Approaching completion target
                    "pattern_completeness": pattern_completeness,  # How close price is to the harmonic completion
                }

                confidence = self._calculate_pattern_confidence(
                    data, index, pattern_factors, base_confidence=pattern["strength"]
                )
                direction = pattern["direction"]

                signal_type = "gartley_bullish" if direction == 1 else "gartley_bearish"

                return SignalResult(
                    signal_type=signal_type,
                    strength=confidence,
                    direction=direction,
                    description="Gartley Harmonic Pattern",
                    timestamp=data.index[index],
                    confidence=confidence,
                    metadata={
                        "pattern": "gartley",
                        "completion_price": pattern["completion_price"],
                        "target_price": pattern["target_price"],
                        "confidence": confidence,
                        "pattern_completeness": pattern_completeness,
                    },
                )

        return None


class ButterflyRecognizer(CandlestickPatternRecognizer):
    """Recognizes Butterfly harmonic patterns."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.lookback_period = config.get("lookback_period", 60) if config else 60
        self.tolerance = config.get("tolerance", 0.05) if config else 0.05
        self.harmonic_analyzer = HarmonicAnalyzer()

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Butterfly pattern at the given index."""
        if index < self.lookback_period:
            return None

        # Search for Butterfly pattern in recent data (reverse order for better performance)
        for start_idx in range(
            min(len(data) - 5, index - 4), max(0, index - self.lookback_period) - 1, -1
        ):
            pattern = self.harmonic_analyzer.find_harmonic_pattern(
                data, "BUTTERFLY", start_idx, self.tolerance
            )

            if pattern and abs(pattern["completion_index"] - index) <= 1:
                # Calculate pattern completeness based on how close price is to completion
                price_deviation = (
                    abs(data.iloc[index]["close"] - pattern["completion_price"])
                    / pattern["completion_price"]
                )
                pattern_completeness = 1.0 - min(
                    1.0, price_deviation * 10
                )  # Closer to completion = higher completeness

                # Use pattern confidence calculation
                pattern_factors = {
                    "trend_strength": self._calculate_trend_strength(data, index, 20),
                    "candle_size": self._calculate_candle_size_confidence(
                        data, index, 0.6
                    ),  # Harmonic patterns are structural
                    "price_movement": self._calculate_price_movement_confidence(
                        data, index, 0.7
                    ),  # Approaching completion target
                    "pattern_completeness": pattern_completeness,  # How close price is to the harmonic completion
                }

                confidence = self._calculate_pattern_confidence(
                    data, index, pattern_factors, base_confidence=pattern["strength"]
                )
                direction = pattern["direction"]

                signal_type = (
                    "butterfly_bullish" if direction == 1 else "butterfly_bearish"
                )

                return SignalResult(
                    signal_type=signal_type,
                    strength=confidence,
                    direction=direction,
                    description="Butterfly Harmonic Pattern",
                    timestamp=data.index[index],
                    metadata={
                        "pattern": "butterfly",
                        "completion_price": pattern["completion_price"],
                        "target_price": pattern["target_price"],
                        "confidence": confidence,
                        "pattern_completeness": pattern_completeness,
                    },
                )

        return None


class BatRecognizer(CandlestickPatternRecognizer):
    """Recognizes Bat harmonic patterns."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.lookback_period = config.get("lookback_period", 60) if config else 60
        self.tolerance = config.get("tolerance", 0.05) if config else 0.05
        self.harmonic_analyzer = HarmonicAnalyzer()

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Bat pattern at the given index."""
        if index < self.lookback_period:
            return None

        # Search for Bat pattern in recent data (reverse order for better performance)
        for start_idx in range(
            min(len(data) - 5, index - 4), max(0, index - self.lookback_period) - 1, -1
        ):
            pattern = self.harmonic_analyzer.find_harmonic_pattern(
                data, "BAT", start_idx, self.tolerance
            )

            if pattern and abs(pattern["completion_index"] - index) <= 1:
                # Calculate pattern completeness based on how close price is to completion
                price_deviation = (
                    abs(data.iloc[index]["close"] - pattern["completion_price"])
                    / pattern["completion_price"]
                )
                pattern_completeness = 1.0 - min(
                    1.0, price_deviation * 10
                )  # Closer to completion = higher completeness

                # Use pattern confidence calculation
                pattern_factors = {
                    "trend_strength": self._calculate_trend_strength(data, index, 20),
                    "candle_size": self._calculate_candle_size_confidence(
                        data, index, 0.6
                    ),  # Harmonic patterns are structural
                    "price_movement": self._calculate_price_movement_confidence(
                        data, index, 0.7
                    ),  # Approaching completion target
                    "pattern_completeness": pattern_completeness,  # How close price is to the harmonic completion
                }

                confidence = self._calculate_pattern_confidence(
                    data, index, pattern_factors, base_confidence=pattern["strength"]
                )
                direction = pattern["direction"]

                signal_type = "bat_bullish" if direction == 1 else "bat_bearish"

                return SignalResult(
                    signal_type=signal_type,
                    strength=confidence,
                    direction=direction,
                    description="Bat Harmonic Pattern",
                    timestamp=data.index[index],
                    metadata={
                        "pattern": "bat",
                        "completion_price": pattern["completion_price"],
                        "target_price": pattern["target_price"],
                        "confidence": confidence,
                        "pattern_completeness": pattern_completeness,
                    },
                )

        return None


class CrabRecognizer(CandlestickPatternRecognizer):
    """Recognizes Crab harmonic patterns."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.lookback_period = config.get("lookback_period", 60) if config else 60
        self.tolerance = config.get("tolerance", 0.05) if config else 0.05
        self.harmonic_analyzer = HarmonicAnalyzer()

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Crab pattern at the given index."""
        if index < self.lookback_period:
            return None

        # Search for Crab pattern in recent data (reverse order for better performance)
        for start_idx in range(
            min(len(data) - 5, index - 4), max(0, index - self.lookback_period) - 1, -1
        ):
            pattern = self.harmonic_analyzer.find_harmonic_pattern(
                data, "CRAB", start_idx, self.tolerance
            )

            if pattern and abs(pattern["completion_index"] - index) <= 1:
                # Calculate pattern completeness based on how close price is to completion
                price_deviation = (
                    abs(data.iloc[index]["close"] - pattern["completion_price"])
                    / pattern["completion_price"]
                )
                pattern_completeness = 1.0 - min(
                    1.0, price_deviation * 10
                )  # Closer to completion = higher completeness

                # Use pattern confidence calculation
                pattern_factors = {
                    "trend_strength": self._calculate_trend_strength(data, index, 20),
                    "candle_size": self._calculate_candle_size_confidence(
                        data, index, 0.6
                    ),  # Harmonic patterns are structural
                    "price_movement": self._calculate_price_movement_confidence(
                        data, index, 0.7
                    ),  # Approaching completion target
                    "pattern_completeness": pattern_completeness,  # How close price is to the harmonic completion
                }

                confidence = self._calculate_pattern_confidence(
                    data, index, pattern_factors, base_confidence=pattern["strength"]
                )
                direction = pattern["direction"]

                signal_type = "crab_bullish" if direction == 1 else "crab_bearish"

                return SignalResult(
                    signal_type=signal_type,
                    strength=confidence,
                    direction=direction,
                    description="Crab Harmonic Pattern",
                    timestamp=data.index[index],
                    confidence=confidence,
                    metadata={
                        "pattern": "crab",
                        "completion_price": pattern["completion_price"],
                        "target_price": pattern["target_price"],
                        "confidence": confidence,
                        "pattern_completeness": pattern_completeness,
                    },
                )

        return None
