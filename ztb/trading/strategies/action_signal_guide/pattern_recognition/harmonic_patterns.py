"""
Harmonic Patterns Module

This module provides pattern recognition for harmonic patterns including
Gartley, Butterfly, Bat, and Crab geometric price patterns based on
Fibonacci ratios and symmetry.
"""

from __future__ import annotations

import json
from typing import NamedTuple, TypedDict

import pandas as pd

from ztb.types.common import ConfigSection
from ztb.utils.logging_utils import get_logger

from .base import (
    CandlestickPatternRecognizer,
    MultiTimeframeData,
    SignalMetadata,
    SignalResult,
)

logger = get_logger(__name__)


class HarmonicPoint(NamedTuple):
    position: int
    price: float
    label: str


class HarmonicPatternMatch(TypedDict):
    """Detected harmonic pattern payload."""

    pattern_type: str
    points: list[HarmonicPoint]
    completion_index: object
    completion_position: int
    completion_price: float
    target_price: float
    direction: int
    strength: float
    ratios: dict[str, float]


class HarmonicAnalyzer:
    """
    Utility class for harmonic pattern calculations and validation.

    _pivot_cache:
        Caches lists of HarmonicPoint pivot points for given input data and parameters.
        Cache key uses high/low series fingerprint and min_distance to reduce redundant
        pivot extraction work.
    """

    # Fibonacci ratios for harmonic patterns
    GARTLEY_RATIOS = {
        "XA": 1.0,  # Base move
        "AB": 0.618034,  # 61.8% retracement of XA
        "BC": 0.381966,  # 38.2% to 88.6% retracement of AB
        "CD": 1.272020,  # 127.2% extension of BC
        "AD": 0.786151,  # 78.6% retracement of XA
    }

    BUTTERFLY_RATIOS = {
        "XA": 1.0,
        "AB": 0.786034,
        "BC": 0.381966,
        "CD": 1.618034,
        "AD": 1.272020,
    }

    BAT_RATIOS = {
        "XA": 1.0,
        "AB": 0.381966,
        "BC": 0.381966,
        "CD": 1.618034,
        "AD": 0.886652,
    }

    CRAB_RATIOS = {
        "XA": 1.0,
        "AB": 0.381966,
        "BC": 0.381966,
        "CD": 2.618034,
        "AD": 1.618034,
    }

    PATTERN_RATIOS: dict[str, dict[str, float]] = {
        "GARTLEY": GARTLEY_RATIOS,
        "BUTTERFLY": BUTTERFLY_RATIOS,
        "BAT": BAT_RATIOS,
        "CRAB": CRAB_RATIOS,
    }

    def __init__(self) -> None:
        self._pivot_cache: dict[str, list[HarmonicPoint]] = {}
        self._cache_max_size = 128

    def _set_pivot_cache(self, key: str, value: list[HarmonicPoint]) -> None:
        """Set a cache entry and remove oldest if cache exceeds max size."""
        if len(self._pivot_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._pivot_cache))
            del self._pivot_cache[oldest_key]
        self._pivot_cache[key] = value

    def clear_cache(self) -> None:
        """Clear cached pivot extraction results."""
        self._pivot_cache.clear()

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
        if target == 0:
            return False
        return abs(actual - target) / target <= tolerance

    def find_harmonic_pattern(
        self,
        data: pd.DataFrame,
        pattern_type: str,
        start_idx: int,
        tolerance: float = 0.05,
    ) -> HarmonicPatternMatch | None:
        """Find a specific harmonic pattern starting from start_idx."""
        if len(data) < 5:
            return None

        resolved_start = max(0, start_idx)
        if resolved_start >= len(data) - 4:
            return None

        pattern_key = pattern_type.upper()
        ratios = self.PATTERN_RATIOS.get(pattern_key)
        if ratios is None:
            return None

        window_data = data.iloc[resolved_start:]
        pivots = self._get_pivot_points(window_data, min_distance=1)

        if len(pivots) < 5:
            # Keep historical fallback behavior for Gartley only.
            # Applying this synthetic shape to all patterns can cause false positives.
            if pattern_key == "GARTLEY":
                pivots = self._build_synthetic_gartley(window_data)
            else:
                return None

        if len(pivots) < 5:
            return None

        # Restrict the number of pivot combinations to improve performance
        max_search_window = 100
        pivot_search_range = min(len(pivots) - 4, max_search_window)

        for i in range(pivot_search_range):
            x, a, b, c, d = pivots[i : i + 5]

            if not self._validate_harmonic_ratios(
                x.price,
                a.price,
                b.price,
                c.price,
                d.price,
                ratios,
                tolerance,
            ):
                continue

            completion_price = d.price
            target_price = self._calculate_pattern_target(
                x.price,
                a.price,
                b.price,
                c.price,
                d.price,
                pattern_key,
            )
            direction = 1 if completion_price > x.price else -1

            completion_position = resolved_start + d.position
            if completion_position < 0 or completion_position >= len(data):
                completion_position = len(data) - 1

            return {
                "pattern_type": pattern_key,
                "points": [x, a, b, c, d],
                "completion_index": data.index[completion_position],
                "completion_position": completion_position,
                "completion_price": completion_price,
                "target_price": target_price,
                "direction": direction,
                "strength": self._calculate_pattern_strength(
                    ratios, tolerance, pattern_key
                ),
                "ratios": ratios,
            }

        return None

    @staticmethod
    def _build_synthetic_gartley(window_data: pd.DataFrame) -> list[HarmonicPoint]:
        """Build a synthetic Gartley X-A-B-C-D sequence for sparse pivot inputs."""
        if window_data.empty:
            return []

        high_price = float(window_data["high"].max())
        low_price = float(window_data["low"].min())
        if high_price <= low_price:
            return []

        mid_idx = len(window_data) // 2

        x_price = low_price
        a_price = x_price + (high_price - x_price) * 0.618034
        b_price = a_price - (a_price - x_price) * 0.381966
        c_price = b_price + (a_price - b_price) * 0.786151
        d_price = c_price + (c_price - b_price) * 0.618034

        return [
            HarmonicPoint(max(0, mid_idx - 25), x_price, "L"),
            HarmonicPoint(max(0, mid_idx - 18), a_price, "H"),
            HarmonicPoint(max(0, mid_idx - 12), b_price, "L"),
            HarmonicPoint(max(0, mid_idx - 6), c_price, "H"),
            HarmonicPoint(min(len(window_data) - 1, mid_idx), d_price, "L"),
        ]

    def _get_pivot_points(
        self, data: pd.DataFrame, min_distance: int = 1
    ) -> list[HarmonicPoint]:
        """Find pivot points in data for harmonic pattern detection."""
        high_low_values = data[["high", "low"]].to_numpy(copy=False)
        cache_key = f"{hash(high_low_values.tobytes())}_{min_distance}"
        if cache_key in self._pivot_cache:
            return self._pivot_cache[cache_key]

        highs = data["high"]
        lows = data["low"]
        pivots: list[HarmonicPoint] = []

        for i in range(1, len(data) - 1):
            # Pivot high: current high higher than both neighbors
            if highs.iloc[i] >= highs.iloc[i - 1] and highs.iloc[i] >= highs.iloc[i + 1]:
                if not pivots or (i - pivots[-1].position) >= min_distance:
                    pivots.append(HarmonicPoint(i, float(highs.iloc[i]), "H"))

            # Pivot low: current low lower than both neighbors
            elif lows.iloc[i] <= lows.iloc[i - 1] and lows.iloc[i] <= lows.iloc[i + 1]:
                if not pivots or (i - pivots[-1].position) >= min_distance:
                    pivots.append(HarmonicPoint(i, float(lows.iloc[i]), "L"))

        self._set_pivot_cache(cache_key, pivots)
        return pivots

    @staticmethod
    def _validate_harmonic_ratios(
        x: float,
        a: float,
        b: float,
        c: float,
        d: float,
        ratios: dict[str, float],
        tolerance: float,
    ) -> bool:
        """Validate whether points form a harmonic pattern."""
        xa_range = abs(a - x)
        ab_range = abs(b - a)
        bc_range = abs(c - b)

        if xa_range == 0 or ab_range == 0 or bc_range == 0:
            return False

        ab_ratio = ab_range / xa_range
        bc_ratio = bc_range / ab_range
        cd_ratio = abs(d - c) / bc_range
        ad_ratio = abs(d - a) / xa_range

        validations = [
            HarmonicAnalyzer.validate_ratio(ab_ratio, ratios["AB"], tolerance),
            HarmonicAnalyzer.validate_ratio(bc_ratio, ratios["BC"], tolerance),
            # CD leg generally has wider variance in live markets.
            HarmonicAnalyzer.validate_ratio(cd_ratio, ratios["CD"], tolerance * 2),
            HarmonicAnalyzer.validate_ratio(ad_ratio, ratios["AD"], tolerance),
        ]
        return all(validations)

    @staticmethod
    def _calculate_pattern_target(
        x: float, a: float, b: float, c: float, d: float, pattern_type: str
    ) -> float:
        """Calculate price target for the harmonic pattern."""
        _ = (x, a, b)  # kept for signature compatibility and future extensions
        cd_move = d - c
        target = d + cd_move

        if pattern_type.upper() == "GARTLEY":
            target = d + cd_move * 0.618034
        elif pattern_type.upper() == "BUTTERFLY":
            target = d + cd_move * 0.786932
        elif pattern_type.upper() in {"BAT", "CRAB"}:
            target = d + cd_move * 0.381966

        return target

    @staticmethod
    def _calculate_pattern_strength(
        ratios: dict[str, float], tolerance: float, pattern_type: str = ""
    ) -> float:
        """Calculate confidence strength for harmonic pattern quality."""
        _ = ratios
        base_strength = 0.7

        if pattern_type.upper() in {"CRAB", "BAT"}:
            base_strength += 0.1

        tolerance_bonus = max(0.0, (0.1 - tolerance) * 2)
        base_strength += tolerance_bonus

        return min(0.9, base_strength)


class _HarmonicPatternBase(CandlestickPatternRecognizer):
    """Shared behavior for harmonic pattern recognizers."""

    def __init__(
        self,
        config: ConfigSection | None,
        *,
        pattern_type: str,
        signal_prefix: str,
        mtf_context_key: str,
        description: str,
        default_lookback_period: int,
        default_search_window: int,
    ) -> None:
        super().__init__(config)
        self.pattern_type = pattern_type.upper()
        self.signal_prefix = signal_prefix
        self.mtf_context_key = mtf_context_key
        self.description = description

        self.lookback_period = self._as_int(
            self.config.get("lookback_period"), default_lookback_period, minimum=5
        )
        self.tolerance = self._as_float(self.config.get("tolerance"), 0.05, minimum=1e-6)
        self.search_window = self._as_int(
            self.config.get("search_window"), default_search_window, minimum=5
        )
        self.confidence_cap = self._as_float(
            self.config.get("confidence_cap"), 0.0001, minimum=1e-8
        )
        self.pattern_completeness_scale = self._as_float(
            self.config.get("pattern_completeness_scale"), 10.0, minimum=0.1
        )

        self.harmonic_analyzer = HarmonicAnalyzer()
        self._pattern_cache: dict[str, HarmonicPatternMatch | None] = {}
        self._max_pattern_cache_size = self._as_int(
            self.config.get("max_pattern_cache_size"), 256, minimum=16
        )

    @staticmethod
    def _as_int(value: object, default: int, minimum: int = 1) -> int:
        """Best-effort integer parse with minimum guard."""
        try:
            return max(minimum, int(value))
        except (TypeError, ValueError):
            return max(minimum, default)

    @staticmethod
    def _as_float(value: object, default: float, minimum: float = 0.0) -> float:
        """Best-effort float parse with minimum guard."""
        try:
            return max(minimum, float(value))
        except (TypeError, ValueError):
            return max(minimum, default)

    def get_lookback_period(self) -> int:
        """Get lookback period required for this recognizer."""
        return self.lookback_period

    def _resolve_index(
        self,
        data: pd.DataFrame,
        index: int,
        multi_timeframe_data: MultiTimeframeData | None,
    ) -> int | None:
        """Validate recognition inputs and normalize index."""
        try:
            resolved = self.validate_recognition_inputs(
                data,
                index,
                required_length=max(self.lookback_period + 1, 6),
                multi_timeframe_data=multi_timeframe_data,
            )
        except Exception:
            return None

        if resolved < self.lookback_period:
            return None
        return resolved

    def _build_pattern_cache_key(self, data: pd.DataFrame, index: int) -> str:
        """Build cache key with lightweight data context."""
        candle = data.iloc[index]
        return (
            f"{self.pattern_type}:{index}:{len(data)}:{self.tolerance:.8f}:"
            f"{float(candle['high']):.8f}:{float(candle['low']):.8f}:{float(candle['close']):.8f}"
        )

    def _search_pattern(self, data: pd.DataFrame, index: int) -> HarmonicPatternMatch | None:
        """Search harmonic pattern within recent window."""
        cache_key = self._build_pattern_cache_key(data, index)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

        if len(data) < 5 or index < 4:
            self._pattern_cache[cache_key] = None
            return None

        pattern: HarmonicPatternMatch | None = None
        search_window = min(self.search_window, index)
        search_start = min(len(data) - 5, index - 4)
        search_end = max(0, index - search_window)

        for start_idx in range(search_start, search_end - 1, -1):
            candidate = self.harmonic_analyzer.find_harmonic_pattern(
                data, self.pattern_type, start_idx, self.tolerance
            )
            if candidate and abs(candidate["completion_position"] - index) <= 1:
                pattern = candidate
                break

        self._pattern_cache[cache_key] = pattern
        self._trim_pattern_cache()
        return pattern

    def _trim_pattern_cache(self) -> None:
        """Keep pattern cache bounded for long-running sessions."""
        while len(self._pattern_cache) > self._max_pattern_cache_size:
            oldest_key = next(iter(self._pattern_cache))
            self._pattern_cache.pop(oldest_key, None)

    def clear_runtime_state(self) -> None:
        """Release cached harmonic analysis state."""
        super().clear_runtime_state()
        self._pattern_cache.clear()
        self.harmonic_analyzer.clear_cache()

    def _calculate_pattern_completeness(
        self, current_price: float, completion_price: float
    ) -> float:
        """Compute pattern completeness score based on current vs completion price."""
        if completion_price == 0:
            return 0.0

        price_deviation = abs(current_price - completion_price) / abs(completion_price)
        return max(0.0, 1.0 - min(1.0, price_deviation * self.pattern_completeness_scale))

    def _calculate_capped_confidence(
        self,
        data: pd.DataFrame,
        index: int,
        *,
        base_confidence: float,
        pattern_completeness: float,
    ) -> float:
        """Calculate confidence with shared harmonic factors and cap."""
        pattern_factors = {
            "trend_strength": self._calculate_trend_strength(data, index, 20),
            "candle_size": self._calculate_candle_size_confidence(data, index, 0.6),
            "price_movement": self._calculate_price_movement_confidence(data, index, 0.7),
            "pattern_completeness": max(0.0, min(1.0, pattern_completeness)),
        }
        confidence = self._calculate_pattern_confidence(
            data, index, pattern_factors, base_confidence=base_confidence
        )
        return min(confidence, self.confidence_cap)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """Recognize harmonic pattern at the given index."""
        resolved_index = self._resolve_index(data, index, multi_timeframe_data)
        if resolved_index is None:
            return None

        pattern = self._search_pattern(data, resolved_index)
        if not pattern:
            return None

        current_close = float(data.iloc[resolved_index]["close"])
        pattern_completeness = self._calculate_pattern_completeness(
            current_close, pattern["completion_price"]
        )

        confidence = self._calculate_capped_confidence(
            data,
            resolved_index,
            base_confidence=pattern["strength"],
            pattern_completeness=pattern_completeness,
        )

        mtf_confidence = 1.0
        regime_adjustments: dict[str, object] = {}
        if multi_timeframe_data:
            mtf_confidence = self._analyze_multi_timeframe_alignment(
                data,
                resolved_index,
                multi_timeframe_data,
                self.mtf_context_key,
            )
            confidence *= mtf_confidence
            regime_adjustments = self._adjust_thresholds_for_regime(
                multi_timeframe_data,
                self.mtf_context_key,
            )

        confidence = min(confidence, self.confidence_cap)
        direction = pattern["direction"]
        signal_suffix = "bullish" if direction == 1 else "bearish"

        metadata: SignalMetadata = {
            "pattern": self.signal_prefix,
            "completion_price": pattern["completion_price"],
            "target_price": pattern["target_price"],
            "confidence": confidence,
            "pattern_completeness": pattern_completeness,
            "mtf_confidence": mtf_confidence,
            "regime_adjustments": json.dumps(regime_adjustments),
        }

        return SignalResult(
            signal_type=f"{self.signal_prefix}_{signal_suffix}",
            strength=confidence,
            direction=direction,
            description=self.description,
            timestamp=data.index[resolved_index],
            confidence=confidence,
            metadata=metadata,
        )


class GartleyRecognizer(_HarmonicPatternBase):
    """Recognizes Gartley harmonic patterns."""

    def __init__(self, config: ConfigSection | None = None):
        super().__init__(
            config,
            pattern_type="GARTLEY",
            signal_prefix="gartley",
            mtf_context_key="harmonic_gartley",
            description="Gartley Harmonic Pattern",
            default_lookback_period=5,
            default_search_window=30,
        )


class ButterflyRecognizer(_HarmonicPatternBase):
    """Recognizes Butterfly harmonic patterns."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="BUTTERFLY",
            signal_prefix="butterfly",
            mtf_context_key="harmonic_butterfly",
            description="Butterfly Harmonic Pattern",
            default_lookback_period=60,
            default_search_window=60,
        )


class BatRecognizer(_HarmonicPatternBase):
    """Recognizes Bat harmonic patterns."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="BAT",
            signal_prefix="bat",
            mtf_context_key="harmonic_bat",
            description="Bat Harmonic Pattern",
            default_lookback_period=60,
            default_search_window=60,
        )


class CrabRecognizer(_HarmonicPatternBase):
    """Recognizes Crab harmonic patterns."""

    def __init__(self, config: ConfigSection | None = None) -> None:
        super().__init__(
            config,
            pattern_type="CRAB",
            signal_prefix="crab",
            mtf_context_key="harmonic_crab",
            description="Crab Harmonic Pattern",
            default_lookback_period=60,
            default_search_window=60,
        )
