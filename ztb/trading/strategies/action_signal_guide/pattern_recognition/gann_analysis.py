"""
Gann Analysis Module

This module provides pattern recognition and analysis based on W.D. Gann's methods,
including Gann squares, angles, fans, and time-price relationships.

Dynamic Calculation Rationale:
---------------------------

This implementation replaces hardcoded temporary values with market-adaptive calculations
to provide more responsive and context-aware Gann analysis:

1. Volatility Adaptation:
   - High volatility markets (>1.5x normal): Use steeper angles and extended square levels
   - Low volatility markets (<0.7x normal): Use shallower angles and contracted levels
   - Adaptive tolerance: Increases in high volatility to account for larger price swings

2. Trend Strength Integration:
   - Strong trends (>0.7): Prioritize trend-following angles, amplify directional signals
   - Weak trends (<0.3): Include more counter-trend angles for potential reversals
   - Direction amplification: Combines price position, level importance, and trend strength

3. Continuous Direction Values:
   - Replaced discrete BUY/SELL with [-1, 1] continuous values
   - Allows for nuanced position sizing and ML integration
   - Factors: price position, level importance, market conditions, proximity to levels

4. Adaptive Signal Strength:
   - Base strength + volatility boost + trend boost + level importance + proximity factor
   - Maximum strength capped at 1.0, minimum at 0.0
   - Real-time confidence scoring based on market context

Key Benefits:
- Market-responsive instead of static analysis
- Improved signal quality through context awareness
- Better integration with modern trading systems
- Reduced false signals in varying market conditions

Implementation Details:
----------------------

- GannAnalyzer: Core utility class with adaptive calculation methods
- PatternRecognizer subclasses: Individual recognizers with market-aware logic
- Multi-timeframe support: Interface ready for enhanced validation
- Comprehensive metadata: Full transparency of calculation factors
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd

try:
    from ztb.features.generators.technical.trend.supertrend import (
        compute_supertrend_direction,
    )
except ImportError:

    def compute_supertrend_direction(df: pd.DataFrame) -> pd.Series:
        return pd.Series([0] * len(df), index=df.index)


try:
    from ztb.features.generators.technical.volatility.bollinger import compute_bb_width
except ImportError:

    def compute_bb_width(df: pd.DataFrame, period: int = 20) -> pd.Series:
        return pd.Series([0.0] * len(df), index=df.index)


try:
    from ztb.features.generators.technical.volume.obv import compute_obv
except ImportError:

    def compute_obv(df: pd.DataFrame) -> pd.Series:
        return pd.Series([0.0] * len(df), index=df.index)


try:
    from ztb.features.generators.technical.trend.sma import compute_sma
except ImportError:

    def compute_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        if "close" not in df.columns:
            return pd.Series(dtype="float64", index=df.index)
        return df["close"].rolling(window=max(1, int(period))).mean()


from .base import CandlestickPatternRecognizer, SignalResult


class TimeClusterInfo(TypedDict):
    type: str
    pivots: list[tuple[int, float]]
    interval: int
    strength: float


@dataclass(frozen=True)
class GannMarketContext:
    volatility_ratio: float
    trend_strength: float


class GannAnalyzer:
    """Utility class for Gann analysis calculations."""

    BASE_GANN_ANGLES = [82.5, 75, 71.25, 63.75, 45, 26.25, 18.75, 15, 7.5]
    BASE_SQUARE_LEVELS = [
        0.125,
        0.25,
        0.375,
        0.5,
        0.625,
        0.75,
        0.875,
        1.0,
        1.125,
        1.25,
        1.375,
        1.5,
        1.625,
        1.75,
        1.875,
        2.0,
    ]

    @staticmethod
    def get_adaptive_gann_angles(
        volatility_ratio: float = 1.0, trend_strength: float = 0.5
    ) -> list[float]:
        """Calculate adaptive Gann angles based on market conditions."""
        base_angles = GannAnalyzer.BASE_GANN_ANGLES.copy()
        volatility_factor = max(0.5, min(2.0, volatility_ratio))

        adaptive_angles: list[float] = []
        for angle in base_angles:
            adjusted_angle = angle * volatility_factor

            if trend_strength > 0.7 and angle in [45, 26.25, 18.75]:
                adjusted_angle *= 0.9
            elif trend_strength < 0.3 and angle in [82.5, 75, 71.25]:
                adjusted_angle *= 1.1

            adjusted_angle = max(5.0, min(85.0, adjusted_angle))
            adaptive_angles.append(float(adjusted_angle))

        return sorted(set(adaptive_angles), reverse=True)

    @staticmethod
    def get_adaptive_square_levels(
        volatility_ratio: float = 1.0, range_extension: float = 1.0
    ) -> list[float]:
        """Calculate adaptive Gann square levels based on market conditions."""
        levels = GannAnalyzer.BASE_SQUARE_LEVELS.copy()

        if volatility_ratio > 1.5 or range_extension > 1.2:
            levels.extend([2.25, 2.5, 2.75, 3.0])
        elif volatility_ratio < 0.7:
            levels = [level for level in levels if level <= 1.5]

        return levels

    @staticmethod
    def calculate_gann_angle_prices_at_time(
        pivot_price: float,
        pivot_time: int,
        current_time: int,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
    ) -> dict[float, float]:
        """Calculate each adaptive Gann angle price at one target time."""
        angle_prices: dict[float, float] = {}
        time_diff = current_time - pivot_time
        if not np.isfinite(time_diff):
            return angle_prices

        for angle_deg in GannAnalyzer.get_adaptive_gann_angles(
            volatility_ratio, trend_strength
        ):
            angle_rad = np.radians(angle_deg)
            price_per_time = np.tan(angle_rad)
            price_diff = time_diff * price_per_time * (pivot_price * 0.01)
            angle_prices[angle_deg] = float(pivot_price + price_diff)

        return angle_prices

    @staticmethod
    def calculate_gann_angles(
        pivot_price: float,
        pivot_time: int,
        time_range: int = 100,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
    ) -> dict[float, np.ndarray]:
        """Legacy API: calculate full Gann angle lines across a time range."""
        angles: dict[float, np.ndarray] = {}

        for angle_deg in GannAnalyzer.get_adaptive_gann_angles(
            volatility_ratio, trend_strength
        ):
            angle_rad = np.radians(angle_deg)
            price_per_time = np.tan(angle_rad)

            time_axis: np.ndarray = np.arange(time_range, dtype=np.float64)
            time_diff = time_axis - float(pivot_time)
            price_diff = time_diff * price_per_time * (pivot_price * 0.01)
            angle_prices = pivot_price + price_diff

            angles[angle_deg] = np.column_stack((time_axis, angle_prices))

        return angles

    @staticmethod
    def calculate_gann_square_levels(
        high: float,
        low: float,
        volatility_ratio: float = 1.0,
        range_extension: float = 1.0,
    ) -> dict[float, float]:
        """Calculate adaptive Gann square levels keyed by level ratio."""
        range_size = high - low
        if range_size <= 0:
            return {}

        level_prices: dict[float, float] = {}
        for level in GannAnalyzer.get_adaptive_square_levels(
            volatility_ratio, range_extension
        ):
            level_prices[level] = float(low + range_size * level)

        return level_prices

    @staticmethod
    def calculate_gann_square(
        high: float,
        low: float,
        volatility_ratio: float = 1.0,
        range_extension: float = 1.0,
    ) -> dict[str, float]:
        """Legacy API: calculate adaptive Gann square levels with string keys."""
        level_prices = GannAnalyzer.calculate_gann_square_levels(
            high, low, volatility_ratio, range_extension
        )
        return {f"level_{level}": price for level, price in level_prices.items()}

    @staticmethod
    def find_gann_time_clusters(
        data: pd.DataFrame,
        index: int,
        lookback: int = 20,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
    ) -> TimeClusterInfo | None:
        """Find Gann time cluster formations with adaptive strength calculation."""
        if index < lookback or "high" not in data.columns or "low" not in data.columns:
            return None

        recent_data = data.iloc[index - lookback : index + 1]
        highs = recent_data["high"]
        lows = recent_data["low"]

        pivot_highs: list[tuple[int, float]] = []
        pivot_lows: list[tuple[int, float]] = []

        for i in range(2, len(recent_data) - 2):
            if (
                highs.iloc[i] > highs.iloc[i - 1]
                and highs.iloc[i] > highs.iloc[i - 2]
                and highs.iloc[i] > highs.iloc[i + 1]
                and highs.iloc[i] > highs.iloc[i + 2]
            ):
                pivot_highs.append((i, float(highs.iloc[i])))

            if (
                lows.iloc[i] < lows.iloc[i - 1]
                and lows.iloc[i] < lows.iloc[i - 2]
                and lows.iloc[i] < lows.iloc[i + 1]
                and lows.iloc[i] < lows.iloc[i + 2]
            ):
                pivot_lows.append((i, float(lows.iloc[i])))

        high_cluster = GannAnalyzer._build_time_cluster(
            pivot_highs,
            cluster_type="time_cluster_highs",
            volatility_ratio=volatility_ratio,
            trend_strength=trend_strength,
        )
        if high_cluster is not None:
            return high_cluster

        low_cluster = GannAnalyzer._build_time_cluster(
            pivot_lows,
            cluster_type="time_cluster_lows",
            volatility_ratio=volatility_ratio,
            trend_strength=trend_strength,
        )
        if low_cluster is not None:
            return low_cluster

        return None

    @staticmethod
    def _build_time_cluster(
        pivots: list[tuple[int, float]],
        cluster_type: str,
        volatility_ratio: float,
        trend_strength: float,
    ) -> TimeClusterInfo | None:
        if len(pivots) < 2:
            return None

        intervals = [pivots[i + 1][0] - pivots[i][0] for i in range(len(pivots) - 1)]
        if not intervals:
            return None

        first = intervals[0]
        if first <= 0 or not all(interval == first for interval in intervals):
            return None

        base_strength = 0.4 + len(pivots) * 0.1
        volatility_boost = min(0.2, volatility_ratio * 0.1)
        trend_boost = trend_strength * 0.1
        interval_quality = min(1.0, first / 10.0)

        strength = min(
            0.9,
            base_strength + volatility_boost + trend_boost + interval_quality * 0.1,
        )

        return {
            "type": cluster_type,
            "pivots": pivots,
            "interval": first,
            "strength": float(strength),
        }


class GannPatternBase(CandlestickPatternRecognizer):
    """Common base for Gann recognizers (shared context + typing helpers)."""

    def __init__(
        self, config: Mapping[str, object] | None = None, default_lookback: int = 30
    ) -> None:
        cfg: dict[str, object] = dict(config) if config else {}
        super().__init__(cfg)
        self.gann_analyzer = GannAnalyzer()
        self.lookback_period = self._to_int(cfg.get("lookback_period"), default_lookback)

    def _resolve_index(self, data: pd.DataFrame, index: int) -> int | None:
        if data.empty:
            return None
        resolved = len(data) - 1 if index < 0 else index
        if resolved < self.lookback_period or resolved >= len(data):
            return None
        return resolved

    def _get_lookback_data(self, data: pd.DataFrame, index: int) -> pd.DataFrame:
        return data.iloc[index - self.lookback_period : index + 1]

    def _calculate_market_context(self, lookback_data: pd.DataFrame) -> GannMarketContext:
        if lookback_data.empty or "close" not in lookback_data.columns:
            return GannMarketContext(volatility_ratio=1.0, trend_strength=0.5)

        returns = lookback_data["close"].pct_change().dropna()
        if returns.empty:
            return GannMarketContext(volatility_ratio=1.0, trend_strength=0.5)

        current_volatility = float(returns.std())
        rolling_vol = returns.rolling(window=min(20, len(returns))).std().dropna()
        avg_volatility = (
            float(rolling_vol.mean()) if not rolling_vol.empty else current_volatility
        )
        if avg_volatility <= 0:
            volatility_ratio = 1.0
        else:
            volatility_ratio = current_volatility / avg_volatility

        sma_series = compute_sma(lookback_data, period=min(20, len(lookback_data)))
        sma_20 = (
            float(sma_series.iloc[-1])
            if not sma_series.empty and not pd.isna(sma_series.iloc[-1])
            else 0.0
        )

        last_close = float(lookback_data["close"].iloc[-1])
        if sma_20 == 0.0:
            trend_strength = 0.5
        else:
            trend_strength = abs((last_close - sma_20) / sma_20)

        return GannMarketContext(
            volatility_ratio=float(max(0.1, min(5.0, volatility_ratio))),
            trend_strength=float(max(0.0, min(2.0, trend_strength))),
        )

    def _build_pattern_factors(
        self,
        data: pd.DataFrame,
        index: int,
        pattern_completeness: float,
        trend_lookback: int,
        candle_expected: float,
        price_expected: float,
    ) -> dict[str, float]:
        return {
            "trend_strength": self._calculate_trend_strength(data, index, trend_lookback),
            "candle_size": self._calculate_candle_size_confidence(
                data, index, candle_expected
            ),
            "price_movement": self._calculate_price_movement_confidence(
                data, index, price_expected
            ),
            "pattern_completeness": pattern_completeness,
        }

    @staticmethod
    def _to_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return float(max(min_value, min(max_value, value)))


class GannAngleRecognizer(GannPatternBase):
    """Recognizes Gann angle support/resistance levels."""

    KEY_ANGLES = {45.0, 26.25, 18.75}

    def __init__(self, config: Mapping[str, object] | None = None) -> None:
        super().__init__(config=config, default_lookback=50)
        self.pattern_type = "gann_angle"

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: dict[str, object] | None = None,
    ) -> SignalResult | None:
        """Recognize Gann angle levels at the given index."""
        del multi_timeframe_data

        resolved_index = self._resolve_index(data, index)
        if resolved_index is None:
            return None

        lookback_data = self._get_lookback_data(data, resolved_index)
        if not {"high", "low", "close"}.issubset(lookback_data.columns):
            return None

        high_values = lookback_data["high"].to_numpy(dtype="float64")
        low_values = lookback_data["low"].to_numpy(dtype="float64")
        if high_values.size == 0 or low_values.size == 0:
            return None

        pivot_high_pos = int(np.argmax(high_values))
        pivot_low_pos = int(np.argmin(low_values))
        pivot_high = float(high_values[pivot_high_pos])
        pivot_low = float(low_values[pivot_low_pos])

        context = self._calculate_market_context(lookback_data)
        current_price = float(data.iloc[resolved_index]["close"])

        if pivot_high_pos > pivot_low_pos:
            pivot_price = pivot_high
            pivot_pos = pivot_high_pos
            base_direction = -0.7
        else:
            pivot_price = pivot_low
            pivot_pos = pivot_low_pos
            base_direction = 0.7

        bars_ago = (len(lookback_data) - 1) - pivot_pos
        pivot_time = resolved_index - bars_ago
        direction = self._clamp(
            base_direction * (1.0 + context.trend_strength * 0.3), -1.0, 1.0
        )

        angle_prices = self.gann_analyzer.calculate_gann_angle_prices_at_time(
            pivot_price=pivot_price,
            pivot_time=pivot_time,
            current_time=resolved_index,
            volatility_ratio=context.volatility_ratio,
            trend_strength=context.trend_strength,
        )

        price_range = float(np.max(high_values) - np.min(low_values))
        tolerance = price_range * 0.02
        if tolerance <= 0:
            return None

        for angle_deg, angle_price in angle_prices.items():
            deviation = abs(current_price - angle_price)
            if deviation > tolerance:
                continue

            pattern_completeness = self._clamp(1.0 - (deviation / tolerance), 0.0, 1.0)
            base_confidence = 0.7 if angle_deg in self.KEY_ANGLES else 0.6

            pattern_factors = self._build_pattern_factors(
                data,
                resolved_index,
                pattern_completeness=pattern_completeness,
                trend_lookback=20,
                candle_expected=0.6,
                price_expected=0.7,
            )
            confidence = self._calculate_pattern_confidence(
                data,
                resolved_index,
                pattern_factors,
                base_confidence=base_confidence,
            )

            signal_type = (
                "gann_angle_support" if direction > 0 else "gann_angle_resistance"
            )
            return SignalResult(
                signal_type=signal_type,
                strength=confidence,
                direction=direction,
                description=f"Gann {angle_deg}° angle level",
                timestamp=data.index[resolved_index],
                confidence=confidence,
                metadata={
                    "pattern": "gann_angle",
                    "angle": angle_deg,
                    "pivot_price": pivot_price,
                    "confidence": confidence,
                    "pattern_completeness": pattern_completeness,
                    "volatility_ratio": context.volatility_ratio,
                    "trend_strength": context.trend_strength,
                },
            )

        return None


class GannSquareRecognizer(GannPatternBase):
    """Recognizes Gann square of 9 levels."""

    KEY_LEVELS = {0.5, 1.0, 1.5, 2.0}

    def __init__(self, config: Mapping[str, object] | None = None) -> None:
        super().__init__(config=config, default_lookback=30)
        self.pattern_type = "gann_square"

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: dict[str, object] | None = None,
    ) -> SignalResult | None:
        """Recognize Gann square levels at the given index."""
        del multi_timeframe_data

        resolved_index = self._resolve_index(data, index)
        if resolved_index is None:
            return None

        lookback_data = self._get_lookback_data(data, resolved_index)
        if not {"high", "low", "close"}.issubset(lookback_data.columns):
            return None

        high_values = lookback_data["high"].to_numpy(dtype="float64")
        low_values = lookback_data["low"].to_numpy(dtype="float64")
        if high_values.size == 0 or low_values.size == 0:
            return None

        recent_high = float(np.max(high_values))
        recent_low = float(np.min(low_values))
        price_range = recent_high - recent_low
        if price_range <= 0:
            return None

        context = self._calculate_market_context(lookback_data)
        current_price = float(data.iloc[resolved_index]["close"])

        square_levels = self.gann_analyzer.calculate_gann_square_levels(
            recent_high,
            recent_low,
            volatility_ratio=context.volatility_ratio,
            range_extension=1.0,
        )
        if not square_levels:
            return None

        base_tolerance = 0.015
        adaptive_tolerance = base_tolerance * (1.0 + context.volatility_ratio * 0.5)
        tolerance = price_range * adaptive_tolerance
        if tolerance <= 0:
            return None

        midpoint = (recent_high + recent_low) / 2.0

        for level_ratio, level_price in square_levels.items():
            deviation = abs(current_price - level_price)
            if deviation > tolerance:
                continue

            level_importance = 1.5 if level_ratio in self.KEY_LEVELS else 1.0
            base_direction = 1.0 if current_price > midpoint else -1.0
            direction = self._clamp(
                base_direction * level_importance * (0.5 + context.trend_strength * 0.5),
                -1.0,
                1.0,
            )

            base_strength = 0.55
            volatility_boost = min(0.2, context.volatility_ratio * 0.1)
            trend_boost = context.trend_strength * 0.15
            level_boost = 0.2 if level_ratio in self.KEY_LEVELS else 0.0

            proximity_factor = self._clamp(1.0 - (deviation / tolerance), 0.0, 1.0)
            pattern_completeness = proximity_factor * (1.0 + level_boost * 0.5)

            pattern_factors = self._build_pattern_factors(
                data,
                resolved_index,
                pattern_completeness=pattern_completeness,
                trend_lookback=20,
                candle_expected=0.6,
                price_expected=0.7,
            )
            confidence = self._calculate_pattern_confidence(
                data,
                resolved_index,
                pattern_factors,
                base_confidence=base_strength
                + volatility_boost
                + trend_boost
                + level_boost,
            )

            signal_type = (
                "gann_square_support" if direction > 0 else "gann_square_resistance"
            )
            return SignalResult(
                signal_type=signal_type,
                strength=confidence,
                direction=direction,
                description=f"Gann Square level {level_ratio} (adaptive)",
                timestamp=data.index[resolved_index],
                metadata={
                    "pattern": "gann_square",
                    "level_ratio": level_ratio,
                    "level_price": level_price,
                    "volatility_ratio": context.volatility_ratio,
                    "trend_strength": context.trend_strength,
                    "tolerance_pct": adaptive_tolerance * 100,
                    "level_importance": level_importance,
                    "confidence": confidence,
                    "pattern_completeness": pattern_completeness,
                },
            )

        return None


class GannTimeClusterRecognizer(GannPatternBase):
    """Recognizes Gann time clusters and cycle alignments."""

    def __init__(self, config: Mapping[str, object] | None = None) -> None:
        super().__init__(config=config, default_lookback=30)
        self.pattern_type = "gann_time_cluster"

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: dict[str, object] | None = None,
    ) -> SignalResult | None:
        """Recognize Gann time clusters at the given index."""
        del multi_timeframe_data

        resolved_index = self._resolve_index(data, index)
        if resolved_index is None:
            return None

        lookback_data = self._get_lookback_data(data, resolved_index)
        context = self._calculate_market_context(lookback_data)

        time_cluster = self.gann_analyzer.find_gann_time_clusters(
            data,
            resolved_index,
            lookback=self.lookback_period,
            volatility_ratio=context.volatility_ratio,
            trend_strength=context.trend_strength,
        )
        if time_cluster is None:
            return None

        cluster_quality = min(1.0, len(time_cluster["pivots"]) / 5.0)
        pattern_completeness = cluster_quality * (0.7 + context.trend_strength * 0.3)

        pattern_factors = self._build_pattern_factors(
            data,
            resolved_index,
            pattern_completeness=pattern_completeness,
            trend_lookback=25,
            candle_expected=0.5,
            price_expected=0.6,
        )
        confidence = self._calculate_pattern_confidence(
            data,
            resolved_index,
            pattern_factors,
            base_confidence=float(time_cluster["strength"]),
        )

        base_direction = 1.0 if time_cluster["type"] == "time_cluster_lows" else -1.0
        direction = self._clamp(
            base_direction * cluster_quality * (0.7 + context.trend_strength * 0.3),
            -1.0,
            1.0,
        )

        signal_type = (
            "gann_time_cluster_support"
            if direction > 0
            else "gann_time_cluster_resistance"
        )

        return SignalResult(
            signal_type=signal_type,
            strength=confidence,
            direction=direction,
            description=(
                "Gann Time Cluster: "
                f"{time_cluster['type'].replace('_', ' ').title()} (adaptive)"
            ),
            timestamp=data.index[resolved_index],
            metadata={
                "pattern": "gann_time_cluster",
                "cluster_type": time_cluster["type"],
                "interval": time_cluster["interval"],
                "pivot_count": len(time_cluster["pivots"]),
                "volatility_ratio": context.volatility_ratio,
                "trend_strength": context.trend_strength,
                "cluster_quality": cluster_quality,
                "confidence": confidence,
                "pattern_completeness": pattern_completeness,
            },
        )
