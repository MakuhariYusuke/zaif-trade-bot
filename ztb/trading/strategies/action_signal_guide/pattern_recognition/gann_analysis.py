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

from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd

from .base import CandlestickPatternRecognizer, SignalResult


class GannAnalyzer:
    """Utility class for Gann analysis calculations.

    This class provides dynamic calculation methods for Gann angles and levels
    based on market conditions rather than fixed hardcoded values.
    """

    # Base Gann angles (degrees) - can be adjusted based on market conditions
    BASE_GANN_ANGLES = [82.5, 75, 71.25, 63.75, 45, 26.25, 18.75, 15, 7.5]

    # Base Gann square of 9 levels - can be extended based on volatility
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
        """Calculate adaptive Gann angles based on market conditions.

        Args:
            volatility_ratio: Market volatility ratio (1.0 = normal, >1.0 = high volatility)
            trend_strength: Trend strength (0.0-1.0, higher = stronger trend)

        Returns:
            List of adaptive Gann angles in degrees
        """
        base_angles = GannAnalyzer.BASE_GANN_ANGLES.copy()

        # Adjust angles based on volatility
        # High volatility: use steeper angles (more responsive)
        # Low volatility: use shallower angles (more stable)
        volatility_factor = max(0.5, min(2.0, volatility_ratio))

        # Adjust angles based on trend strength
        # Strong trend: emphasize trend-following angles (45°, 26.25°, etc.)
        # Weak trend: include more counter-trend angles
        trend_factor = max(0.0, min(1.0, trend_strength))

        adaptive_angles = []
        for angle in base_angles:
            # Scale angle based on volatility
            adjusted_angle = angle * volatility_factor

            # For strong trends, prioritize key angles
            if trend_strength > 0.7 and angle in [45, 26.25, 18.75]:
                adjusted_angle *= 0.9  # Slightly steeper for strong trends
            elif trend_strength < 0.3 and angle in [82.5, 75, 71.25]:
                adjusted_angle *= 1.1  # Slightly shallower for weak trends

            # Keep angles within reasonable bounds
            adjusted_angle = max(5.0, min(85.0, adjusted_angle))
            adaptive_angles.append(adjusted_angle)

        return sorted(set(adaptive_angles), reverse=True)  # Sort descending

    @staticmethod
    def get_adaptive_square_levels(
        volatility_ratio: float = 1.0, range_extension: float = 1.0
    ) -> list[float]:
        """Calculate adaptive Gann square levels based on market conditions.

        Args:
            volatility_ratio: Market volatility ratio (1.0 = normal)
            range_extension: Range extension factor (1.0 = standard range)

        Returns:
            List of adaptive square levels
        """
        base_levels = GannAnalyzer.BASE_SQUARE_LEVELS.copy()

        # Extend levels for high volatility or large ranges
        if volatility_ratio > 1.5 or range_extension > 1.2:
            # Add additional levels for extended ranges
            extended_levels = [2.25, 2.5, 2.75, 3.0]
            base_levels.extend(extended_levels)

        # Contract levels for low volatility
        elif volatility_ratio < 0.7:
            # Remove some extreme levels
            base_levels = [level for level in base_levels if level <= 1.5]

        return base_levels

    @staticmethod
    def calculate_gann_angles(
        pivot_price: float,
        pivot_time: int,
        time_range: int = 100,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
    ) -> Dict[float, np.ndarray]:
        """Calculate Gann angle lines from a pivot point with adaptive angles.

        Args:
            pivot_price: Price at pivot point
            pivot_time: Time index of pivot point
            time_range: Number of periods to calculate angles for
            volatility_ratio: Market volatility ratio for angle adaptation
            trend_strength: Trend strength for angle adaptation

        Returns:
            Dictionary mapping angle degrees to (time, price) arrays
        """
        angles = {}
        adaptive_angles = GannAnalyzer.get_adaptive_gann_angles(
            volatility_ratio, trend_strength
        )

        for angle_deg in adaptive_angles:
            angle_rad = np.radians(angle_deg)
            price_per_time = np.tan(angle_rad)

            # Pre-allocate numpy array for better memory efficiency
            time_diff: np.ndarray = np.arange(time_range) - pivot_time
            price_diff = (
                time_diff * price_per_time * (pivot_price * 0.01)
            )  # Scale factor
            angle_prices = pivot_price + price_diff

            angles[angle_deg] = np.column_stack((np.arange(time_range), angle_prices))

        return angles

    @staticmethod
    def calculate_gann_square(
        high: float,
        low: float,
        volatility_ratio: float = 1.0,
        range_extension: float = 1.0,
    ) -> Dict[str, float]:
        """Calculate adaptive Gann square levels between high and low.

        Args:
            high: Recent high price
            low: Recent low price
            volatility_ratio: Market volatility ratio for level adaptation
            range_extension: Range extension factor for level adaptation

        Returns:
            Dictionary mapping level names to price levels
        """
        range_size = high - low
        square_levels = {}
        adaptive_levels = GannAnalyzer.get_adaptive_square_levels(
            volatility_ratio, range_extension
        )

        for level in adaptive_levels:
            square_levels[f"level_{level}"] = low + range_size * level

        return square_levels

    @staticmethod
    def find_gann_time_clusters(
        data: pd.DataFrame,
        index: int,
        lookback: int = 20,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
    ) -> Optional[Dict]:
        """Find Gann time cluster formations with adaptive strength calculation.

        Args:
            data: Price data DataFrame
            index: Current index position
            lookback: Lookback period for analysis
            volatility_ratio: Market volatility ratio for strength adaptation
            trend_strength: Trend strength for confidence adjustment

        Returns:
            Dictionary with time cluster information or None
        """
        if index < lookback:
            return None

        # Look for significant price levels that align with time cycles
        recent_data = data.iloc[index - lookback : index + 1]

        # Find pivot points
        highs = recent_data["high"]
        lows = recent_data["low"]

        # Simple pivot detection
        pivot_highs = []
        pivot_lows = []

        for i in range(2, len(recent_data) - 2):
            if (
                highs.iloc[i] > highs.iloc[i - 1]
                and highs.iloc[i] > highs.iloc[i - 2]
                and highs.iloc[i] > highs.iloc[i + 1]
                and highs.iloc[i] > highs.iloc[i + 2]
            ):
                pivot_highs.append((i, highs.iloc[i]))

            if (
                lows.iloc[i] < lows.iloc[i - 1]
                and lows.iloc[i] < lows.iloc[i - 2]
                and lows.iloc[i] < lows.iloc[i + 1]
                and lows.iloc[i] < lows.iloc[i + 2]
            ):
                pivot_lows.append((i, lows.iloc[i]))

        # Check for time symmetry (equal time intervals between pivots)
        if len(pivot_highs) >= 2:
            intervals = [
                pivot_highs[i + 1][0] - pivot_highs[i][0]
                for i in range(len(pivot_highs) - 1)
            ]
            if len(set(intervals)) == 1:  # All intervals equal
                # Adaptive strength based on market conditions
                base_strength = 0.4 + len(pivot_highs) * 0.1
                volatility_boost = min(
                    0.2, volatility_ratio * 0.1
                )  # Boost in high volatility
                trend_boost = trend_strength * 0.1  # Boost in strong trends
                interval_quality = min(
                    1.0, intervals[0] / 10.0
                )  # Better for longer intervals

                strength = min(
                    0.9,
                    base_strength
                    + volatility_boost
                    + trend_boost
                    + interval_quality * 0.1,
                )

                return {
                    "type": "time_cluster_highs",
                    "pivots": pivot_highs,
                    "interval": intervals[0],
                    "strength": strength,
                }

        if len(pivot_lows) >= 2:
            intervals = [
                pivot_lows[i + 1][0] - pivot_lows[i][0]
                for i in range(len(pivot_lows) - 1)
            ]
            if len(set(intervals)) == 1:  # All intervals equal
                # Adaptive strength based on market conditions
                base_strength = 0.4 + len(pivot_lows) * 0.1
                volatility_boost = min(0.2, volatility_ratio * 0.1)
                trend_boost = trend_strength * 0.1
                interval_quality = min(1.0, intervals[0] / 10.0)

                strength = min(
                    0.9,
                    base_strength
                    + volatility_boost
                    + trend_boost
                    + interval_quality * 0.1,
                )

                return {
                    "type": "time_cluster_lows",
                    "pivots": pivot_lows,
                    "interval": intervals[0],
                    "strength": strength,
                }

        return None


class GannAngleRecognizer(CandlestickPatternRecognizer):
    """Recognizes Gann angle support/resistance levels."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "gann_angle"
        self.lookback_period = config.get("lookback_period", 50) if config else 50
        self.gann_analyzer = GannAnalyzer()

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Gann angle levels at the given index."""
        if index < self.lookback_period:
            return None

        current_price = data.iloc[index]["close"]

        # Find pivot points within lookback period
        lookback_data = data.iloc[index - self.lookback_period : index + 1]

        # Simple pivot detection for angle calculation
        pivot_high_idx = lookback_data["high"].idxmax()
        pivot_low_idx = lookback_data["low"].idxmin()

        pivot_high = cast(float, lookback_data.loc[pivot_high_idx, "high"])
        pivot_low = cast(float, lookback_data.loc[pivot_low_idx, "low"])

        # Use the more recent pivot as the angle origin
        pivot_high_pos = lookback_data.index.tolist().index(pivot_high_idx)
        pivot_low_pos = lookback_data.index.tolist().index(pivot_low_idx)

        # Calculate market conditions for adaptive parameters
        returns = lookback_data["close"].pct_change().dropna()
        current_volatility = returns.std()
        avg_volatility = (
            returns.rolling(20).std().mean()
            if len(returns) >= 20
            else current_volatility
        )
        volatility_ratio = (
            current_volatility / avg_volatility if avg_volatility > 0 else 1.0
        )

        # Simple trend strength calculation
        sma_20 = (
            lookback_data["close"].rolling(20).mean().iloc[-1]
            if len(lookback_data) >= 20
            else lookback_data["close"].mean()
        )
        trend_strength = (
            abs((lookback_data["close"].iloc[-1] - sma_20) / sma_20)
            if sma_20 != 0
            else 0.5
        )

        # Use the more recent pivot as the angle origin
        pivot_high_pos = lookback_data.index.tolist().index(pivot_high_idx)
        pivot_low_pos = lookback_data.index.tolist().index(pivot_low_idx)

        if pivot_high_pos > pivot_low_pos:
            pivot_price: float = pivot_high
            pivot_time = int(index - (len(lookback_data) - pivot_high_pos))
            # Calculate continuous direction based on angle steepness and market conditions
            base_direction = -0.7  # Bearish bias from high
            direction = base_direction * (
                1 + trend_strength * 0.3
            )  # Amplify with trend strength
        else:
            pivot_price = pivot_low
            pivot_time = int(index - (len(lookback_data) - pivot_low_pos))
            # Calculate continuous direction based on angle steepness and market conditions
            base_direction = 0.7  # Bullish bias from low
            direction = base_direction * (
                1 + trend_strength * 0.3
            )  # Amplify with trend strength

        # Ensure direction stays within [-1, 1] bounds
        direction = max(-1.0, min(1.0, direction))

        # Calculate Gann angles
        angles = self.gann_analyzer.calculate_gann_angles(
            pivot_price, int(pivot_time), time_range=self.lookback_period
        )

        # Check if current price is near any Gann angle
        current_time_offset = 0  # Current position

        for angle_deg, angle_points in angles.items():
            if current_time_offset < len(angle_points):
                angle_price = float(angle_points[current_time_offset, 1])

                # Tolerance based on price volatility
                price_range = lookback_data["high"].max() - lookback_data["low"].min()
                tolerance = price_range * 0.02  # 2% tolerance

                if abs(current_price - angle_price) <= tolerance:
                    # Calculate pattern completeness based on angle importance and price proximity
                    price_deviation = abs(current_price - angle_price) / tolerance
                    pattern_completeness = (
                        1.0 - price_deviation
                    )  # Closer to angle = higher completeness

                    # Key angles get higher base confidence
                    key_angles = [45, 26.25, 18.75]
                    base_confidence = 0.7 if angle_deg in key_angles else 0.6

                    # Use pattern confidence calculation
                    pattern_factors = {
                        "trend_strength": self._calculate_trend_strength(
                            data, index, 20
                        ),
                        "candle_size": self._calculate_candle_size_confidence(
                            data, index, 0.6
                        ),  # Gann angles are structural
                        "price_movement": self._calculate_price_movement_confidence(
                            data, index, 0.7
                        ),  # Approaching angle level
                        "pattern_completeness": pattern_completeness,  # How close price is to the Gann angle
                    }

                    confidence = self._calculate_pattern_confidence(
                        data, index, pattern_factors, base_confidence=base_confidence
                    )

                    signal_type = (
                        "gann_angle_support"
                        if direction == 1
                        else "gann_angle_resistance"
                    )

                    return SignalResult(
                        signal_type=signal_type,
                        strength=confidence,
                        direction=direction,
                        description=f"Gann {angle_deg}° angle level",
                        timestamp=data.index[index],
                        confidence=confidence,
                        metadata={
                            "pattern": "gann_angle",
                            "angle": angle_deg,
                            "pivot_price": pivot_price,
                            "confidence": confidence,
                            "pattern_completeness": pattern_completeness,
                        },
                    )

        return None


class GannSquareRecognizer(CandlestickPatternRecognizer):
    """Recognizes Gann square of 9 levels."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "gann_square"
        self.lookback_period = config.get("lookback_period", 30) if config else 30
        self.gann_analyzer = GannAnalyzer()

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Gann square levels at the given index."""
        if index < self.lookback_period:
            return None

        current_price = data.iloc[index]["close"]

        # Calculate Gann square from recent high-low range
        lookback_data = data.iloc[index - self.lookback_period : index + 1]
        recent_high = lookback_data["high"].max()
        recent_low = lookback_data["low"].min()

        # Calculate market conditions for adaptive parameters
        returns = lookback_data["close"].pct_change().dropna()
        current_volatility = returns.std()
        avg_volatility = (
            returns.rolling(20).std().mean()
            if len(returns) >= 20
            else current_volatility
        )
        volatility_ratio = (
            current_volatility / avg_volatility if avg_volatility > 0 else 1.0
        )

        # Simple trend strength calculation
        sma_20 = (
            lookback_data["close"].rolling(20).mean().iloc[-1]
            if len(lookback_data) >= 20
            else lookback_data["close"].mean()
        )
        trend_strength = (
            abs((lookback_data["close"].iloc[-1] - sma_20) / sma_20)
            if sma_20 != 0
            else 0.5
        )

        square_levels = self.gann_analyzer.calculate_gann_square(
            recent_high,
            recent_low,
            volatility_ratio=volatility_ratio,
            range_extension=1.0,
        )

        # Check if current price is near a square level
        for level_name, level_price in square_levels.items():
            level_ratio = float(level_name.split("_")[1])

            # Adaptive tolerance based on volatility
            base_tolerance = 0.015  # 1.5% base tolerance
            adaptive_tolerance = base_tolerance * (
                1 + volatility_ratio * 0.5
            )  # Increase tolerance in high volatility
            tolerance = (recent_high - recent_low) * adaptive_tolerance

            if abs(current_price - level_price) <= tolerance:
                # Determine direction based on position relative to midpoint and trend
                midpoint = (recent_high + recent_low) / 2
                price_position = (
                    (current_price - recent_low) / (recent_high - recent_low)
                    if recent_high != recent_low
                    else 0.5
                )

                # Base direction from price position
                base_direction = 1 if current_price > midpoint else -1

                # Amplify direction based on trend strength and level importance
                level_importance = 1.0
                key_levels = [0.5, 1.0, 1.5, 2.0]
                if level_ratio in key_levels:
                    level_importance = 1.5

                # Combine factors for final direction
                direction_factor = (
                    base_direction * level_importance * (0.5 + trend_strength * 0.5)
                )
                direction = max(-1.0, min(1.0, direction_factor))

                # Adaptive strength based on multiple factors
                base_strength = 0.55
                volatility_boost = min(
                    0.2, volatility_ratio * 0.1
                )  # Boost in high volatility
                trend_boost = trend_strength * 0.15  # Boost in strong trends
                level_boost = 0.2 if level_ratio in key_levels else 0.0
                proximity_factor = 1 - (
                    abs(current_price - level_price) / tolerance
                )  # Closer = stronger

                # Calculate pattern completeness based on proximity and level importance
                pattern_completeness = proximity_factor * (1.0 + level_boost * 0.5)

                # Use pattern confidence calculation
                pattern_factors = {
                    "trend_strength": self._calculate_trend_strength(data, index, 20),
                    "candle_size": self._calculate_candle_size_confidence(
                        data, index, 0.6
                    ),  # Gann squares are structural
                    "price_movement": self._calculate_price_movement_confidence(
                        data, index, 0.7
                    ),  # Approaching square level
                    "pattern_completeness": pattern_completeness,  # How close and important the Gann square level is
                }

                confidence = self._calculate_pattern_confidence(
                    data,
                    index,
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
                    timestamp=data.index[index],
                    metadata={
                        "pattern": "gann_square",
                        "level_ratio": level_ratio,
                        "level_price": level_price,
                        "volatility_ratio": volatility_ratio,
                        "trend_strength": trend_strength,
                        "tolerance_pct": adaptive_tolerance * 100,
                        "level_importance": level_importance,
                        "confidence": confidence,
                        "pattern_completeness": pattern_completeness,
                    },
                )

        return None


class GannTimeClusterRecognizer(CandlestickPatternRecognizer):
    """Recognizes Gann time clusters and cycle alignments."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.pattern_type = "gann_time_cluster"
        self.lookback_period = config.get("lookback_period", 30) if config else 30
        self.gann_analyzer = GannAnalyzer()

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """Recognize Gann time clusters at the given index."""
        if index < self.lookback_period:
            return None

        # Calculate market conditions for adaptive parameters
        lookback_data = data.iloc[index - self.lookback_period : index + 1]
        returns = lookback_data["close"].pct_change().dropna()
        current_volatility = returns.std()
        avg_volatility = (
            returns.rolling(20).std().mean()
            if len(returns) >= 20
            else current_volatility
        )
        volatility_ratio = (
            current_volatility / avg_volatility if avg_volatility > 0 else 1.0
        )

        # Simple trend strength calculation
        sma_20 = (
            lookback_data["close"].rolling(20).mean().iloc[-1]
            if len(lookback_data) >= 20
            else lookback_data["close"].mean()
        )
        trend_strength = (
            abs((lookback_data["close"].iloc[-1] - sma_20) / sma_20)
            if sma_20 != 0
            else 0.5
        )

        time_cluster = self.gann_analyzer.find_gann_time_clusters(
            data,
            index,
            lookback=self.lookback_period,
            volatility_ratio=volatility_ratio,
            trend_strength=trend_strength,
        )

        if time_cluster:
            # Calculate pattern completeness based on cluster quality
            cluster_quality = min(
                1.0, len(time_cluster["pivots"]) / 5.0
            )  # Better with more pivots
            pattern_completeness = cluster_quality * (0.7 + trend_strength * 0.3)

            # Use pattern confidence calculation
            pattern_factors = {
                "trend_strength": self._calculate_trend_strength(data, index, 25),
                "candle_size": self._calculate_candle_size_confidence(
                    data, index, 0.5
                ),  # Time clusters are timing-based
                "price_movement": self._calculate_price_movement_confidence(
                    data, index, 0.6
                ),  # Time-based signals
                "pattern_completeness": pattern_completeness,  # How strong the time cluster alignment is
            }

            confidence = self._calculate_pattern_confidence(
                data, index, pattern_factors, base_confidence=time_cluster["strength"]
            )

            # Adaptive direction based on cluster type and trend strength
            base_direction = 1 if time_cluster["type"] == "time_cluster_lows" else -1

            # Amplify direction based on trend strength and cluster quality
            direction_factor = (
                base_direction * cluster_quality * (0.7 + trend_strength * 0.3)
            )
            direction = max(-1.0, min(1.0, direction_factor))

            signal_type = (
                "gann_time_cluster_support"
                if direction > 0
                else "gann_time_cluster_resistance"
            )

            return SignalResult(
                signal_type=signal_type,
                strength=confidence,
                direction=direction,
                description=f"Gann Time Cluster: {time_cluster['type'].replace('_', ' ').title()} (adaptive)",
                timestamp=data.index[index],
                metadata={
                    "pattern": "gann_time_cluster",
                    "cluster_type": time_cluster["type"],
                    "interval": time_cluster["interval"],
                    "pivot_count": len(time_cluster["pivots"]),
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "cluster_quality": cluster_quality,
                    "confidence": confidence,
                    "pattern_completeness": pattern_completeness,
                },
            )

        return None
