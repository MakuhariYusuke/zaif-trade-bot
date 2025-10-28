"""
Gann Analysis Module

This module provides pattern recognition and analysis based on W.D. Gann's methods,
including Gann squares, angles, fans, and time-price relationships.
"""

from typing import Dict, Optional, cast, Any

import numpy as np
import pandas as pd

from ztb.trading.constants import ACTION_BUY, ACTION_SELL

from .base import PatternRecognizer, SignalResult


class GannAnalyzer:
    """Utility class for Gann analysis calculations."""

    # Gann angles (degrees)
    GANN_ANGLES = [82.5, 75, 71.25, 63.75, 45, 26.25, 18.75, 15, 7.5]

    # Gann square of 9 levels
    SQUARE_OF_9_LEVELS = [
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
    def calculate_gann_angles(
        pivot_price: float, pivot_time: int, time_range: int = 100
    ) -> Dict[float, np.ndarray]:
        """Calculate Gann angle lines from a pivot point."""
        angles = {}

        for angle_deg in GannAnalyzer.GANN_ANGLES:
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
    def calculate_gann_square(high: float, low: float) -> Dict[str, float]:
        """Calculate Gann square levels between high and low."""
        range_size = high - low
        square_levels = {}

        for level in GannAnalyzer.SQUARE_OF_9_LEVELS:
            square_levels[f"level_{level}"] = low + range_size * level

        return square_levels

    @staticmethod
    def find_gann_time_clusters(
        data: pd.DataFrame, index: int, lookback: int = 20
    ) -> Optional[Dict]:
        """Find Gann time cluster formations."""
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
                return {
                    "type": "time_cluster_highs",
                    "pivots": pivot_highs,
                    "interval": intervals[0],
                    "strength": min(0.8, 0.4 + len(pivot_highs) * 0.1),
                }

        if len(pivot_lows) >= 2:
            intervals = [
                pivot_lows[i + 1][0] - pivot_lows[i][0]
                for i in range(len(pivot_lows) - 1)
            ]
            if len(set(intervals)) == 1:  # All intervals equal
                return {
                    "type": "time_cluster_lows",
                    "pivots": pivot_lows,
                    "interval": intervals[0],
                    "strength": min(0.8, 0.4 + len(pivot_lows) * 0.1),
                }

        return None


class GannAngleRecognizer(PatternRecognizer):
    """Recognizes Gann angle support/resistance levels."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.lookback_period = config.get('lookback_period', 50) if config else 50
        self.gann_analyzer = GannAnalyzer()

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
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

        if pivot_high_pos > pivot_low_pos:
            pivot_price: float = pivot_high
            pivot_time = int(index - (len(lookback_data) - pivot_high_pos))
            direction = ACTION_SELL  # Bearish angle from high
        else:
            pivot_price = pivot_low
            pivot_time = int(index - (len(lookback_data) - pivot_low_pos))
            direction = ACTION_BUY  # Bullish angle from low

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
                    strength = 0.6

                    # Stronger signals for key angles
                    key_angles = [45, 26.25, 18.75]
                    if angle_deg in key_angles:
                        strength += 0.2

                    signal_type = (
                        "gann_angle_support"
                        if direction == 1
                        else "gann_angle_resistance"
                    )

                    return SignalResult(
                        signal_type=signal_type,
                        strength=strength,
                        direction=direction,
                        description=f"Gann {angle_deg}° angle level",
                        timestamp=data.index[index],
                        metadata={
                            "pattern": "gann_angle",
                            "angle": angle_deg,
                            "pivot_price": pivot_price,
                            "confidence": strength,
                        },
                    )

        return None


class GannSquareRecognizer(PatternRecognizer):
    """Recognizes Gann square of 9 levels."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.lookback_period = config.get('lookback_period', 30) if config else 30
        self.gann_analyzer = GannAnalyzer()

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Gann square levels at the given index."""
        if index < self.lookback_period:
            return None

        current_price = data.iloc[index]["close"]

        # Calculate Gann square from recent high-low range
        lookback_data = data.iloc[index - self.lookback_period : index + 1]
        recent_high = lookback_data["high"].max()
        recent_low = lookback_data["low"].min()

        square_levels = self.gann_analyzer.calculate_gann_square(
            recent_high, recent_low
        )

        # Check if current price is near a square level
        for level_name, level_price in square_levels.items():
            level_ratio = float(level_name.split("_")[1])

            tolerance = (recent_high - recent_low) * 0.015  # 1.5% tolerance

            if abs(current_price - level_price) <= tolerance:
                # Determine direction based on position relative to midpoint
                midpoint = (recent_high + recent_low) / 2
                direction = 1 if current_price > midpoint else -1

                strength = 0.55

                # Stronger for key levels
                key_levels = [0.5, 1.0, 1.5, 2.0]
                if level_ratio in key_levels:
                    strength += 0.2

                signal_type = (
                    "gann_square_support"
                    if direction == 1
                    else "gann_square_resistance"
                )

                return SignalResult(
                    signal_type=signal_type,
                    strength=strength,
                    direction=direction,
                    description=f"Gann Square level {level_ratio}",
                    timestamp=data.index[index],
                    metadata={
                        "pattern": "gann_square",
                        "level": level_ratio,
                        "level_price": level_price,
                        "confidence": strength,
                    },
                )

        return None


class GannTimeClusterRecognizer(PatternRecognizer):
    """Recognizes Gann time clusters and cycle alignments."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.lookback_period = config.get('lookback_period', 30) if config else 30
        self.gann_analyzer = GannAnalyzer()

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Gann time clusters at the given index."""
        time_cluster = self.gann_analyzer.find_gann_time_clusters(
            data, index, lookback=self.lookback_period
        )

        if time_cluster:
            strength = time_cluster["strength"]
            direction = 1 if time_cluster["type"] == "time_cluster_lows" else -1

            signal_type = (
                "gann_time_cluster_support"
                if direction == 1
                else "gann_time_cluster_resistance"
            )

            return SignalResult(
                signal_type=signal_type,
                strength=strength,
                direction=direction,
                description=f"Gann Time Cluster: {time_cluster['type'].replace('_', ' ').title()}",
                timestamp=data.index[index],
                metadata={
                    "pattern": "gann_time_cluster",
                    "cluster_type": time_cluster["type"],
                    "interval": time_cluster["interval"],
                    "pivot_count": len(time_cluster["pivots"]),
                    "confidence": strength,
                },
            )

        return None
