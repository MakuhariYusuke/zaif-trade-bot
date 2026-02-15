"""
Heikin-Ashi pattern recognition for Action Signal Guide.

Heikin-Ashi is a Japanese candlestick technique that modifies the traditional
candlestick chart to better reflect the trend and momentum.
"""

from typing import Optional, TypedDict

import pandas as pd


from ztb.features.generators.technical.trend.heikin_ashi import HeikinAshi

from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    MultiTimeframeData,
    PatternRecognizer,
    SignalResult,
)


class HeikinAshiTrendSignal(TypedDict):
    """Internal signal payload for Heikin-Ashi trend analysis."""

    direction: float
    strength: float
    description: str
    confidence: float


class HeikinAshiRecognizer(PatternRecognizer):
    """
    Recognizes patterns using Heikin-Ashi candlesticks.

    Heikin-Ashi candlesticks smooth price action and make trends more visible.
    The signals are based on the relationship between consecutive Heikin-Ashi
    candlesticks and their color changes.
    """

    def __init__(self, config: Optional[dict[str, object]] = None):
        super().__init__(config)
        self.pattern_type = "heikin_ashi"
        self.period = int(self.config.get("period", 1))  # Number of periods to look back
        self.trend_threshold = float(
            self.config.get("trend_threshold", 0.001)
        )  # Minimum trend strength
        self.volume_weighted = bool(
            self.config.get("volume_weighted", False)
        )  # Use volume weighting

        # Use existing HeikinAshi feature class
        self.heikin_ashi_calculator = HeikinAshi()

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[MultiTimeframeData] = None,
    ) -> Optional[SignalResult]:
        """
        Recognize Heikin-Ashi patterns in the data.

        Args:
            data: OHLCV DataFrame
            index: Index to analyze (-1 for latest)

        Returns:
            SignalResult if pattern detected, None otherwise
        """
        if len(data) < 2:
            return None

        resolved_index = self.resolve_analysis_index(
            len(data), index, min_required_index=1
        )
        if resolved_index is None:
            return None

        analysis_data = data.iloc[: resolved_index + 1]
        ha_data = self._calculate_heikin_ashi(analysis_data)
        local_index = len(ha_data) - 1

        if local_index < 1:
            return None

        current = ha_data.iloc[local_index]
        previous = ha_data.iloc[local_index - 1]

        # Analyze trend based on Heikin-Ashi candles
        signal = self._analyze_trend(current, previous)

        if signal:
            return SignalResult(
                signal_type="heikin_ashi",
                strength=abs(signal["strength"]),
                direction=signal["direction"],
                description=signal["description"],
                confidence=self.clamp(signal["confidence"], 0.0, 1.0),
            )

        return None

    def _calculate_heikin_ashi(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heikin-Ashi values using the existing HeikinAshi feature class.

        Returns:
            DataFrame with HA_Open, HA_High, HA_Low, HA_Close, HA_Body columns
        """
        # Use the existing HeikinAshi feature class
        ha_data = self.heikin_ashi_calculator.compute(data)

        # Rename columns to match expected format
        ha_data = ha_data.rename(
            columns={
                "ha_open": "HA_Open",
                "ha_high": "HA_High",
                "ha_low": "HA_Low",
                "ha_close": "HA_Close",
            }
        )

        # Calculate body size for analysis
        ha_data["HA_Body"] = abs(ha_data["HA_Close"] - ha_data["HA_Open"])

        return ha_data

    def _analyze_trend(
        self, current: pd.Series, previous: pd.Series
    ) -> Optional[HeikinAshiTrendSignal]:
        """
        Analyze trend based on Heikin-Ashi candle patterns using ratio-based measurements.

        Returns signal dictionary or None if no clear signal.
        """
        # Determine candle colors
        current_green = float(current["HA_Close"]) > float(current["HA_Open"])
        previous_green = float(previous["HA_Close"]) > float(previous["HA_Open"])

        # Calculate ratio-based trend strength
        # Use body-to-range ratio for more accurate trend measurement
        current_range = float(current["HA_High"] - current["HA_Low"])
        current_body = abs(float(current["HA_Close"]) - float(current["HA_Open"]))

        if current_range > 0:
            body_to_range_ratio = current_body / current_range
        else:
            body_to_range_ratio = 0.0

        # Calculate trend momentum using price change ratios
        prev_open = float(previous["HA_Open"])
        if prev_open > 0:
            trend_momentum = (
                float(current["HA_Close"]) - float(previous["HA_Close"])
            ) / prev_open
        else:
            trend_momentum = 0.0

        # Combine body ratio and momentum for overall strength
        trend_strength = (body_to_range_ratio * 0.6) + (abs(trend_momentum) * 0.4)

        # Ensure reasonable bounds
        trend_strength = min(1.0, max(0.0, trend_strength))

        # Strong trend signals
        if trend_strength > self.trend_threshold:
            # Bullish trend continuation (green candle after green)
            if current_green and previous_green:
                if float(current["HA_Close"]) > float(previous["HA_Close"]):
                    return {
                        "direction": 1.0,  # Strong bullish signal
                        "strength": trend_strength,
                        "description": f"Heikin-Ashi: Strong bullish trend continuation (strength: {trend_strength:.4f})",
                        "confidence": min(0.9, 0.35 + trend_strength * 0.55),
                    }

            # Bearish trend continuation (red candle after red)
            elif not current_green and not previous_green:
                if float(current["HA_Close"]) < float(previous["HA_Close"]):
                    return {
                        "direction": -1.0,  # Strong bearish signal
                        "strength": trend_strength,
                        "description": f"Heikin-Ashi: Strong bearish trend continuation (strength: {trend_strength:.4f})",
                        "confidence": min(0.9, 0.35 + trend_strength * 0.55),
                    }

        # Reversal signals
        # Bullish reversal (green after red)
        if current_green and not previous_green:
            if float(current["HA_Close"]) > float(previous["HA_Open"]):
                return {
                    "direction": 0.7,  # Moderate bullish signal
                    "strength": trend_strength,
                    "description": f"Heikin-Ashi: Bullish reversal signal (strength: {trend_strength:.4f})",
                    "confidence": min(0.7, 0.25 + trend_strength * 0.45),
                }

        # Bearish reversal (red after green)
        elif not current_green and previous_green:
            if float(current["HA_Close"]) < float(previous["HA_Open"]):
                return {
                    "direction": -0.7,  # Moderate bearish signal
                    "strength": trend_strength,
                    "description": f"Heikin-Ashi: Bearish reversal signal (strength: {trend_strength:.4f})",
                    "confidence": min(0.7, 0.25 + trend_strength * 0.45),
                }

        # Doji or weak signals - neutral
        if body_to_range_ratio < 0.1:  # Very small body indicates indecision
            return {
                "direction": 0.0,  # Neutral signal
                "strength": 0.1,
                "description": "Heikin-Ashi: Indecision/Doji pattern detected",
                "confidence": 0.5,
            }

        return None
