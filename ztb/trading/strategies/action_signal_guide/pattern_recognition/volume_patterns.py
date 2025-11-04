"""
Volume Pattern Recognizers - Chaikin AD

This module provides pattern recognition for volume-based technical indicators.
"""

from typing import Any, Dict, Optional

import pandas as pd

try:
    from ztb.features.generators.technical.volume.chaikin_ad import compute_chaikin_ad
except ImportError:
    def compute_chaikin_ad(df: pd.DataFrame) -> pd.Series:
        return pd.Series([0.0] * len(df), index=df.index)

from .base import PatternRecognizer, SignalResult


class ChaikinADRecognizer(PatternRecognizer):
    """
    Chaikin Accumulation/Distribution pattern recognizer.
    Identifies accumulation/distribution patterns using Chaikin AD.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.pattern_type = "chaikin_ad"
        self.confirmation_period = self.config.get("confirmation_period", 5)
        self.divergence_threshold = self.config.get("divergence_threshold", 0.1)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """
        Recognize Chaikin AD patterns.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze

        Returns:
            SignalResult with Chaikin AD analysis
        """
        if index < 20:  # Need sufficient data for Chaikin AD calculation
            return SignalResult(
                signal_type="chaikin_ad_neutral",
                strength=0.0,
                direction=0.0,
                description="Insufficient data for Chaikin AD analysis",
                metadata={},
                validity_period=1,
                risk_level="low",
            )

        try:
            chaikin_ad_series = compute_chaikin_ad(data)
            current_ad = chaikin_ad_series.iloc[index]

            # Analyze trend direction and strength
            if index >= self.confirmation_period:
                recent_ad = chaikin_ad_series.iloc[
                    index - self.confirmation_period : index + 1
                ]
                ad_trend = self._calculate_volume_trend_strength(recent_ad)

                # Check for divergence with price
                price_trend = self._calculate_price_trend(
                    data.iloc[index - self.confirmation_period : index + 1]
                )

                if (
                    ad_trend > self.divergence_threshold
                    and price_trend < -self.divergence_threshold
                ):
                    # Bullish divergence - Chaikin AD rising while price falling
                    return SignalResult(
                        signal_type="chaikin_ad_bullish_divergence",
                        strength=min(ad_trend, 1.0),
                        direction=1.0,  # Buy signal
                        description=f"Chaikin AD bullish divergence detected (AD trend: {ad_trend:.3f}, Price trend: {price_trend:.3f})",
                        metadata={
                            "ad_trend": ad_trend,
                            "price_trend": price_trend,
                            "divergence": "bullish",
                        },
                        validity_period=10,
                        risk_level="medium",
                    )
                elif (
                    ad_trend < -self.divergence_threshold
                    and price_trend > self.divergence_threshold
                ):
                    # Bearish divergence - Chaikin AD falling while price rising
                    return SignalResult(
                        signal_type="chaikin_ad_bearish_divergence",
                        strength=min(abs(ad_trend), 1.0),
                        direction=-1.0,  # Sell signal
                        description=f"Chaikin AD bearish divergence detected (AD trend: {ad_trend:.3f}, Price trend: {price_trend:.3f})",
                        metadata={
                            "ad_trend": ad_trend,
                            "price_trend": price_trend,
                            "divergence": "bearish",
                        },
                        validity_period=10,
                        risk_level="medium",
                    )
                elif ad_trend > self.divergence_threshold:
                    # Strong accumulation - potential buy signal
                    return SignalResult(
                        signal_type="chaikin_ad_accumulation",
                        strength=min(ad_trend, 0.8),
                        direction=1.0,  # Buy signal
                        description=f"Chaikin AD showing accumulation (trend strength: {ad_trend:.3f})",
                        metadata={"ad_trend": ad_trend, "pattern": "accumulation"},
                        validity_period=5,
                        risk_level="low",
                    )
                elif ad_trend < -self.divergence_threshold:
                    # Strong distribution - potential sell signal
                    return SignalResult(
                        signal_type="chaikin_ad_distribution",
                        strength=min(abs(ad_trend), 0.8),
                        direction=-1.0,  # Sell signal
                        description=f"Chaikin AD showing distribution (trend strength: {ad_trend:.3f})",
                        metadata={"ad_trend": ad_trend, "pattern": "distribution"},
                        validity_period=5,
                        risk_level="low",
                    )

            # Neutral or weak signal
            return SignalResult(
                signal_type="chaikin_ad_neutral",
                strength=0.0,
                direction=0.0,
                description=f"Chaikin AD neutral pattern (current value: {current_ad:.6f})",
                metadata={"chaikin_ad": current_ad, "pattern": "neutral"},
                validity_period=1,
                risk_level="low",
            )

        except Exception as e:
            return SignalResult(
                signal_type="chaikin_ad_error",
                strength=0.0,
                direction=0.0,
                description=f"Chaikin AD calculation error: {str(e)}",
                metadata={"error": str(e)},
                validity_period=1,
                risk_level="low",
            )

    def _calculate_volume_trend_strength(self, series: pd.Series) -> float:
        """
        Calculate the trend strength of a series using linear regression slope.

        Args:
            series: Time series data

        Returns:
            Trend strength (normalized slope)
        """
        if len(series) < 2:
            return 0.0

        # Simple trend calculation using first and last values
        start_val = series.iloc[0]
        end_val = series.iloc[-1]

        if start_val == 0:
            return 0.0

        # Normalize trend by dividing by absolute start value and period length
        trend = (end_val - start_val) / abs(start_val) / len(series)
        return float(trend)

    def _calculate_price_trend(self, price_data: pd.DataFrame) -> float:
        """
        Calculate price trend using close prices.

        Args:
            price_data: OHLCV DataFrame slice

        Returns:
            Price trend strength
        """
        if len(price_data) < 2:
            return 0.0

        close_prices = price_data["close"]
        start_price = close_prices.iloc[0]
        end_price = close_prices.iloc[-1]

        if start_price == 0:
            return 0.0

        # Normalize trend
        trend = (end_price - start_price) / start_price / len(close_prices)
        return float(trend)
