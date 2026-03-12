"""
Volume Pattern Recognizers - Chaikin AD

This module provides pattern recognition for volume-based technical indicators.
"""

import pandas as pd

try:
    from ztb.features.generators.technical.volume.chaikin_ad import compute_chaikin_ad
except ImportError:
    def compute_chaikin_ad(df: pd.DataFrame) -> pd.Series:
        return pd.Series([0.0] * len(df), index=df.index)

from .base import IndicatorPatternRecognizer, MultiTimeframeData, SignalResult

class ChaikinADRecognizer(IndicatorPatternRecognizer):
    """
    Chaikin Accumulation/Distribution pattern recognizer.
    Identifies accumulation/distribution patterns using Chaikin AD.
    """

    def __init__(self, config: dict[str, object] | None = None):
        super().__init__(config)
        self.pattern_type = "chaikin_ad"
        self.confirmation_period = int(self.config.get("confirmation_period", 5))
        self.divergence_threshold = float(self.config.get("divergence_threshold", 0.1))

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """
        Recognize Chaikin AD patterns.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze

        Returns:
            SignalResult with Chaikin AD analysis
        """
        min_required_periods = max(20, self.confirmation_period + 1)
        resolved_index = self.resolve_indicator_index(
            data,
            index,
            min_required_periods=min_required_periods,
        )
        if resolved_index is None:  # Need sufficient data for Chaikin AD calculation
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
            analysis_data, local_index = self.build_indicator_view(
                data,
                resolved_index,
                min_required_periods=min_required_periods,
                window_multiplier=12,
                min_window=max(160, min_required_periods * 8),
                max_window=1200,
            )

            chaikin_ad_series = compute_chaikin_ad(analysis_data)
            if chaikin_ad_series.empty or chaikin_ad_series.isna().all():
                return SignalResult(
                    signal_type="chaikin_ad_neutral",
                    strength=0.0,
                    direction=0.0,
                    description="Chaikin AD calculation returned empty values",
                    metadata={},
                    validity_period=1,
                    risk_level="low",
                )

            current_ad = float(chaikin_ad_series.iloc[local_index])

            # Analyze trend direction and strength
            if local_index >= self.confirmation_period:
                start_idx = local_index - self.confirmation_period
                recent_ad = chaikin_ad_series.iloc[
                    start_idx : local_index + 1
                ]
                ad_trend = self._calculate_volume_trend_strength(recent_ad)

                # Check for divergence with price
                price_trend = self._calculate_price_trend(
                    analysis_data.iloc[start_idx : local_index + 1]
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
        start_val = float(series.iloc[0])
        end_val = float(series.iloc[-1])

        # Normalize trend by dividing by absolute start value and period length
        normalized_change = self.safe_ratio(
            end_val - start_val,
            max(abs(start_val), 1e-9),
            default=0.0,
        )
        return float(normalized_change / max(len(series), 1))

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
        start_price = float(close_prices.iloc[0])
        end_price = float(close_prices.iloc[-1])

        # Normalize trend
        normalized_change = self.safe_ratio(
            end_price - start_price,
            max(abs(start_price), 1e-9),
            default=0.0,
        )
        return float(normalized_change / max(len(close_prices), 1))
