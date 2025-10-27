"""
ADX (Average Directional Index) Pattern Recognizer
ADXパターン認識 - トレンド強度と方向性分析
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from .base import PatternRecognizer, SignalResult
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.features.trend.adx import compute_adx, compute_plus_di, compute_minus_di


class ADXRecognizer(PatternRecognizer):
    """
    ADX (Average Directional Index) pattern recognizer.
    ADXベースのパターン認識 - トレンド強度と方向性分析
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.period = self.config.get("period", 14)
        self.strong_trend_threshold = self.config.get("strong_trend_threshold", 25)
        self.weak_trend_threshold = self.config.get("weak_trend_threshold", 20)
        self.di_cross_threshold = self.config.get("di_cross_threshold", 1.0)  # DIクロスの最小差

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """
        Recognize ADX patterns.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze

        Returns:
            SignalResult with ADX analysis
        """
        if not self.validate_data(data):
            return None

        if len(data) < self.period * 2:  # Need sufficient data for ADX calculation
            return SignalResult(
                signal_type="adx_insufficient_data",
                strength=0.0,
                direction=ACTION_HOLD,
                description=f"Insufficient data for ADX (need {self.period * 2} periods)",
                metadata={},
                validity_period=1,
                risk_level="low"
            )

        try:
            # Calculate ADX and DI values
            adx_series = compute_adx(data, self.period)
            plus_di_series = compute_plus_di(data, self.period)
            minus_di_series = compute_minus_di(data, self.period)

            current_adx = adx_series.iloc[index]
            current_plus_di = plus_di_series.iloc[index]
            current_minus_di = minus_di_series.iloc[index]

            # Get previous values for trend analysis
            if index > 0:
                prev_adx = adx_series.iloc[index-1]
                prev_plus_di = plus_di_series.iloc[index-1]
                prev_minus_di = minus_di_series.iloc[index-1]
            else:
                prev_adx = current_adx
                prev_plus_di = current_plus_di
                prev_minus_di = current_minus_di

            # 1. Strong Trend Detection (ADX > strong_trend_threshold)
            if current_adx >= self.strong_trend_threshold:
                # Determine trend direction
                di_difference = current_plus_di - current_minus_di

                if di_difference > self.di_cross_threshold:
                    # Strong uptrend
                    strength = min(0.9, current_adx / 50.0)  # Normalize strength
                    return SignalResult(
                        signal_type="adx_strong_uptrend",
                        strength=strength,
                        direction=ACTION_BUY,
                        description=f"Strong uptrend detected (ADX: {current_adx:.2f}, +DI: {current_plus_di:.2f})",
                        metadata={
                            "adx": current_adx,
                            "plus_di": current_plus_di,
                            "minus_di": current_minus_di,
                            "di_difference": di_difference,
                            "trend_strength": "strong",
                            "direction": "up"
                        },
                        validity_period=5,
                        risk_level="medium"
                    )

                elif di_difference < -self.di_cross_threshold:
                    # Strong downtrend
                    strength = min(0.9, current_adx / 50.0)  # Normalize strength
                    return SignalResult(
                        signal_type="adx_strong_downtrend",
                        strength=strength,
                        direction=ACTION_SELL,
                        description=f"Strong downtrend detected (ADX: {current_adx:.2f}, -DI: {current_minus_di:.2f})",
                        metadata={
                            "adx": current_adx,
                            "plus_di": current_plus_di,
                            "minus_di": current_minus_di,
                            "di_difference": di_difference,
                            "trend_strength": "strong",
                            "direction": "down"
                        },
                        validity_period=5,
                        risk_level="medium"
                    )

                else:
                    # Strong trend but unclear direction
                    strength = min(0.7, current_adx / 50.0)
                    return SignalResult(
                        signal_type="adx_strong_trend_unclear",
                        strength=strength,
                        direction=ACTION_HOLD,
                        description=f"Strong trend detected but direction unclear (ADX: {current_adx:.2f})",
                        metadata={
                            "adx": current_adx,
                            "plus_di": current_plus_di,
                            "minus_di": current_minus_di,
                            "di_difference": di_difference,
                            "trend_strength": "strong",
                            "direction": "unclear"
                        },
                        validity_period=3,
                        risk_level="medium"
                    )

            # 2. Weak Trend / Ranging Market (ADX < weak_trend_threshold)
            elif current_adx <= self.weak_trend_threshold:
                strength = max(0.1, (self.weak_trend_threshold - current_adx) / self.weak_trend_threshold)
                return SignalResult(
                    signal_type="adx_weak_trend",
                    strength=strength,
                    direction=ACTION_HOLD,
                    description=f"Weak trend or ranging market (ADX: {current_adx:.2f})",
                    metadata={
                        "adx": current_adx,
                        "plus_di": current_plus_di,
                        "minus_di": current_minus_di,
                        "trend_strength": "weak"
                    },
                    validity_period=2,
                    risk_level="low"
                )

            # 3. DI Cross Signals (トレンド変化の兆候)
            # Bullish DI cross: +DI crosses above -DI
            if (prev_plus_di <= prev_minus_di and current_plus_di > current_minus_di):
                if abs(current_plus_di - current_minus_di) >= self.di_cross_threshold:
                    strength = min(0.6, abs(current_plus_di - current_minus_di) / 10.0)
                    return SignalResult(
                        signal_type="adx_di_cross_bullish",
                        strength=strength,
                        direction=ACTION_BUY,
                        description=f"+DI crossed above -DI (potential trend change to up)",
                        metadata={
                            "adx": current_adx,
                            "plus_di": current_plus_di,
                            "minus_di": current_minus_di,
                            "prev_plus_di": prev_plus_di,
                            "prev_minus_di": prev_minus_di,
                            "cross_type": "bullish"
                        },
                        validity_period=3,
                        risk_level="medium"
                    )

            # Bearish DI cross: -DI crosses above +DI
            elif (prev_minus_di <= prev_plus_di and current_minus_di > current_plus_di):
                if abs(current_minus_di - current_plus_di) >= self.di_cross_threshold:
                    strength = min(0.6, abs(current_minus_di - current_plus_di) / 10.0)
                    return SignalResult(
                        signal_type="adx_di_cross_bearish",
                        strength=strength,
                        direction=ACTION_SELL,
                        description=f"-DI crossed above +DI (potential trend change to down)",
                        metadata={
                            "adx": current_adx,
                            "plus_di": current_plus_di,
                            "minus_di": current_minus_di,
                            "prev_plus_di": prev_plus_di,
                            "prev_minus_di": prev_minus_di,
                            "cross_type": "bearish"
                        },
                        validity_period=3,
                        risk_level="medium"
                    )

            # 4. Moderate Trend with Direction
            else:
                di_difference = current_plus_di - current_minus_di

                if di_difference > self.di_cross_threshold:
                    # Moderate uptrend
                    strength = min(0.5, current_adx / 40.0)
                    return SignalResult(
                        signal_type="adx_moderate_uptrend",
                        strength=strength,
                        direction=ACTION_BUY,
                        description=f"Moderate uptrend (ADX: {current_adx:.2f})",
                        metadata={
                            "adx": current_adx,
                            "plus_di": current_plus_di,
                            "minus_di": current_minus_di,
                            "di_difference": di_difference,
                            "trend_strength": "moderate",
                            "direction": "up"
                        },
                        validity_period=3,
                        risk_level="medium"
                    )

                elif di_difference < -self.di_cross_threshold:
                    # Moderate downtrend
                    strength = min(0.5, current_adx / 40.0)
                    return SignalResult(
                        signal_type="adx_moderate_downtrend",
                        strength=strength,
                        direction=ACTION_SELL,
                        description=f"Moderate downtrend (ADX: {current_adx:.2f})",
                        metadata={
                            "adx": current_adx,
                            "plus_di": current_plus_di,
                            "minus_di": current_minus_di,
                            "di_difference": di_difference,
                            "trend_strength": "moderate",
                            "direction": "down"
                        },
                        validity_period=3,
                        risk_level="medium"
                    )

            # 5. ADX Rising (トレンド強度が増加中)
            if current_adx > prev_adx and current_adx > self.weak_trend_threshold:
                adx_change = (current_adx - prev_adx) / prev_adx
                if adx_change > 0.05:  # 5% increase
                    strength = min(0.4, adx_change * 5.0)
                    return SignalResult(
                        signal_type="adx_strengthening",
                        strength=strength,
                        direction=ACTION_HOLD,
                        description=f"ADX strengthening ({adx_change:.1%} increase)",
                        metadata={
                            "adx": current_adx,
                            "prev_adx": prev_adx,
                            "adx_change": adx_change,
                            "trend_status": "strengthening"
                        },
                        validity_period=2,
                        risk_level="low"
                    )

            # Default: neutral signal
            return SignalResult(
                signal_type="adx_neutral",
                strength=0.0,
                direction=ACTION_HOLD,
                description=f"ADX in neutral zone (ADX: {current_adx:.2f})",
                metadata={
                    "adx": current_adx,
                    "plus_di": current_plus_di,
                    "minus_di": current_minus_di,
                    "trend_strength": "neutral"
                },
                validity_period=1,
                risk_level="low"
            )

        except Exception as e:
            return SignalResult(
                signal_type="adx_error",
                strength=0.0,
                direction=ACTION_HOLD,
                description=f"ADX calculation error: {str(e)}",
                metadata={"error": str(e)},
                validity_period=1,
                risk_level="low"
            )