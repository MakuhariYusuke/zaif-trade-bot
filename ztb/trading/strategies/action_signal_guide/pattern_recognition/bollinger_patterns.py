"""
Bollinger Bands Pattern Recognizer
ボリンジャーバンドパターン認識 - ボラティリティベースのシグナル生成
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from .base import PatternRecognizer, SignalResult
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.features.volatility.bollinger import (
    compute_bb_upper, compute_bb_lower, compute_bb_middle, compute_bb_width
)


class BollingerBandsRecognizer(PatternRecognizer):
    """
    Bollinger Bands pattern recognizer.
    ボリンジャーバンドベースのパターン認識
    価格のボラティリティとトレンドを分析
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.period = self.config.get("period", 20)
        self.std_dev = self.config.get("std_dev", 2.0)
        self.squeeze_threshold = self.config.get("squeeze_threshold", 0.05)  # バンド幅の収縮閾値
        self.expansion_threshold = self.config.get("expansion_threshold", 0.15)  # バンド幅の拡大閾値
        self.touch_distance = self.config.get("touch_distance", 0.001)  # バンドタッチの距離閾値

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """
        Recognize Bollinger Bands patterns.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze

        Returns:
            SignalResult with Bollinger Bands analysis
        """
        if not self.validate_data(data):
            return None

        if len(data) < self.period:
            return SignalResult(
                signal_type="bb_insufficient_data",
                strength=0.0,
                direction=ACTION_HOLD,
                description=f"Insufficient data for Bollinger Bands (need {self.period} periods)",
                metadata={},
                validity_period=1,
                risk_level="low"
            )

        try:
            # Calculate Bollinger Bands
            bb_upper = compute_bb_upper(data, self.period, self.std_dev)
            bb_lower = compute_bb_lower(data, self.period, self.std_dev)
            bb_middle = compute_bb_middle(data, self.period)
            bb_width = compute_bb_width(data, self.period, self.std_dev)

            current_price = data.iloc[index]['close']
            current_upper = bb_upper.iloc[index]
            current_lower = bb_lower.iloc[index]
            current_middle = bb_middle.iloc[index]
            current_width = bb_width.iloc[index]

            # Get previous values for trend analysis
            if index > 0:
                prev_price = data.iloc[index-1]['close']
                prev_upper = bb_upper.iloc[index-1]
                prev_lower = bb_lower.iloc[index-1]
                prev_middle = bb_middle.iloc[index-1]
                prev_width = bb_width.iloc[index-1]
            else:
                prev_price = current_price
                prev_upper = current_upper
                prev_lower = current_lower
                prev_middle = current_middle
                prev_width = current_width

            # 1. Band Touch Signals (価格がバンドにタッチ)
            upper_touch_distance = abs(current_price - current_upper) / current_price
            lower_touch_distance = abs(current_price - current_lower) / current_price

            if upper_touch_distance <= self.touch_distance:
                # Upper band touch - potential sell signal
                strength = min(0.8, (1 - upper_touch_distance / self.touch_distance) * 0.8)
                return SignalResult(
                    signal_type="bb_upper_touch",
                    strength=strength,
                    direction=ACTION_SELL,
                    description=f"Price touched upper Bollinger Band at {current_price:.4f}",
                    metadata={
                        "band": "upper",
                        "price": current_price,
                        "band_value": current_upper,
                        "distance": upper_touch_distance,
                        "width": current_width
                    },
                    validity_period=3,
                    risk_level="medium"
                )

            elif lower_touch_distance <= self.touch_distance:
                # Lower band touch - potential buy signal
                strength = min(0.8, (1 - lower_touch_distance / self.touch_distance) * 0.8)
                return SignalResult(
                    signal_type="bb_lower_touch",
                    strength=strength,
                    direction=ACTION_BUY,
                    description=f"Price touched lower Bollinger Band at {current_price:.4f}",
                    metadata={
                        "band": "lower",
                        "price": current_price,
                        "band_value": current_lower,
                        "distance": lower_touch_distance,
                        "width": current_width
                    },
                    validity_period=3,
                    risk_level="medium"
                )

            # 2. Band Squeeze (バンドの収縮 - ボラティリティ低下)
            if current_width <= self.squeeze_threshold:
                strength = min(0.6, (self.squeeze_threshold - current_width) / self.squeeze_threshold)
                return SignalResult(
                    signal_type="bb_squeeze",
                    strength=strength,
                    direction=ACTION_HOLD,
                    description=f"Bollinger Bands squeeze detected (width: {current_width:.4f})",
                    metadata={
                        "width": current_width,
                        "threshold": self.squeeze_threshold,
                        "squeeze_ratio": current_width / self.squeeze_threshold
                    },
                    validity_period=5,
                    risk_level="low"
                )

            # 3. Band Expansion (バンドの拡大 - ボラティリティ上昇)
            if current_width >= self.expansion_threshold:
                strength = min(0.7, (current_width - self.expansion_threshold) / (1 - self.expansion_threshold))
                return SignalResult(
                    signal_type="bb_expansion",
                    strength=strength,
                    direction=ACTION_HOLD,
                    description=f"Bollinger Bands expansion detected (width: {current_width:.4f})",
                    metadata={
                        "width": current_width,
                        "threshold": self.expansion_threshold,
                        "expansion_ratio": current_width / self.expansion_threshold
                    },
                    validity_period=3,
                    risk_level="high"
                )

            # 4. Middle Band Cross (ミドルバンドとのクロス)
            if (prev_price <= prev_middle and current_price > current_middle):
                # Bullish cross of middle band
                strength = 0.5
                return SignalResult(
                    signal_type="bb_middle_cross_bullish",
                    strength=strength,
                    direction=ACTION_BUY,
                    description=f"Price crossed above middle Bollinger Band at {current_price:.4f}",
                    metadata={
                        "cross_type": "bullish",
                        "middle_value": current_middle,
                        "prev_price": prev_price,
                        "current_price": current_price
                    },
                    validity_period=2,
                    risk_level="medium"
                )

            elif (prev_price >= prev_middle and current_price < current_middle):
                # Bearish cross of middle band
                strength = 0.5
                return SignalResult(
                    signal_type="bb_middle_cross_bearish",
                    strength=strength,
                    direction=ACTION_SELL,
                    description=f"Price crossed below middle Bollinger Band at {current_price:.4f}",
                    metadata={
                        "cross_type": "bearish",
                        "middle_value": current_middle,
                        "prev_price": prev_price,
                        "current_price": current_price
                    },
                    validity_period=2,
                    risk_level="medium"
                )

            # 5. Band Walk (バンド内での動き分析)
            # Price near upper band within bands
            if current_price > current_middle and current_price <= current_upper * 0.98:
                upper_position = (current_price - current_middle) / (current_upper - current_middle)
                if upper_position > 0.7:
                    strength = min(0.4, upper_position * 0.4)
                    return SignalResult(
                        signal_type="bb_upper_walk",
                        strength=strength,
                        direction=ACTION_HOLD,
                        description=f"Price walking upper band region (position: {upper_position:.2f})",
                        metadata={
                            "position": upper_position,
                            "region": "upper",
                            "middle": current_middle,
                            "upper": current_upper
                        },
                        validity_period=1,
                        risk_level="low"
                    )

            # Price near lower band within bands
            elif current_price < current_middle and current_price >= current_lower * 1.02:
                lower_position = (current_middle - current_price) / (current_middle - current_lower)
                if lower_position > 0.7:
                    strength = min(0.4, lower_position * 0.4)
                    return SignalResult(
                        signal_type="bb_lower_walk",
                        strength=strength,
                        direction=ACTION_HOLD,
                        description=f"Price walking lower band region (position: {lower_position:.2f})",
                        metadata={
                            "position": lower_position,
                            "region": "lower",
                            "middle": current_middle,
                            "lower": current_lower
                        },
                        validity_period=1,
                        risk_level="low"
                    )

            # 6. Neutral zone (ミドルバンド付近)
            middle_distance = abs(current_price - current_middle) / current_middle
            if middle_distance <= 0.01:  # Within 1% of middle band
                return SignalResult(
                    signal_type="bb_neutral",
                    strength=0.0,
                    direction=ACTION_HOLD,
                    description=f"Price near middle Bollinger Band (distance: {middle_distance:.4f})",
                    metadata={
                        "middle_distance": middle_distance,
                        "middle_value": current_middle,
                        "width": current_width
                    },
                    validity_period=1,
                    risk_level="low"
                )

            # Default: no significant signal
            return SignalResult(
                signal_type="bb_neutral",
                strength=0.0,
                direction=ACTION_HOLD,
                description=f"Price within normal Bollinger Bands range",
                metadata={
                    "price": current_price,
                    "upper": current_upper,
                    "middle": current_middle,
                    "lower": current_lower,
                    "width": current_width,
                    "upper_distance": (current_upper - current_price) / current_price,
                    "lower_distance": (current_price - current_lower) / current_price
                },
                validity_period=1,
                risk_level="low"
            )

        except Exception as e:
            return SignalResult(
                signal_type="bb_error",
                strength=0.0,
                direction=ACTION_HOLD,
                description=f"Bollinger Bands calculation error: {str(e)}",
                metadata={"error": str(e)},
                validity_period=1,
                risk_level="low"
            )