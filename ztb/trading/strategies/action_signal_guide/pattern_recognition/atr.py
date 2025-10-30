"""
ATR (Average True Range) Pattern Recognizer
既存のATR特徴量クラスを使用したパターン認識
"""

from typing import Any, Dict, Optional

import pandas as pd

from ztb.features.volatility.atr import compute_atr
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    PatternRecognizer,
    SignalResult,
)


class ATRPatternRecognizer(PatternRecognizer):
    """
    ATR-based pattern recognition using existing ATR feature class.
    既存のATR特徴量クラスを使用したパターン認識
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.atr_period = self.config.get("atr_period", 14)
        self.volatility_threshold = self.config.get("volatility_threshold", 1.0)
        self.trend_strength_period = self.config.get("trend_strength_period", 5)

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """
        Recognize ATR-based patterns.
        ATRベースのパターン認識
        """
        if not self.validate_data(data):
            return None

        if len(data) < self.atr_period + self.trend_strength_period:
            return None

        # Calculate ATR using existing feature class
        atr_values = compute_atr(data, period=self.atr_period)

        if atr_values.empty or atr_values.isna().all():
            return None

        current_atr = (
            atr_values.iloc[index] if index < len(atr_values) else atr_values.iloc[-1]
        )
        avg_atr = atr_values.tail(20).mean()  # Use longer period for baseline

        # Volatility breakout signals
        if current_atr > avg_atr * self.volatility_threshold:
            # High volatility - potential breakout
            return self._analyze_breakout(data, current_atr, avg_atr, index)

        # Trend strength analysis using ATR
        trend_signal = self._analyze_trend_strength(data, atr_values, index)
        if trend_signal:
            return trend_signal

        # Low volatility consolidation
        if current_atr < avg_atr * 0.8:
            return SignalResult(
                signal_type="ATR_low_volatility",
                strength=0.2,
                direction=0,  # 0.0
                description=f"ATR low volatility consolidation (ATR: {current_atr:.6f})",
                confidence=0.6,
            )

        return None

    def _analyze_breakout(
        self, data: pd.DataFrame, current_atr: float, avg_atr: float, index: int
    ) -> Optional[SignalResult]:
        """
        Analyze potential breakout during high volatility.
        高ボラティリティ時のブレイクアウト分析
        """
        start_idx = max(0, index - 4)
        recent_prices = data["close"].iloc[start_idx : index + 1]

        if len(recent_prices) < 2:
            return None

        price_change = (
            recent_prices.iloc[-1] - recent_prices.iloc[0]
        ) / recent_prices.iloc[0]

        volatility_ratio = current_atr / avg_atr
        strength = min(volatility_ratio / self.volatility_threshold, 1.0)

        if abs(price_change) > 0.005:  # 0.5% price movement
            if price_change > 0:
                return SignalResult(
                    signal_type="ATR_bullish_breakout",
                    strength=strength,
                    direction=1,  # 1.0
                    description=f"ATR bullish breakout (Vol: {volatility_ratio:.2f}, Price: +{price_change:.2%})",
                    confidence=min(0.8, strength * 0.8),
                )
            else:
                return SignalResult(
                    signal_type="ATR_bearish_breakout",
                    strength=strength,
                    direction=-1,  # -1.0
                    description=f"ATR bearish breakout (Vol: {volatility_ratio:.2f}, Price: {price_change:.2%})",
                    confidence=min(0.8, strength * 0.8),
                )

        return SignalResult(
            signal_type="ATR_high_volatility",
            strength=strength * 0.5,
            direction=0,  # 0.0
            description=f"ATR high volatility, awaiting direction (Vol: {volatility_ratio:.2f})",
            confidence=0.6,
        )

    def _analyze_trend_strength(
        self, data: pd.DataFrame, atr_values: pd.Series, index: int
    ) -> Optional[SignalResult]:
        """
        Analyze trend strength using ATR changes.
        ATR変化によるトレンド強度分析
        """
        if len(atr_values) < self.trend_strength_period + 5:
            return None

        # Calculate ATR trend
        start_idx = max(0, index - self.trend_strength_period + 1)
        recent_atr = atr_values.iloc[start_idx : index + 1]
        if len(recent_atr) < 2:
            return None
        atr_trend = (recent_atr.iloc[-1] - recent_atr.iloc[0]) / recent_atr.iloc[0]

        # Calculate price trend
        recent_prices = data["close"].iloc[start_idx : index + 1]
        if len(recent_prices) < 2:
            return None
        price_trend = (
            recent_prices.iloc[-1] - recent_prices.iloc[0]
        ) / recent_prices.iloc[0]

        # Strong trend with increasing ATR (healthy trend)
        if abs(price_trend) > 0.01 and atr_trend > 0.05:
            if price_trend > 0:
                strength = min(abs(price_trend) * 10, 0.6)
                return SignalResult(
                    signal_type="ATR_strong_bullish_trend",
                    strength=strength,
                    direction=1,  # 1.0
                    description=f"ATR strong bullish trend (Price: +{price_trend:.2%}, ATR: +{atr_trend:.2%})",
                    confidence=min(0.75, strength * 1.25),
                )
            else:
                strength = min(abs(price_trend) * 10, 0.6)
                return SignalResult(
                    signal_type="ATR_strong_bearish_trend",
                    strength=strength,
                    direction=-1,  # -1.0
                    description=f"ATR strong bearish trend (Price: {price_trend:.2%}, ATR: +{atr_trend:.2%})",
                    confidence=min(0.75, strength * 1.25),
                )

        # Weak trend with decreasing ATR (potential reversal)
        elif abs(price_trend) < 0.005 and atr_trend < -0.05:
            return SignalResult(
                signal_type="ATR_weakening_trend",
                strength=0.3,
                direction=0,  # 0.0
                description=f"ATR weakening trend (Price: {price_trend:.2%}, ATR: {atr_trend:.2%})",
                confidence=0.65,
            )

        return None
