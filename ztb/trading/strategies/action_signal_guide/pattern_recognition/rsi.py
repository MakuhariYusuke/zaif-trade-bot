"""
RSI (Relative Strength Index) Pattern Recognizer
既存のRSI特徴量クラスを使用したパターン認識
"""

from typing import Dict, Any, Optional
import pandas as pd

from ztb.features.momentum.rsi import compute_rsi
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    PatternRecognizer,
    SignalResult,
)


class RSIPatternRecognizer(PatternRecognizer):
    """
    RSI-based pattern recognition using existing RSI feature class.
    既存のRSI特徴量クラスを使用したパターン認識
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.rsi_period = self.config.get("rsi_period", 14)
        self.overbought_level = self.config.get("overbought_level", 70)
        self.oversold_level = self.config.get("oversold_level", 30)
        self.divergence_lookback = self.config.get("divergence_lookback", 5)

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """
        Recognize RSI-based patterns.
        RSIベースのパターン認識
        """
        if not self.validate_data(data):
            return None

        if len(data) < self.rsi_period + self.divergence_lookback:
            return None

        # Calculate RSI using existing feature class
        rsi_values = compute_rsi(data, period=self.rsi_period)

        if rsi_values.empty or rsi_values.isna().all():
            return None

        current_rsi = rsi_values.iloc[index] if index < len(rsi_values) else rsi_values.iloc[-1]
        previous_rsi = rsi_values.iloc[index-1] if index > 0 and index-1 < len(rsi_values) else current_rsi

        # Check for overbought/oversold signals
        if current_rsi <= self.oversold_level and previous_rsi > self.oversold_level:
            # RSI crossed below oversold level - potential buy signal
            strength = (self.oversold_level - current_rsi) / self.oversold_level
            return SignalResult(
                signal_type="RSI_oversold",
                strength=min(strength, 1.0),
                direction=ACTION_BUY,
                description=f"RSI oversold signal (RSI: {current_rsi:.2f})",
                confidence=min(0.8, strength * 0.8),
            )

        elif current_rsi >= self.overbought_level and previous_rsi < self.overbought_level:
            # RSI crossed above overbought level - potential sell signal
            strength = (current_rsi - self.overbought_level) / (100 - self.overbought_level)
            return SignalResult(
                signal_type="RSI_overbought",
                strength=min(strength, 1.0),
                direction=-ACTION_SELL,
                description=f"RSI overbought signal (RSI: {current_rsi:.2f})",
                confidence=min(0.8, strength * 0.8),
            )

        # Check for divergence signals
        divergence_signal = self._check_divergence(data, rsi_values, index)
        if divergence_signal:
            return divergence_signal

        # Center line cross signals
        if previous_rsi <= 50 and current_rsi > 50:
            return SignalResult(
                signal_type="RSI_centerline_bullish",
                strength=0.3,
                direction=ACTION_BUY,
                description=f"RSI center line cross up (RSI: {current_rsi:.2f})",
                confidence=0.6,
            )
        elif previous_rsi >= 50 and current_rsi < 50:
            return SignalResult(
                signal_type="RSI_centerline_bearish",
                strength=0.3,
                direction=-ACTION_SELL,
                description=f"RSI center line cross down (RSI: {current_rsi:.2f})",
                confidence=0.6,
            )

        return None

    def _check_divergence(self, data: pd.DataFrame, rsi_values: pd.Series, index: int) -> Optional[SignalResult]:
        """
        Check for RSI divergence patterns.
        RSIダイバージェンスパターンのチェック
        """
        if len(rsi_values) < self.divergence_lookback + 2:
            return None

        # Get recent data
        start_idx = max(0, index - self.divergence_lookback)
        recent_prices = data['close'].iloc[start_idx:index+1]
        recent_rsi = rsi_values.iloc[start_idx:index+1]

        if len(recent_prices) < 2 or len(recent_rsi) < 2:
            return None

        # Check for bullish divergence (price making lower low, RSI making higher low)
        price_trend = recent_prices.iloc[-1] < recent_prices.iloc[0]
        rsi_trend = recent_rsi.iloc[-1] > recent_rsi.iloc[0]

        if price_trend and rsi_trend:
            return SignalResult(
                signal_type="RSI_bullish_divergence",
                strength=0.4,
                direction=ACTION_BUY,
                description="RSI bullish divergence detected",
                confidence=0.7,
            )

        # Check for bearish divergence (price making higher high, RSI making lower high)
        price_trend = recent_prices.iloc[-1] > recent_prices.iloc[0]
        rsi_trend = recent_rsi.iloc[-1] < recent_rsi.iloc[0]

        if price_trend and rsi_trend:
            return SignalResult(
                signal_type="RSI_bearish_divergence",
                strength=0.4,
                direction=-ACTION_SELL,
                description="RSI bearish divergence detected",
                confidence=0.7,
            )

        return None