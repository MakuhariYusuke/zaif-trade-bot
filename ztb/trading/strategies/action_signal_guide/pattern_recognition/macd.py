"""
MACD (Moving Average Convergence Divergence) Pattern Recognizer
既存のMACD特徴量クラスを使用したパターン認識
"""

from typing import Dict, Any, Optional
import pandas as pd

from ztb.features.momentum.macd import compute_macd
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    PatternRecognizer,
    SignalResult,
)


class MACDPatternRecognizer(PatternRecognizer):
    """
    MACD-based pattern recognition using existing MACD feature class.
    既存のMACD特徴量クラスを使用したパターン認識
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.fast_period = self.config.get("fast_period", 12)
        self.slow_period = self.config.get("slow_period", 26)
        self.signal_period = self.config.get("signal_period", 9)
        self.histogram_threshold = self.config.get("histogram_threshold", 0.0)

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """
        Recognize MACD-based patterns.
        MACDベースのパターン認識
        """
        if not self.validate_data(data):
            return None

        min_periods = max(self.fast_period, self.slow_period) + self.signal_period
        if len(data) < min_periods:
            return None

        # Calculate MACD histogram using existing feature class
        try:
            macd_hist = compute_macd(
                data,
                fast_period=self.fast_period,
                slow_period=self.slow_period,
                signal_period=self.signal_period,
            )
        except Exception:
            # Fallback to manual calculation if TaLib fails
            macd_hist = self._calculate_macd_manual(data)

        if macd_hist.empty or macd_hist.isna().all():
            return None

        current_hist = macd_hist.iloc[index] if index < len(macd_hist) else macd_hist.iloc[-1]
        previous_hist = macd_hist.iloc[index-1] if index > 0 and index-1 < len(macd_hist) else 0

        # Zero line cross signals
        if previous_hist <= 0 and current_hist > 0:
            # MACD histogram crossed above zero - bullish signal
            strength = min(abs(current_hist) / abs(macd_hist.min()), 1.0)
            return SignalResult(
                signal_type="MACD_zero_cross_bullish",
                strength=strength,
                direction=ACTION_BUY,
                description=f"MACD histogram zero line cross up (Hist: {current_hist:.6f})",
                confidence=min(0.8, strength * 0.8),
            )

        elif previous_hist >= 0 and current_hist < 0:
            # MACD histogram crossed below zero - bearish signal
            strength = min(abs(current_hist) / abs(macd_hist.max()), 1.0)
            return SignalResult(
                signal_type="MACD_zero_cross_bearish",
                strength=strength,
                direction=-ACTION_SELL,
                description=f"MACD histogram zero line cross down (Hist: {current_hist:.6f})",
                confidence=min(0.8, strength * 0.8),
            )

        # Histogram momentum signals
        hist_change = current_hist - previous_hist
        if abs(hist_change) > self.histogram_threshold:
            if hist_change > 0 and current_hist > 0:
                # Increasing bullish momentum
                strength = min(abs(hist_change) / abs(macd_hist.std()), 0.5)
                return SignalResult(
                    signal_type="MACD_bullish_momentum",
                    strength=strength,
                    direction=ACTION_BUY,
                    description=f"MACD bullish momentum (Change: {hist_change:.6f})",
                    confidence=min(0.7, strength * 1.4),
                )
            elif hist_change < 0 and current_hist < 0:
                # Increasing bearish momentum
                strength = min(abs(hist_change) / abs(macd_hist.std()), 0.5)
                return SignalResult(
                    signal_type="MACD_bearish_momentum",
                    strength=strength,
                    direction=-ACTION_SELL,
                    description=f"MACD bearish momentum (Change: {hist_change:.6f})",
                    confidence=min(0.7, strength * 1.4),
                )

        # Convergence/divergence signals
        convergence_signal = self._check_convergence(data, macd_hist, index)
        if convergence_signal:
            return convergence_signal

        return None

    def _calculate_macd_manual(self, data: pd.DataFrame) -> pd.Series:
        """
        Manual MACD calculation as fallback.
        TaLibが失敗した場合の手動MACD計算
        """
        close = data['close']

        # Calculate EMAs
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        return histogram

    def _check_convergence(self, data: pd.DataFrame, macd_hist: pd.Series, index: int) -> Optional[SignalResult]:
        """
        Check for MACD convergence/divergence patterns.
        MACD収束/発散パターンのチェック
        """
        if len(macd_hist) < 10:
            return None

        # Get recent data
        start_idx = max(0, index - 9)
        recent_prices = data['close'].iloc[start_idx:index+1]
        recent_hist = macd_hist.iloc[start_idx:index+1]

        if len(recent_prices) < 2 or len(recent_hist) < 2:
            return None

        # Check for bullish convergence (price down, MACD up)
        price_down = recent_prices.iloc[-1] < recent_prices.iloc[0]
        hist_up = recent_hist.iloc[-1] > recent_hist.iloc[0]

        if price_down and hist_up:
            return SignalResult(
                signal_type="MACD_bullish_convergence",
                strength=0.4,
                direction=ACTION_BUY,
                description="MACD bullish convergence detected",
                confidence=0.7,
            )

        # Check for bearish divergence (price up, MACD down)
        price_up = recent_prices.iloc[-1] > recent_prices.iloc[0]
        hist_down = recent_hist.iloc[-1] < recent_hist.iloc[0]

        if price_up and hist_down:
            return SignalResult(
                signal_type="MACD_bearish_divergence",
                strength=0.4,
                direction=-ACTION_SELL,
                description="MACD bearish divergence detected",
                confidence=0.7,
            )

        return None