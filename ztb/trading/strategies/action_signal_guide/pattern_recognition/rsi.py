"""
RSI (Relative Strength Index) Pattern Recognizer
既存のRSI特徴量クラスを使用したパターン認識
"""

from typing import Any, Dict, Optional

import pandas as pd

from ztb.features.momentum.rsi import compute_rsi
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

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """
        Recognize RSI-based patterns.
        RSIベースのパターン認識
        """
        if not self.validate_data(data):
            return None

        if len(data) < self.rsi_period + self.divergence_lookback:
            return None

        # Calculate market conditions for adaptive parameters
        lookback_data = (
            data.iloc[max(0, index - 20) : index + 1] if index >= 0 else data.iloc[-21:]
        )
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

        # Calculate RSI using existing feature class
        rsi_values = compute_rsi(data, period=self.rsi_period)

        if rsi_values.empty or rsi_values.isna().all():
            return None

        current_rsi = (
            rsi_values.iloc[index] if index < len(rsi_values) else rsi_values.iloc[-1]
        )
        previous_rsi = (
            rsi_values.iloc[index - 1]
            if index > 0 and index - 1 < len(rsi_values)
            else current_rsi
        )

        # Check for overbought/oversold signals
        if current_rsi <= self.oversold_level and previous_rsi > self.oversold_level:
            # RSI crossed below oversold level - potential buy signal
            base_strength = (self.oversold_level - current_rsi) / self.oversold_level

            # Adaptive direction based on oversold depth and market conditions
            oversold_depth = (
                self.oversold_level - current_rsi
            ) / self.oversold_level  # 0-1 scale
            direction_factor = oversold_depth * (
                0.8 + trend_strength * 0.2
            )  # Amplify in strong trends
            direction = min(1.0, direction_factor)

            # Adaptive strength with volatility boost
            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(1.0, base_strength + volatility_boost)

            return SignalResult(
                signal_type="RSI_oversold",
                strength=strength,
                direction=direction,
                description=f"RSI oversold signal (RSI: {current_rsi:.2f})",
                confidence=min(0.8, strength * 0.8),
                metadata={
                    "rsi_value": current_rsi,
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "oversold_depth": oversold_depth,
                },
            )

        elif (
            current_rsi >= self.overbought_level
            and previous_rsi < self.overbought_level
        ):
            # RSI crossed above overbought level - potential sell signal
            base_strength = (current_rsi - self.overbought_level) / (
                100 - self.overbought_level
            )

            # Adaptive direction based on overbought depth and market conditions
            overbought_depth = (current_rsi - self.overbought_level) / (
                100 - self.overbought_level
            )  # 0-1 scale
            direction_factor = -overbought_depth * (
                0.8 + trend_strength * 0.2
            )  # Amplify in strong trends
            direction = max(-1.0, direction_factor)

            # Adaptive strength with volatility boost
            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(1.0, base_strength + volatility_boost)

            return SignalResult(
                signal_type="RSI_overbought",
                strength=strength,
                direction=direction,
                description=f"RSI overbought signal (RSI: {current_rsi:.2f})",
                confidence=min(0.8, strength * 0.8),
                metadata={
                    "rsi_value": current_rsi,
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "overbought_depth": overbought_depth,
                },
            )

        # Check for divergence signals
        divergence_signal = self._check_divergence(
            data, rsi_values, index, volatility_ratio, trend_strength
        )
        if divergence_signal:
            return divergence_signal

        # Center line cross signals
        if previous_rsi <= 50 and current_rsi > 50:
            # Adaptive direction and strength for centerline cross
            base_direction = 0.6  # Moderate bullish
            trend_amplification = trend_strength * 0.4
            direction = min(1.0, base_direction + trend_amplification)

            base_strength = 0.3
            volatility_boost = min(0.1, volatility_ratio * 0.05)
            strength = min(0.6, base_strength + volatility_boost)

            return SignalResult(
                signal_type="RSI_centerline_bullish",
                strength=strength,
                direction=direction,
                description=f"RSI center line cross up (RSI: {current_rsi:.2f})",
                confidence=min(0.6, strength * 1.2),
                metadata={
                    "rsi_value": current_rsi,
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                },
            )
        elif previous_rsi >= 50 and current_rsi < 50:
            # Adaptive direction and strength for centerline cross
            base_direction = -0.6  # Moderate bearish
            trend_amplification = trend_strength * 0.4
            direction = max(-1.0, base_direction - trend_amplification)

            base_strength = 0.3
            volatility_boost = min(0.1, volatility_ratio * 0.05)
            strength = min(0.6, base_strength + volatility_boost)

            return SignalResult(
                signal_type="RSI_centerline_bearish",
                strength=strength,
                direction=direction,
                description=f"RSI center line cross down (RSI: {current_rsi:.2f})",
                confidence=min(0.6, strength * 1.2),
                metadata={
                    "rsi_value": current_rsi,
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                },
            )

        return None

    def _check_divergence(
        self,
        data: pd.DataFrame,
        rsi_values: pd.Series,
        index: int,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
    ) -> Optional[SignalResult]:
        """
        Check for RSI divergence patterns.
        RSIダイバージェンスパターンのチェック
        """
        if len(rsi_values) < self.divergence_lookback + 2:
            return None

        # Get recent data
        start_idx = max(0, index - self.divergence_lookback)
        recent_prices = data["close"].iloc[start_idx : index + 1]
        recent_rsi = rsi_values.iloc[start_idx : index + 1]

        if len(recent_prices) < 2 or len(recent_rsi) < 2:
            return None

        # Check for bullish divergence (price making lower low, RSI making higher low)
        price_trend = recent_prices.iloc[-1] < recent_prices.iloc[0]
        rsi_trend = recent_rsi.iloc[-1] > recent_rsi.iloc[0]

        if price_trend and rsi_trend:
            # Adaptive strength and direction for bullish divergence
            base_strength = 0.4
            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(0.8, base_strength + volatility_boost)

            # Direction amplified by trend strength
            base_direction = 0.7  # Strong bullish divergence signal
            trend_amplification = trend_strength * 0.3
            direction = min(1.0, base_direction + trend_amplification)

            return SignalResult(
                signal_type="RSI_bullish_divergence",
                strength=strength,
                direction=direction,
                description="RSI bullish divergence detected",
                confidence=min(0.8, strength * 1.0),
                metadata={
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "divergence_type": "bullish",
                },
            )

        # Check for bearish divergence (price making higher high, RSI making lower high)
        price_trend = recent_prices.iloc[-1] > recent_prices.iloc[0]
        rsi_trend = recent_rsi.iloc[-1] < recent_rsi.iloc[0]

        if price_trend and rsi_trend:
            # Adaptive strength and direction for bearish divergence
            base_strength = 0.4
            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(0.8, base_strength + volatility_boost)

            # Direction amplified by trend strength
            base_direction = -0.7  # Strong bearish divergence signal
            trend_amplification = trend_strength * 0.3
            direction = max(-1.0, base_direction - trend_amplification)

            return SignalResult(
                signal_type="RSI_bearish_divergence",
                strength=strength,
                direction=direction,
                description="RSI bearish divergence detected",
                confidence=min(0.8, strength * 1.0),
                metadata={
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "divergence_type": "bearish",
                },
            )

        return None
