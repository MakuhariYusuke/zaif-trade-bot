"""
ATR (Average True Range) Pattern Recognizer
既存のATR特徴量クラスを使用したパターン認識
"""

from typing import Any, Dict, Optional

import pandas as pd

from ztb.features.generators.technical.volatility.atr import compute_atr
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    MultiTimeframeData,
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

        # Multi-timeframe settings
        self.enable_multi_timeframe = self.config.get("enable_multi_timeframe", True)
        self.mtf_weight = self.config.get("mtf_weight", 0.3)
        self.regime_aware = self.config.get("regime_aware", True)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """
        Recognize ATR-based patterns with multi-timeframe support.
        ATRベースのパターン認識（複数時間軸対応）
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

        # Multi-timeframe analysis for enhanced signal reliability
        mtf_confidence = self._analyze_multi_timeframe_volatility(
            current_atr, avg_atr, multi_timeframe_data
        )

        # Market regime awareness - adjust thresholds based on regime
        regime_adjusted_thresholds = self._adjust_thresholds_for_regime(
            multi_timeframe_data
        )

        # Volatility breakout signals with multi-timeframe confirmation
        volatility_ratio = current_atr / avg_atr
        if current_atr > avg_atr * regime_adjusted_thresholds["volatility_threshold"]:
            # High volatility - potential breakout
            breakout_signal = self._analyze_breakout_mtf(
                data,
                current_atr,
                avg_atr,
                index,
                mtf_confidence,
                regime_adjusted_thresholds,
            )
            if breakout_signal:
                return breakout_signal

        # Trend strength analysis using ATR with multi-timeframe
        trend_signal = self._analyze_trend_strength_mtf(
            data, atr_values, index, mtf_confidence, regime_adjusted_thresholds
        )
        if trend_signal:
            return trend_signal

        # Low volatility consolidation with regime context
        if (
            current_atr
            < avg_atr * regime_adjusted_thresholds["low_volatility_threshold"]
        ):
            strength = (
                max(
                    0.1,
                    (
                        avg_atr * regime_adjusted_thresholds["low_volatility_threshold"]
                        - current_atr
                    )
                    / avg_atr,
                )
                * mtf_confidence
            )
            return SignalResult(
                signal_type="ATR_low_volatility_mtf",
                strength=strength,
                direction=0.0,
                description=f"ATR low volatility consolidation (ATR: {current_atr:.6f}, MTF: {mtf_confidence:.2f})",
                metadata={
                    "atr": current_atr,
                    "avg_atr": avg_atr,
                    "volatility_ratio": volatility_ratio,
                    "mtf_confidence": mtf_confidence,
                    "regime_adjusted": True,
                },
                validity_period=2,
                risk_level="low",
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

    def _analyze_multi_timeframe_volatility(
        self,
        current_atr: float,
        avg_atr: float,
        multi_timeframe_data: Optional[Dict[str, Any]],
    ) -> float:
        """
        Analyze multi-timeframe volatility alignment for enhanced signal confidence.

        Args:
            current_atr: Current ATR value
            avg_atr: Average ATR value
            multi_timeframe_data: Multi-timeframe data

        Returns:
            Confidence multiplier (0.5 to 1.5)
        """
        if not multi_timeframe_data:
            return 1.0  # No multi-timeframe data, use base confidence

        confidence: float = 1.0

        # Higher timeframe volatility alignment
        higher_volatility = multi_timeframe_data.get("higher_timeframe_volatility", 0)
        volatility_ratio = current_atr / avg_atr

        if higher_volatility > 1.2 and volatility_ratio > 1.1:
            confidence *= 1.2  # Strong volatility alignment with higher timeframe
        elif higher_volatility < 0.8 and volatility_ratio < 0.9:
            confidence *= 1.1  # Alignment with low volatility market

        # Timeframe alignment score
        tf_alignment = multi_timeframe_data.get("timeframe_alignment", 0.5)
        confidence *= 0.8 + tf_alignment * 0.4  # 0.8 to 1.2 range

        # Market regime consideration
        regime_cluster = multi_timeframe_data.get("regime_cluster", 1)
        if regime_cluster == 2:  # High volatility regime
            confidence *= 1.1  # Increase confidence in volatile markets

        return min(1.5, max(0.5, confidence))

    def _adjust_thresholds_for_regime(
        self,
        multi_timeframe_data: Optional[MultiTimeframeData],
        pattern_type: str = "general",
    ) -> Dict[str, Any]:
        """
        Adjust ATR thresholds based on market regime.

        Args:
            multi_timeframe_data: Multi-timeframe data containing regime info

        Returns:
            Adjusted thresholds dictionary
        """
        base_thresholds = {
            "volatility_threshold": self.volatility_threshold,
            "low_volatility_threshold": 0.8,
        }

        if not multi_timeframe_data:
            return base_thresholds

        # Adjust based on market regime (from regime clustering)
        regime_cluster = multi_timeframe_data.get("regime_cluster", 1)

        if regime_cluster == 0:  # Trending regime
            # Lower volatility thresholds for trend-following in trending markets
            return {
                "volatility_threshold": max(0.8, self.volatility_threshold * 0.9),
                "low_volatility_threshold": 0.75,
            }
        elif regime_cluster == 2:  # Volatile/high-risk regime
            # Higher thresholds for more conservative signals
            return {
                "volatility_threshold": min(1.5, self.volatility_threshold * 1.3),
                "low_volatility_threshold": 0.9,
            }
        else:  # Neutral/mixed regime (cluster 1)
            return base_thresholds

    def _analyze_breakout_mtf(
        self,
        data: pd.DataFrame,
        current_atr: float,
        avg_atr: float,
        index: int,
        mtf_confidence: float,
        regime_adjusted_thresholds: Dict[str, float],
    ) -> Optional[SignalResult]:
        """
        Analyze potential breakout during high volatility with multi-timeframe confirmation.
        高ボラティリティ時のブレイクアウト分析（複数時間軸対応）
        """
        start_idx = max(0, index - 4)
        recent_prices = data["close"].iloc[start_idx : index + 1]

        if len(recent_prices) < 2:
            return None

        price_change = (
            recent_prices.iloc[-1] - recent_prices.iloc[0]
        ) / recent_prices.iloc[0]

        volatility_ratio = current_atr / avg_atr
        strength = (
            min(
                volatility_ratio / regime_adjusted_thresholds["volatility_threshold"],
                1.0,
            )
            * mtf_confidence
        )

        if abs(price_change) > 0.005:  # 0.5% price movement
            if price_change > 0:
                return SignalResult(
                    signal_type="ATR_bullish_breakout_mtf",
                    strength=strength,
                    direction=1.0,
                    description=f"ATR bullish breakout MTF (Vol: {volatility_ratio:.2f}, Price: +{price_change:.2%}, MTF: {mtf_confidence:.2f})",
                    metadata={
                        "atr": current_atr,
                        "avg_atr": avg_atr,
                        "volatility_ratio": volatility_ratio,
                        "price_change": price_change,
                        "mtf_confidence": mtf_confidence,
                        "regime_adjusted": True,
                    },
                    validity_period=3,
                    risk_level="medium",
                )
            else:
                return SignalResult(
                    signal_type="ATR_bearish_breakout_mtf",
                    strength=strength,
                    direction=-1.0,
                    description=f"ATR bearish breakout MTF (Vol: {volatility_ratio:.2f}, Price: {price_change:.2%}, MTF: {mtf_confidence:.2f})",
                    metadata={
                        "atr": current_atr,
                        "avg_atr": avg_atr,
                        "volatility_ratio": volatility_ratio,
                        "price_change": price_change,
                        "mtf_confidence": mtf_confidence,
                        "regime_adjusted": True,
                    },
                    validity_period=3,
                    risk_level="medium",
                )

        return SignalResult(
            signal_type="ATR_high_volatility_mtf",
            strength=strength * 0.5,
            direction=0.0,
            description=f"ATR high volatility MTF, awaiting direction (Vol: {volatility_ratio:.2f}, MTF: {mtf_confidence:.2f})",
            metadata={
                "atr": current_atr,
                "avg_atr": avg_atr,
                "volatility_ratio": volatility_ratio,
                "mtf_confidence": mtf_confidence,
                "regime_adjusted": True,
            },
            validity_period=2,
            risk_level="low",
        )

    def _analyze_trend_strength_mtf(
        self,
        data: pd.DataFrame,
        atr_values: pd.Series,
        index: int,
        mtf_confidence: float,
        regime_adjusted_thresholds: Dict[str, float],
    ) -> Optional[SignalResult]:
        """
        Analyze trend strength using ATR changes with multi-timeframe support.
        ATR変化によるトレンド強度分析（複数時間軸対応）
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

        # Strong trend with increasing ATR (healthy trend) with MTF confirmation
        if abs(price_trend) > 0.01 and atr_trend > 0.05:
            if price_trend > 0:
                strength = min(abs(price_trend) * 10, 0.6) * mtf_confidence
                return SignalResult(
                    signal_type="ATR_strong_bullish_trend_mtf",
                    strength=strength,
                    direction=1.0,
                    description=f"ATR strong bullish trend MTF (Price: +{price_trend:.2%}, ATR: +{atr_trend:.2%}, MTF: {mtf_confidence:.2f})",
                    metadata={
                        "atr_trend": atr_trend,
                        "price_trend": price_trend,
                        "mtf_confidence": mtf_confidence,
                        "regime_adjusted": True,
                    },
                    validity_period=4,
                    risk_level="medium",
                )
            else:
                strength = min(abs(price_trend) * 10, 0.6) * mtf_confidence
                return SignalResult(
                    signal_type="ATR_strong_bearish_trend_mtf",
                    strength=strength,
                    direction=-1.0,
                    description=f"ATR strong bearish trend MTF (Price: {price_trend:.2%}, ATR: +{atr_trend:.2%}, MTF: {mtf_confidence:.2f})",
                    metadata={
                        "atr_trend": atr_trend,
                        "price_trend": price_trend,
                        "mtf_confidence": mtf_confidence,
                        "regime_adjusted": True,
                    },
                    validity_period=4,
                    risk_level="medium",
                )

        # Weak trend with decreasing ATR (potential reversal) with MTF context
        elif abs(price_trend) < 0.005 and atr_trend < -0.05:
            strength = 0.3 * mtf_confidence
            return SignalResult(
                signal_type="ATR_weakening_trend_mtf",
                strength=strength,
                direction=0.0,
                description=f"ATR weakening trend MTF (Price: {price_trend:.2%}, ATR: {atr_trend:.2%}, MTF: {mtf_confidence:.2f})",
                metadata={
                    "atr_trend": atr_trend,
                    "price_trend": price_trend,
                    "mtf_confidence": mtf_confidence,
                    "regime_adjusted": True,
                },
                validity_period=3,
                risk_level="low",
            )

        return None
