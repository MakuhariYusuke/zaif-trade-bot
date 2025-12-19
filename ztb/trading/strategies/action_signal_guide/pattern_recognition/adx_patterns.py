"""
ADX (Average Directional Index) Pattern Recognizer
ADXパターン認識 - トレンド強度と方向性分析
"""

from typing import Dict, Optional, Union

import pandas as pd

try:
    from ztb.features.trend.adx import compute_adx, compute_minus_di, compute_plus_di
except ImportError:
    # Mock functions if trend module is not available
    def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        raise ImportError("ztb.features.trend.adx.compute_adx is not available. Please ensure the trend module is installed.")

    def compute_minus_di(df: pd.DataFrame, period: int = 14) -> pd.Series:
        raise ImportError("ztb.features.trend.adx.compute_minus_di is not available. Please ensure the trend module is installed.")

    def compute_plus_di(df: pd.DataFrame, period: int = 14) -> pd.Series:
        raise ImportError("ztb.features.trend.adx.compute_plus_di is not available. Please ensure the trend module is installed.")

from ztb.utils.performance_utils import timed

from .base import PatternRecognizer, SignalResult


class ADXRecognizer(PatternRecognizer):
    """
    ADX (Average Directional Index) pattern recognizer.
    ADXベースのパターン認識 - トレンド強度と方向性分析
    """

    def __init__(self, config: Optional[Dict[str, Union[int, float, bool, str, Dict]]] = None):
        super().__init__(config)
        self.pattern_type = "adx"
        self.period = self.config.get("period", 14)
        self.strong_trend_threshold = self.config.get("strong_trend_threshold", 25)
        self.weak_trend_threshold = self.config.get("weak_trend_threshold", 20)
        self.di_cross_threshold = self.config.get(
            "di_cross_threshold", 1.0
        )  # DIクロスの最小差

        # Multi-timeframe settings
        self.enable_multi_timeframe = self.config.get("enable_multi_timeframe", True)
        self.mtf_weight = self.config.get("mtf_weight", 0.3)  # Weight for MTF confirmation
        self.regime_aware = self.config.get("regime_aware", True)  # Adjust thresholds based on regime

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Union[float, int, str, Dict, None]]] = None,
    ) -> Optional[SignalResult]:
        """
        Recognize ADX patterns with multi-timeframe support.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze
            multi_timeframe_data: Multi-timeframe data for enhanced analysis

        Returns:
            SignalResult with ADX analysis
        """
        if not self.validate_data(data):
            return None

        if len(data) < self.period * 2:  # Need sufficient data for ADX calculation
            return SignalResult(
                signal_type="adx_insufficient_data",
                strength=0.0,
                direction=0.0,
                description=f"Insufficient data for ADX (need {self.period * 2} periods)",
                metadata={},
                validity_period=1,
                risk_level="low",
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
                prev_adx = adx_series.iloc[index - 1]
                prev_plus_di = plus_di_series.iloc[index - 1]
                prev_minus_di = minus_di_series.iloc[index - 1]
            else:
                prev_adx = current_adx
                prev_plus_di = current_plus_di
                prev_minus_di = current_minus_di

            # Multi-timeframe analysis for enhanced signal reliability
            mtf_confidence = self._analyze_adx_multi_timeframe_alignment(
                current_adx, current_plus_di, current_minus_di, multi_timeframe_data
            )

            # Market regime awareness - adjust thresholds based on regime
            regime_adjusted_thresholds = self._adjust_thresholds_for_regime(
                multi_timeframe_data
            )

            # 1. Strong Trend Detection with multi-timeframe confirmation
            if current_adx >= regime_adjusted_thresholds["strong_trend"]:
                # Determine trend direction
                di_difference = current_plus_di - current_minus_di

                if di_difference > regime_adjusted_thresholds["di_cross"]:
                    # Strong uptrend with multi-timeframe confirmation
                    strength = min(0.9, (current_adx / 50.0) * mtf_confidence)
                    return SignalResult(
                        signal_type="adx_strong_uptrend_mtf",
                        strength=strength,
                        direction=1.0,
                        description=f"Strong uptrend (ADX: {current_adx:.2f}, MTF: {mtf_confidence:.2f})",
                        metadata={
                            "adx": current_adx,
                            "plus_di": current_plus_di,
                            "minus_di": current_minus_di,
                            "di_difference": di_difference,
                            "trend_strength": 0.8,
                            "direction": "up",
                            "mtf_confidence": mtf_confidence,
                            "regime_adjusted": True,
                        },
                        validity_period=5,
                        risk_level="medium",
                    )

                elif di_difference < -regime_adjusted_thresholds["di_cross"]:
                    # Strong downtrend with multi-timeframe confirmation
                    strength = min(0.9, (current_adx / 50.0) * mtf_confidence)
                    return SignalResult(
                        signal_type="adx_strong_downtrend_mtf",
                        strength=strength,
                        direction=-1.0,
                        description=f"Strong downtrend (ADX: {current_adx:.2f}, MTF: {mtf_confidence:.2f})",
                        metadata={
                            "adx": current_adx,
                            "plus_di": current_plus_di,
                            "minus_di": current_minus_di,
                            "di_difference": di_difference,
                            "trend_strength": 0.8,
                            "direction": "down",
                            "mtf_confidence": mtf_confidence,
                            "regime_adjusted": True,
                        },
                        validity_period=5,
                        risk_level="medium",
                    )

                else:
                    # Strong trend but unclear direction
                    strength = min(0.7, (current_adx / 50.0) * mtf_confidence)
                    return SignalResult(
                        signal_type="adx_strong_trend_unclear_mtf",
                        strength=strength,
                        direction=0.0,
                        description=f"Strong trend unclear (ADX: {current_adx:.2f}, MTF: {mtf_confidence:.2f})",
                        metadata={
                            "adx": current_adx,
                            "plus_di": current_plus_di,
                            "minus_di": current_minus_di,
                            "di_difference": di_difference,
                            "trend_strength": 0.8,
                            "direction": "unclear",
                            "mtf_confidence": mtf_confidence,
                            "regime_adjusted": True,
                        },
                        validity_period=3,
                        risk_level="medium",
                    )

            # 2. Weak Trend / Ranging Market with regime awareness
            elif current_adx <= regime_adjusted_thresholds["weak_trend"]:
                strength = max(
                    0.1,
                    (regime_adjusted_thresholds["weak_trend"] - current_adx)
                    / regime_adjusted_thresholds["weak_trend"],
                ) * mtf_confidence
                return SignalResult(
                    signal_type="adx_weak_trend_mtf",
                    strength=strength,
                    direction=0.0,
                    description=f"Weak trend/ranging (ADX: {current_adx:.2f}, MTF: {mtf_confidence:.2f})",
                    metadata={
                        "adx": current_adx,
                        "plus_di": current_plus_di,
                        "minus_di": current_minus_di,
                        "trend_strength": 0.2,
                        "mtf_confidence": mtf_confidence,
                        "regime_adjusted": True,
                    },
                    validity_period=2,
                    risk_level="low",
                )

            # 3. DI Cross Signals with multi-timeframe confirmation
            cross_signal = self._detect_di_cross_with_mtf(
                prev_plus_di, prev_minus_di, current_plus_di, current_minus_di,
                regime_adjusted_thresholds["di_cross"], mtf_confidence
            )
            if cross_signal:
                return cross_signal

            # 4. Moderate Trend with multi-timeframe alignment
            else:
                di_difference = current_plus_di - current_minus_di

                if di_difference > regime_adjusted_thresholds["di_cross"]:
                    # Moderate uptrend with MTF confirmation
                    strength = min(0.5, (current_adx / 40.0) * mtf_confidence)
                    return SignalResult(
                        signal_type="adx_moderate_uptrend_mtf",
                        strength=strength,
                        direction=1.0,
                        description=f"Moderate uptrend (ADX: {current_adx:.2f}, MTF: {mtf_confidence:.2f})",
                        metadata={
                            "adx": current_adx,
                            "plus_di": current_plus_di,
                            "minus_di": current_minus_di,
                            "di_difference": di_difference,
                            "trend_strength": 0.5,
                            "direction": "up",
                            "mtf_confidence": mtf_confidence,
                            "regime_adjusted": True,
                        },
                        validity_period=3,
                        risk_level="medium",
                    )

                elif di_difference < -regime_adjusted_thresholds["di_cross"]:
                    # Moderate downtrend with MTF confirmation
                    strength = min(0.5, (current_adx / 40.0) * mtf_confidence)
                    return SignalResult(
                        signal_type="adx_moderate_downtrend_mtf",
                        strength=strength,
                        direction=-1.0,
                        description=f"Moderate downtrend (ADX: {current_adx:.2f}, MTF: {mtf_confidence:.2f})",
                        metadata={
                            "adx": current_adx,
                            "plus_di": current_plus_di,
                            "minus_di": current_minus_di,
                            "di_difference": di_difference,
                            "trend_strength": 0.5,
                            "direction": "down",
                            "mtf_confidence": mtf_confidence,
                            "regime_adjusted": True,
                        },
                        validity_period=3,
                        risk_level="medium",
                    )

            # 5. ADX Strengthening with regime context
            if current_adx > prev_adx and current_adx > regime_adjusted_thresholds["weak_trend"]:
                adx_change = (current_adx - prev_adx) / prev_adx
                if adx_change > 0.05:  # 5% increase
                    strength = min(0.4, adx_change * 5.0) * mtf_confidence
                    return SignalResult(
                        signal_type="adx_strengthening_mtf",
                        strength=strength,
                        direction=0.0,
                        description=f"ADX strengthening ({adx_change:.1%}, MTF: {mtf_confidence:.2f})",
                        metadata={
                            "adx": current_adx,
                            "prev_adx": prev_adx,
                            "adx_change": adx_change,
                            "trend_status": "strengthening",
                            "mtf_confidence": mtf_confidence,
                            "regime_adjusted": True,
                        },
                        validity_period=2,
                        risk_level="low",
                    )

            # Default: neutral signal with context
            return SignalResult(
                signal_type="adx_neutral_mtf",
                strength=0.0,
                direction=0.0,
                description=f"ADX neutral (ADX: {current_adx:.2f}, MTF: {mtf_confidence:.2f})",
                metadata={
                    "adx": current_adx,
                    "plus_di": current_plus_di,
                    "minus_di": current_minus_di,
                    "trend_strength": 0.0,
                    "mtf_confidence": mtf_confidence,
                    "regime_adjusted": True,
                },
                validity_period=1,
                risk_level="low",
            )

        except Exception as e:
            return SignalResult(
                signal_type="adx_error",
                strength=0.0,
                direction=0.0,
                description=f"ADX calculation error: {str(e)}",
                metadata={"error": str(e)},
                validity_period=1,
                risk_level="low",
            )

    @timed
    def _analyze_adx_multi_timeframe_alignment(
        self,
        current_adx: float,
        current_plus_di: float,
        current_minus_di: float,
        multi_timeframe_data: Optional[Dict[str, Union[float, int, str, Dict, None]]]
    ) -> float:
        """
        Analyze multi-timeframe alignment for enhanced signal confidence.

        Args:
            current_adx: Current ADX value
            current_plus_di: Current +DI value
            current_minus_di: Current -DI value
            multi_timeframe_data: Multi-timeframe data

        Returns:
            Confidence multiplier (0.5 to 1.5)
        """
        if not multi_timeframe_data:
            return 1.0  # No multi-timeframe data, use base confidence

        confidence: float = 1.0

        # Higher timeframe trend strength alignment
        higher_trend = multi_timeframe_data.get("higher_timeframe_trend", 0)
        if higher_trend > 0.7 and current_adx > 25:
            confidence *= 1.2  # Strong alignment with higher timeframe
        elif higher_trend < 0.3 and current_adx < 20:
            confidence *= 1.1  # Alignment with ranging market

        # Timeframe alignment score
        tf_alignment = multi_timeframe_data.get("timeframe_alignment", 0.5)
        confidence *= (0.8 + tf_alignment * 0.4)  # 0.8 to 1.2 range

        # Support/resistance alignment
        support_resistance = multi_timeframe_data.get("multi_timeframe_support", {})
        if support_resistance:
            # If we have support/resistance data, slightly increase confidence
            confidence *= 1.05

        return min(1.5, max(0.5, confidence))

    def _adjust_thresholds_for_regime(
        self, multi_timeframe_data: Optional[Dict[str, Union[float, int, str, Dict, None]]], pattern_type: str = "general"
    ) -> Dict[str, Union[float, int]]:
        """
        Adjust ADX thresholds based on market regime.

        Args:
            multi_timeframe_data: Multi-timeframe data containing regime info

        Returns:
            Adjusted thresholds dictionary
        """
        base_thresholds = {
            "strong_trend": self.strong_trend_threshold,
            "weak_trend": self.weak_trend_threshold,
            "di_cross": self.di_cross_threshold,
        }

        if not multi_timeframe_data:
            return base_thresholds

        # Adjust based on market regime (from regime clustering)
        regime_cluster = multi_timeframe_data.get("regime_cluster", 1)

        if regime_cluster == 0:  # Trending regime
            # Lower thresholds for trend detection in trending markets
            return {
                "strong_trend": max(20, self.strong_trend_threshold * 0.8),
                "weak_trend": max(15, self.weak_trend_threshold * 0.8),
                "di_cross": max(0.5, self.di_cross_threshold * 0.8),
            }
        elif regime_cluster == 2:  # Volatile/high-risk regime
            # Higher thresholds for more conservative signals
            return {
                "strong_trend": min(35, self.strong_trend_threshold * 1.2),
                "weak_trend": min(25, self.weak_trend_threshold * 1.2),
                "di_cross": min(2.0, self.di_cross_threshold * 1.5),
            }
        else:  # Neutral/mixed regime (cluster 1)
            return base_thresholds

    def _detect_di_cross_with_mtf(
        self,
        prev_plus_di: float,
        prev_minus_di: float,
        current_plus_di: float,
        current_minus_di: float,
        di_threshold: float,
        mtf_confidence: float
    ) -> Optional[SignalResult]:
        """
        Detect DI cross signals with multi-timeframe confirmation.

        Args:
            prev_plus_di: Previous +DI value
            prev_minus_di: Previous -DI value
            current_plus_di: Current +DI value
            current_minus_di: Current -DI value
            di_threshold: DI cross threshold
            mtf_confidence: Multi-timeframe confidence

        Returns:
            SignalResult if cross detected, None otherwise
        """
        # Bullish DI cross: +DI crosses above -DI
        if prev_plus_di <= prev_minus_di and current_plus_di > current_minus_di:
            if abs(current_plus_di - current_minus_di) >= di_threshold:
                strength = min(0.6, abs(current_plus_di - current_minus_di) / 10.0) * mtf_confidence
                return SignalResult(
                    signal_type="adx_di_cross_bullish_mtf",
                    strength=strength,
                    direction=1.0,
                    description="+DI crossed above -DI (MTF confirmed)",
                    metadata={
                        "plus_di": current_plus_di,
                        "minus_di": current_minus_di,
                        "prev_plus_di": prev_plus_di,
                        "prev_minus_di": prev_minus_di,
                        "cross_type": "bullish",
                        "mtf_confidence": mtf_confidence,
                        "regime_adjusted": True,
                    },
                    validity_period=3,
                    risk_level="medium",
                )

        # Bearish DI cross: -DI crosses above +DI
        elif prev_minus_di <= prev_plus_di and current_minus_di > current_plus_di:
            if abs(current_minus_di - current_plus_di) >= di_threshold:
                strength = min(0.6, abs(current_minus_di - current_plus_di) / 10.0) * mtf_confidence
                return SignalResult(
                    signal_type="adx_di_cross_bearish_mtf",
                    strength=strength,
                    direction=-1.0,
                    description="-DI crossed above +DI (MTF confirmed)",
                    metadata={
                        "plus_di": current_plus_di,
                        "minus_di": current_minus_di,
                        "prev_plus_di": prev_plus_di,
                        "prev_minus_di": prev_minus_di,
                        "cross_type": "bearish",
                        "mtf_confidence": mtf_confidence,
                        "regime_adjusted": True,
                    },
                    validity_period=3,
                    risk_level="medium",
                )

        return None
