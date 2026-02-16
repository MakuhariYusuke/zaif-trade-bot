"""
Oscillator Pattern Recognizers - CCI, Stochastic, Williams %R, MFI

This module provides pattern recognition for oscillator-based technical indicators.
"""

from typing import Optional

import pandas as pd

try:
    from ztb.features.generators.technical.momentum.williams_r import compute_williams_r
except ImportError:
    def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        return pd.Series([-50.0] * len(df), index=df.index)

try:
    from ztb.features.generators.technical.oscillator.cci import compute_cci
except ImportError:
    def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        return pd.Series([0.0] * len(df), index=df.index)

try:
    from ztb.features.generators.technical.oscillator.stochastic import compute_stochastic
except ImportError:
    def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        return pd.DataFrame({"stoch_k": [50.0] * len(df), "stoch_d": [50.0] * len(df)}, index=df.index)

try:
    from ztb.features.generators.technical.volume.mfi import compute_mfi
except ImportError:
    def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        return pd.Series([50.0] * len(df), index=df.index)

from ..types import PatternConfig
from .base import MultiTimeframeData, PatternRecognizer, RegimeAdjustment, SignalResult


def _coerce_level(value: object, default: float) -> float:
    """Safely coerce threshold level values into float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_index(data: pd.DataFrame, index: int, *, min_required_index: int = 20) -> int | None:
    """Resolve negative indices and validate minimum history requirements."""
    return PatternRecognizer.resolve_analysis_index(
        len(data), index, min_required_index=min_required_index
    )


def _build_analysis_view(
    data: pd.DataFrame,
    resolved_index: int,
    *,
    min_required_periods: int = 20,
    analysis_window: int = 240,
) -> tuple[pd.DataFrame, int]:
    """Slice a bounded historical window ending at `resolved_index`."""
    window_size = max(min_required_periods, int(analysis_window))
    start_idx = max(0, resolved_index - window_size + 1)
    view = data.iloc[start_idx : resolved_index + 1]

    if len(view) < min_required_periods:
        view = data.iloc[: resolved_index + 1]

    return view, len(view) - 1


class CCIRecognizer(PatternRecognizer):
    """
    Commodity Channel Index (CCI) pattern recognizer.
    Identifies overbought/oversold conditions and trend signals.
    """

    def __init__(self, config: Optional[PatternConfig] = None):
        super().__init__(config)
        self.pattern_type = "cci"
        self.overbought_level = self.config.get("overbought_level", 100)
        self.oversold_level = self.config.get("oversold_level", -100)
        self.analysis_window = int(self.config.get("analysis_window", 240))

        # Multi-timeframe settings
        self.enable_multi_timeframe = self.config.get("enable_multi_timeframe", True)
        self.mtf_weight = self.config.get("mtf_weight", 0.3)
        self.regime_aware = self.config.get("regime_aware", True)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[MultiTimeframeData] = None,
    ) -> Optional[SignalResult]:
        """
        Recognize CCI patterns with multi-timeframe support.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze
            multi_timeframe_data: Multi-timeframe data dictionary

        Returns:
            SignalResult with CCI analysis
        """
        resolved_index = _resolve_index(data, index, min_required_index=20)
        if resolved_index is None:  # Need sufficient data for CCI calculation
            return SignalResult(
                signal_type="cci_neutral",
                strength=0.0,
                direction=0.0,
                description="Insufficient data for CCI analysis",
                metadata={},
                validity_period=1,
                risk_level="low",
            )

        try:
            analysis_data, local_index = _build_analysis_view(
                data,
                resolved_index,
                min_required_periods=20,
                analysis_window=self.analysis_window,
            )
            cci_series = compute_cci(analysis_data)
            current_cci = float(cci_series.iloc[local_index])

            # Multi-timeframe analysis
            mtf_confidence = 1.0
            if self.enable_multi_timeframe and multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_cci_alignment(
                    current_cci, multi_timeframe_data
                )

            # Regime-aware threshold adjustment
            if self.regime_aware and multi_timeframe_data:
                adjusted_levels = self._adjust_thresholds_for_regime(multi_timeframe_data)
                overbought_level = _coerce_level(
                    adjusted_levels.get("overbought"), float(self.overbought_level)
                )
                oversold_level = _coerce_level(
                    adjusted_levels.get("oversold"), float(self.oversold_level)
                )
            else:
                overbought_level = float(self.overbought_level)
                oversold_level = float(self.oversold_level)

            # Determine signal based on CCI levels
            if current_cci >= overbought_level:
                # Overbought - potential sell signal
                base_strength = min(abs(current_cci) / 200, 1.0)  # Normalize strength
                strength = base_strength * mtf_confidence
                return SignalResult(
                    signal_type="cci_overbought_mtf" if mtf_confidence > 1.0 else "cci_overbought",
                    strength=strength,
                    direction=-1.0,  # Sell signal
                    description=f"CCI overbought at {current_cci:.2f} (MTF: {mtf_confidence:.2f})",
                    metadata={
                        "cci_value": current_cci,
                        "level": "overbought",
                        "mtf_confidence": mtf_confidence,
                        "adjusted_overbought": overbought_level
                    },
                    validity_period=5,
                    risk_level="medium",
                )
            elif current_cci <= oversold_level:
                # Oversold - potential buy signal
                base_strength = min(abs(current_cci) / 200, 1.0)  # Normalize strength
                strength = base_strength * mtf_confidence
                return SignalResult(
                    signal_type="cci_oversold_mtf" if mtf_confidence > 1.0 else "cci_oversold",
                    strength=strength,
                    direction=1.0,  # Buy signal
                    description=f"CCI oversold at {current_cci:.2f} (MTF: {mtf_confidence:.2f})",
                    metadata={
                        "cci_value": current_cci,
                        "level": "oversold",
                        "mtf_confidence": mtf_confidence,
                        "adjusted_oversold": oversold_level
                    },
                    validity_period=5,
                    risk_level="medium",
                )
            else:
                # Neutral zone
                return SignalResult(
                    signal_type="cci_neutral",
                    strength=0.0,
                    direction=0.0,
                    description=f"CCI in neutral zone at {current_cci:.2f}",
                    metadata={"cci_value": current_cci, "level": "neutral"},
                    validity_period=1,
                    risk_level="low",
                )

        except Exception as e:
            return SignalResult(
                signal_type="cci_error",
                strength=0.0,
                direction=0.0,
                description=f"CCI calculation error: {str(e)}",
                metadata={"error": str(e)},
                validity_period=1,
                risk_level="low",
            )

    def _analyze_multi_timeframe_cci_alignment(
        self,
        current_cci: float,
        multi_timeframe_data: MultiTimeframeData,
    ) -> float:
        """
        Analyze CCI alignment across multiple timeframes.

        Args:
            current_cci: Current timeframe CCI value
            multi_timeframe_data: Multi-timeframe data dictionary

        Returns:
            Confidence multiplier based on timeframe alignment
        """
        try:
            alignment_score = 1.0
            aligned_timeframes = 0

            # Check alignment with higher timeframes
            for tf_df in PatternRecognizer.iter_multi_timeframe_frames(multi_timeframe_data, min_length=21):
                try:
                    tf_cci = compute_cci(tf_df)
                    if len(tf_cci) > 0:
                        latest_tf_cci = tf_cci.iloc[-1]

                        # Check if signals align
                        base_signal = 1 if current_cci > 0 else -1
                        tf_signal = 1 if latest_tf_cci > 0 else -1

                        if base_signal == tf_signal:
                            alignment_score += 0.2
                            aligned_timeframes += 1
                        else:
                            alignment_score -= 0.1
                except Exception:
                    continue

            # Boost confidence if multiple timeframes align
            if aligned_timeframes >= 2:
                alignment_score += 0.3

            return max(0.5, min(2.0, alignment_score))

        except Exception:
            return 1.0

    def _adjust_thresholds_for_regime(
        self,
        multi_timeframe_data: Optional[MultiTimeframeData],
        pattern_type: str = "general"
    ) -> RegimeAdjustment:
        """
        Adjust CCI thresholds based on market regime.

        Args:
            multi_timeframe_data: Multi-timeframe data dictionary
            pattern_type: Type of pattern for specific analysis

        Returns:
            Dictionary with adjusted threshold levels
        """
        try:
            # Default thresholds
            adjusted_overbought = self.overbought_level
            adjusted_oversold = self.oversold_level

            # Analyze volatility from multi-timeframe data
            if multi_timeframe_data:
                volatility_indicators = []

                for tf_df in PatternRecognizer.iter_multi_timeframe_frames(multi_timeframe_data, min_length=21):
                    try:
                        # Calculate ATR as volatility proxy
                        high_low = tf_df["high"] - tf_df["low"]
                        high_close = (tf_df["high"] - tf_df["close"].shift(1)).abs()
                        low_close = (tf_df["low"] - tf_df["close"].shift(1)).abs()
                        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                        atr = tr.rolling(14).mean()

                        if len(atr) > 0:
                            current_atr = atr.iloc[-1]
                            avg_price = tf_df["close"].iloc[-1]
                            volatility = current_atr / avg_price
                            volatility_indicators.append(volatility)
                    except Exception:
                        continue

                if volatility_indicators:
                    avg_volatility = sum(volatility_indicators) / len(volatility_indicators)

                    # Adjust thresholds based on volatility
                    if avg_volatility > 0.02:  # High volatility
                        # Widen thresholds in high volatility
                        adjusted_overbought = self.overbought_level * 1.2
                        adjusted_oversold = self.oversold_level * 1.2
                    elif avg_volatility < 0.005:  # Low volatility
                        # Tighten thresholds in low volatility
                        adjusted_overbought = self.overbought_level * 0.8
                        adjusted_oversold = self.oversold_level * 0.8

            return {
                'overbought': adjusted_overbought,
                'oversold': adjusted_oversold
            }

        except Exception:
            return {
                'overbought': self.overbought_level,
                'oversold': self.oversold_level
            }


class StochasticRecognizer(PatternRecognizer):
    """
    Stochastic Oscillator pattern recognizer.
    Identifies overbought/oversold conditions using %K and %D lines.
    """

    def __init__(self, config: Optional[PatternConfig] = None):
        super().__init__(config)
        self.pattern_type = "stochastic"
        self.overbought_level = self.config.get("overbought_level", 80)
        self.oversold_level = self.config.get("oversold_level", 20)
        self.analysis_window = int(self.config.get("analysis_window", 240))

        # Multi-timeframe settings
        self.enable_multi_timeframe = self.config.get("enable_multi_timeframe", True)
        self.mtf_weight = self.config.get("mtf_weight", 0.3)
        self.regime_aware = self.config.get("regime_aware", True)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[MultiTimeframeData] = None,
    ) -> Optional[SignalResult]:
        """
        Recognize Stochastic patterns with multi-timeframe support.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze
            multi_timeframe_data: Multi-timeframe data dictionary

        Returns:
            SignalResult with Stochastic analysis
        """
        resolved_index = _resolve_index(data, index, min_required_index=20)
        if resolved_index is None:  # Need sufficient data for Stochastic calculation
            return SignalResult(
                signal_type="stochastic_neutral",
                strength=0.0,
                direction=0.0,
                description="Insufficient data for Stochastic analysis",
                metadata={},
                validity_period=1,
                risk_level="low",
            )

        try:
            analysis_data, local_index = _build_analysis_view(
                data,
                resolved_index,
                min_required_periods=20,
                analysis_window=self.analysis_window,
            )
            stoch_data = compute_stochastic(analysis_data)
            current_k = float(stoch_data["stoch_k"].iloc[local_index])
            current_d = float(stoch_data["stoch_d"].iloc[local_index])

            # Multi-timeframe analysis
            mtf_confidence = 1.0
            if self.enable_multi_timeframe and multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_stochastic_alignment(
                    current_k, current_d, multi_timeframe_data
                )

            # Regime-aware threshold adjustment
            if self.regime_aware and multi_timeframe_data:
                adjusted_levels = self._adjust_thresholds_for_regime(multi_timeframe_data)
                overbought_level = _coerce_level(
                    adjusted_levels.get("overbought"), float(self.overbought_level)
                )
                oversold_level = _coerce_level(
                    adjusted_levels.get("oversold"), float(self.oversold_level)
                )
            else:
                overbought_level = float(self.overbought_level)
                oversold_level = float(self.oversold_level)

            # Determine signal based on Stochastic levels
            if current_k >= overbought_level and current_d >= overbought_level:
                # Overbought - potential sell signal
                strength = min((current_k + current_d) / 160, 1.0)  # Normalize strength
                strength *= mtf_confidence
                return SignalResult(
                    signal_type="stochastic_overbought_mtf" if mtf_confidence > 1.0 else "stochastic_overbought",
                    strength=strength,
                    direction=-1.0,  # Sell signal
                    description=f"Stochastic overbought at %K={current_k:.2f}, %D={current_d:.2f} (MTF: {mtf_confidence:.2f})",
                    metadata={
                        "stoch_k": current_k,
                        "stoch_d": current_d,
                        "level": "overbought",
                        "mtf_confidence": mtf_confidence,
                        "adjusted_overbought": overbought_level
                    },
                    validity_period=5,
                    risk_level="medium",
                )
            elif current_k <= oversold_level and current_d <= oversold_level:
                # Oversold - potential buy signal
                strength = min((100 - current_k + 100 - current_d) / 160, 1.0)  # Normalize strength
                strength *= mtf_confidence
                return SignalResult(
                    signal_type="stochastic_oversold_mtf" if mtf_confidence > 1.0 else "stochastic_oversold",
                    strength=strength,
                    direction=1.0,  # Buy signal
                    description=f"Stochastic oversold at %K={current_k:.2f}, %D={current_d:.2f} (MTF: {mtf_confidence:.2f})",
                    metadata={
                        "stoch_k": current_k,
                        "stoch_d": current_d,
                        "level": "oversold",
                        "mtf_confidence": mtf_confidence,
                        "adjusted_oversold": oversold_level
                    },
                    validity_period=5,
                    risk_level="medium",
                )
            elif current_k > current_d and current_k < overbought_level:
                # Bullish divergence potential
                strength = min((current_k - current_d) / 20, 1.0) * 0.7  # Moderate strength
                return SignalResult(
                    signal_type="stochastic_bullish_divergence",
                    strength=strength,
                    direction=1.0,  # Buy signal
                    description=f"Stochastic bullish divergence %K={current_k:.2f}, %D={current_d:.2f}",
                    metadata={
                        "stoch_k": current_k,
                        "stoch_d": current_d,
                        "divergence": "bullish"
                    },
                    validity_period=3,
                    risk_level="medium",
                )
            elif current_k < current_d and current_k > oversold_level:
                # Bearish divergence potential
                strength = min((current_d - current_k) / 20, 1.0) * 0.7  # Moderate strength
                return SignalResult(
                    signal_type="stochastic_bearish_divergence",
                    strength=strength,
                    direction=-1.0,  # Sell signal
                    description=f"Stochastic bearish divergence %K={current_k:.2f}, %D={current_d:.2f}",
                    metadata={
                        "stoch_k": current_k,
                        "stoch_d": current_d,
                        "divergence": "bearish"
                    },
                    validity_period=3,
                    risk_level="medium",
                )
            else:
                # Neutral zone
                return SignalResult(
                    signal_type="stochastic_neutral",
                    strength=0.0,
                    direction=0.0,
                    description=f"Stochastic in neutral zone %K={current_k:.2f}, %D={current_d:.2f}",
                    metadata={"stoch_k": current_k, "stoch_d": current_d, "level": "neutral"},
                    validity_period=1,
                    risk_level="low",
                )

        except Exception as e:
            return SignalResult(
                signal_type="stochastic_error",
                strength=0.0,
                direction=0.0,
                description=f"Stochastic calculation error: {str(e)}",
                metadata={"error": str(e)},
                validity_period=1,
                risk_level="low",
            )

    def _analyze_multi_timeframe_stochastic_alignment(
        self,
        base_k: float,
        base_d: float,
        multi_timeframe_data: MultiTimeframeData,
    ) -> float:
        """
        Analyze Stochastic alignment across multiple timeframes.

        Args:
            base_k: Current timeframe stochastic %K
            base_d: Current timeframe stochastic %D
            multi_timeframe_data: Multi-timeframe data dictionary

        Returns:
            Confidence multiplier based on timeframe alignment
        """
        try:
            alignment_score = 1.0
            aligned_timeframes = 0

            # Check alignment with higher timeframes
            for tf_df in PatternRecognizer.iter_multi_timeframe_frames(multi_timeframe_data, min_length=21):
                try:
                    tf_stoch = compute_stochastic(tf_df)
                    if len(tf_stoch) > 0:
                        latest_tf_k = tf_stoch["stoch_k"].iloc[-1]
                        latest_tf_d = tf_stoch["stoch_d"].iloc[-1]

                        # Check if signals align (both overbought/oversold)
                        base_overbought = base_k >= 80 and base_d >= 80
                        base_oversold = base_k <= 20 and base_d <= 20
                        tf_overbought = latest_tf_k >= 80 and latest_tf_d >= 80
                        tf_oversold = latest_tf_k <= 20 and latest_tf_d <= 20

                        if (base_overbought and tf_overbought) or (
                            base_oversold and tf_oversold
                        ):
                            alignment_score += 0.25
                            aligned_timeframes += 1
                        elif (base_overbought and tf_oversold) or (
                            base_oversold and tf_overbought
                        ):
                            alignment_score -= 0.15
                except Exception:
                    continue

            # Boost confidence if multiple timeframes align
            if aligned_timeframes >= 2:
                alignment_score += 0.4

            return max(0.5, min(2.0, alignment_score))

        except Exception:
            return 1.0

    def _adjust_thresholds_for_regime(
        self,
        multi_timeframe_data: Optional[MultiTimeframeData],
        pattern_type: str = "general"
    ) -> RegimeAdjustment:
        """
        Adjust Stochastic thresholds based on market regime.

        Args:
            multi_timeframe_data: Multi-timeframe data dictionary
            pattern_type: Type of pattern for specific analysis

        Returns:
            Dictionary with adjusted threshold levels
        """
        try:
            # Default thresholds
            adjusted_overbought = self.overbought_level
            adjusted_oversold = self.oversold_level

            # Analyze trend strength from multi-timeframe data
            if multi_timeframe_data:
                trend_indicators = []

                for tf_df in PatternRecognizer.iter_multi_timeframe_frames(multi_timeframe_data, min_length=21):
                    try:
                        # Calculate trend strength using moving averages
                        sma_20 = tf_df["close"].rolling(20).mean()
                        sma_50 = tf_df["close"].rolling(50).mean()

                        if len(sma_20) > 0 and len(sma_50) > 0:
                            current_price = tf_df["close"].iloc[-1]
                            current_sma20 = sma_20.iloc[-1]
                            current_sma50 = sma_50.iloc[-1]

                            # Trend strength based on MA separation
                            trend_strength = abs(current_sma20 - current_sma50) / current_price
                            trend_indicators.append(trend_strength)
                    except Exception:
                        continue

                if trend_indicators:
                    avg_trend_strength = sum(trend_indicators) / len(trend_indicators)

                    # Adjust thresholds based on trend strength
                    if avg_trend_strength > 0.05:  # Strong trend
                        # Tighten thresholds in strong trends (more reliable signals)
                        adjusted_overbought = self.overbought_level * 0.9
                        adjusted_oversold = self.oversold_level * 1.1
                    elif avg_trend_strength < 0.01:  # Weak trend
                        # Widen thresholds in weak trends (less reliable signals)
                        adjusted_overbought = self.overbought_level * 1.1
                        adjusted_oversold = self.oversold_level * 0.9

            return {
                'overbought': adjusted_overbought,
                'oversold': adjusted_oversold
            }

        except Exception:
            return {
                'overbought': self.overbought_level,
                'oversold': self.oversold_level
            }


class WilliamsRRecognizer(PatternRecognizer):
    """
    Williams %R pattern recognizer.
    Identifies overbought/oversold conditions using Williams %R oscillator.
    """

    def __init__(self, config: Optional[PatternConfig] = None):
        super().__init__(config)
        self.pattern_type = "williams_r"
        self.overbought_level = self.config.get("overbought_level", -20)
        self.oversold_level = self.config.get("oversold_level", -80)
        self.analysis_window = int(self.config.get("analysis_window", 240))

        # Multi-timeframe settings
        self.enable_multi_timeframe = self.config.get("enable_multi_timeframe", True)
        self.mtf_weight = self.config.get("mtf_weight", 0.3)
        self.regime_aware = self.config.get("regime_aware", True)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[MultiTimeframeData] = None,
    ) -> Optional[SignalResult]:
        """
        Recognize Williams %R patterns with multi-timeframe support.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze
            multi_timeframe_data: Multi-timeframe data dictionary

        Returns:
            SignalResult with Williams %R analysis
        """
        resolved_index = _resolve_index(data, index, min_required_index=20)
        if resolved_index is None:  # Need sufficient data for Williams %R calculation
            return SignalResult(
                signal_type="williams_r_neutral",
                strength=0.0,
                direction=0.0,
                description="Insufficient data for Williams %R analysis",
                metadata={},
                validity_period=1,
                risk_level="low",
            )

        try:
            analysis_data, local_index = _build_analysis_view(
                data,
                resolved_index,
                min_required_periods=20,
                analysis_window=self.analysis_window,
            )
            williams_r_series = compute_williams_r(analysis_data)
            current_wr = float(williams_r_series.iloc[local_index])

            # Multi-timeframe analysis
            mtf_confidence = 1.0
            if self.enable_multi_timeframe and multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_williams_r_alignment(
                    current_wr, multi_timeframe_data
                )

            # Regime-aware threshold adjustment
            if self.regime_aware and multi_timeframe_data:
                adjusted_levels = self._adjust_thresholds_for_regime(multi_timeframe_data)
                overbought_level = _coerce_level(
                    adjusted_levels.get("overbought"), float(self.overbought_level)
                )
                oversold_level = _coerce_level(
                    adjusted_levels.get("oversold"), float(self.oversold_level)
                )
            else:
                overbought_level = float(self.overbought_level)
                oversold_level = float(self.oversold_level)

            # Determine signal based on Williams %R levels
            if current_wr >= overbought_level:
                # Overbought - potential sell signal
                strength = min(abs(current_wr) / 100, 1.0)  # Normalize strength
                strength *= mtf_confidence
                return SignalResult(
                    signal_type="williams_r_overbought_mtf" if mtf_confidence > 1.0 else "williams_r_overbought",
                    strength=strength,
                    direction=-1.0,  # Sell signal
                    description=f"Williams %R overbought at {current_wr:.2f} (MTF: {mtf_confidence:.2f})",
                    metadata={
                        "williams_r": current_wr,
                        "level": "overbought",
                        "mtf_confidence": mtf_confidence,
                        "adjusted_overbought": overbought_level
                    },
                    validity_period=5,
                    risk_level="medium",
                )
            elif current_wr <= oversold_level:
                # Oversold - potential buy signal
                strength = min((100 + current_wr) / 100, 1.0)  # Normalize strength
                strength *= mtf_confidence
                return SignalResult(
                    signal_type="williams_r_oversold_mtf" if mtf_confidence > 1.0 else "williams_r_oversold",
                    strength=strength,
                    direction=1.0,  # Buy signal
                    description=f"Williams %R oversold at {current_wr:.2f} (MTF: {mtf_confidence:.2f})",
                    metadata={
                        "williams_r": current_wr,
                        "level": "oversold",
                        "mtf_confidence": mtf_confidence,
                        "adjusted_oversold": oversold_level
                    },
                    validity_period=5,
                    risk_level="medium",
                )
            else:
                # Neutral zone
                return SignalResult(
                    signal_type="williams_r_neutral",
                    strength=0.0,
                    direction=0.0,
                    description=f"Williams %R in neutral zone at {current_wr:.2f}",
                    metadata={"williams_r": current_wr, "level": "neutral"},
                    validity_period=1,
                    risk_level="low",
                )

        except Exception as e:
            return SignalResult(
                signal_type="williams_r_error",
                strength=0.0,
                direction=0.0,
                description=f"Williams %R calculation error: {str(e)}",
                metadata={"error": str(e)},
                validity_period=1,
                risk_level="low",
            )

    def _analyze_multi_timeframe_williams_r_alignment(
        self,
        base_wr: float,
        multi_timeframe_data: MultiTimeframeData,
    ) -> float:
        """
        Analyze Williams %R alignment across multiple timeframes.

        Args:
            base_wr: Current timeframe Williams %R value
            multi_timeframe_data: Multi-timeframe data dictionary

        Returns:
            Confidence multiplier based on timeframe alignment
        """
        try:
            alignment_score = 1.0
            aligned_timeframes = 0

            # Check alignment with higher timeframes
            for tf_df in PatternRecognizer.iter_multi_timeframe_frames(multi_timeframe_data, min_length=21):
                try:
                    tf_wr = compute_williams_r(tf_df)
                    if len(tf_wr) > 0:
                        latest_tf_wr = tf_wr.iloc[-1]

                        # Check if signals align (both overbought/oversold)
                        base_overbought = base_wr >= -20
                        base_oversold = base_wr <= -80
                        tf_overbought = latest_tf_wr >= -20
                        tf_oversold = latest_tf_wr <= -80

                        if (base_overbought and tf_overbought) or (
                            base_oversold and tf_oversold
                        ):
                            alignment_score += 0.25
                            aligned_timeframes += 1
                        elif (base_overbought and tf_oversold) or (
                            base_oversold and tf_overbought
                        ):
                            alignment_score -= 0.15
                except Exception:
                    continue

            # Boost confidence if multiple timeframes align
            if aligned_timeframes >= 2:
                alignment_score += 0.4

            return max(0.5, min(2.0, alignment_score))

        except Exception:
            return 1.0

    def _adjust_thresholds_for_regime(
        self,
        multi_timeframe_data: Optional[MultiTimeframeData],
        pattern_type: str = "general"
    ) -> RegimeAdjustment:
        """
        Adjust Williams %R thresholds based on market regime.

        Args:
            multi_timeframe_data: Multi-timeframe data dictionary
            pattern_type: Type of pattern for specific analysis

        Returns:
            Dictionary with adjusted threshold levels
        """
        try:
            # Default thresholds
            adjusted_overbought = self.overbought_level
            adjusted_oversold = self.oversold_level

            # Analyze momentum from multi-timeframe data
            if multi_timeframe_data:
                momentum_indicators = []

                for tf_df in PatternRecognizer.iter_multi_timeframe_frames(multi_timeframe_data, min_length=21):
                    try:
                        # Calculate momentum using ROC (Rate of Change)
                        roc = (
                            (tf_df["close"] - tf_df["close"].shift(10))
                            / tf_df["close"].shift(10)
                            * 100
                        )

                        if len(roc) > 0:
                            current_roc = roc.iloc[-1]
                            momentum_indicators.append(abs(current_roc))
                    except Exception:
                        continue

                if momentum_indicators:
                    avg_momentum = sum(momentum_indicators) / len(momentum_indicators)

                    # Adjust thresholds based on momentum
                    if avg_momentum > 5.0:  # High momentum
                        # Tighten thresholds in high momentum (more reliable signals)
                        adjusted_overbought = self.overbought_level * 0.9
                        adjusted_oversold = self.oversold_level * 1.1
                    elif avg_momentum < 1.0:  # Low momentum
                        # Widen thresholds in low momentum (less reliable signals)
                        adjusted_overbought = self.overbought_level * 1.1
                        adjusted_oversold = self.oversold_level * 0.9

            return {
                'overbought': adjusted_overbought,
                'oversold': adjusted_oversold
            }

        except Exception:
            return {
                'overbought': self.overbought_level,
                'oversold': self.oversold_level
            }


class MFIRecognizer(PatternRecognizer):
    """
    Money Flow Index (MFI) pattern recognizer.
    Identifies overbought/oversold conditions using volume-weighted momentum.
    """

    def __init__(self, config: Optional[PatternConfig] = None):
        super().__init__(config)
        self.pattern_type = "mfi"
        self.overbought_level = self.config.get("overbought_level", 80)
        self.oversold_level = self.config.get("oversold_level", 20)
        self.analysis_window = int(self.config.get("analysis_window", 240))

        # Multi-timeframe settings
        self.enable_multi_timeframe = self.config.get("enable_multi_timeframe", True)
        self.mtf_weight = self.config.get("mtf_weight", 0.3)
        self.regime_aware = self.config.get("regime_aware", True)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[MultiTimeframeData] = None,
    ) -> Optional[SignalResult]:
        """
        Recognize MFI patterns with multi-timeframe support.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze
            multi_timeframe_data: Multi-timeframe data dictionary

        Returns:
            SignalResult with MFI analysis
        """
        resolved_index = _resolve_index(data, index, min_required_index=20)
        if resolved_index is None:  # Need sufficient data for MFI calculation
            return SignalResult(
                signal_type="mfi_neutral",
                strength=0.0,
                direction=0.0,
                description="Insufficient data for MFI analysis",
                metadata={},
                validity_period=1,
                risk_level="low",
            )

        try:
            analysis_data, local_index = _build_analysis_view(
                data,
                resolved_index,
                min_required_periods=20,
                analysis_window=self.analysis_window,
            )
            mfi_series = compute_mfi(analysis_data)
            current_mfi = float(mfi_series.iloc[local_index])

            # Multi-timeframe analysis
            mtf_confidence = 1.0
            if self.enable_multi_timeframe and multi_timeframe_data:
                mtf_confidence = self._analyze_multi_timeframe_mfi_alignment(
                    current_mfi, multi_timeframe_data
                )

            # Regime-aware threshold adjustment
            if self.regime_aware and multi_timeframe_data:
                adjusted_levels = self._adjust_thresholds_for_regime(multi_timeframe_data)
                overbought_level = _coerce_level(
                    adjusted_levels.get("overbought"), float(self.overbought_level)
                )
                oversold_level = _coerce_level(
                    adjusted_levels.get("oversold"), float(self.oversold_level)
                )
            else:
                overbought_level = float(self.overbought_level)
                oversold_level = float(self.oversold_level)

            # Determine signal based on MFI levels
            if current_mfi >= overbought_level:
                # Overbought - potential sell signal
                strength = min(current_mfi / 100, 1.0)  # Normalize strength
                strength *= mtf_confidence
                return SignalResult(
                    signal_type="mfi_overbought_mtf" if mtf_confidence > 1.0 else "mfi_overbought",
                    strength=strength,
                    direction=-1.0,  # Sell signal
                    description=f"MFI overbought at {current_mfi:.2f} (MTF: {mtf_confidence:.2f})",
                    metadata={
                        "mfi": current_mfi,
                        "level": "overbought",
                        "mtf_confidence": mtf_confidence,
                        "adjusted_overbought": overbought_level
                    },
                    validity_period=5,
                    risk_level="medium",
                )
            elif current_mfi <= oversold_level:
                # Oversold - potential buy signal
                strength = min((100 - current_mfi) / 100, 1.0)  # Normalize strength
                strength *= mtf_confidence
                return SignalResult(
                    signal_type="mfi_oversold_mtf" if mtf_confidence > 1.0 else "mfi_oversold",
                    strength=strength,
                    direction=1.0,  # Buy signal
                    description=f"MFI oversold at {current_mfi:.2f} (MTF: {mtf_confidence:.2f})",
                    metadata={
                        "mfi": current_mfi,
                        "level": "oversold",
                        "mtf_confidence": mtf_confidence,
                        "adjusted_oversold": oversold_level
                    },
                    validity_period=5,
                    risk_level="medium",
                )
            else:
                # Neutral zone
                return SignalResult(
                    signal_type="mfi_neutral",
                    strength=0.0,
                    direction=0.0,
                    description=f"MFI in neutral zone at {current_mfi:.2f}",
                    metadata={"mfi": current_mfi, "level": "neutral"},
                    validity_period=1,
                    risk_level="low",
                )

        except Exception as e:
            return SignalResult(
                signal_type="mfi_error",
                strength=0.0,
                direction=0.0,
                description=f"MFI calculation error: {str(e)}",
                metadata={"error": str(e)},
                validity_period=1,
                risk_level="low",
            )

    def _analyze_multi_timeframe_mfi_alignment(
        self,
        base_mfi: float,
        multi_timeframe_data: MultiTimeframeData,
    ) -> float:
        """
        Analyze MFI alignment across multiple timeframes.

        Args:
            base_mfi: Current timeframe MFI value
            multi_timeframe_data: Multi-timeframe data dictionary

        Returns:
            Confidence multiplier based on timeframe alignment
        """
        try:
            alignment_score = 1.0
            aligned_timeframes = 0

            # Check alignment with higher timeframes
            for tf_df in PatternRecognizer.iter_multi_timeframe_frames(multi_timeframe_data, min_length=21):
                try:
                    tf_mfi = compute_mfi(tf_df)
                    if len(tf_mfi) > 0:
                        latest_tf_mfi = tf_mfi.iloc[-1]

                        # Check if signals align (both overbought/oversold)
                        base_overbought = base_mfi >= 80
                        base_oversold = base_mfi <= 20
                        tf_overbought = latest_tf_mfi >= 80
                        tf_oversold = latest_tf_mfi <= 20

                        if (base_overbought and tf_overbought) or (
                            base_oversold and tf_oversold
                        ):
                            alignment_score += 0.25
                            aligned_timeframes += 1
                        elif (base_overbought and tf_oversold) or (
                            base_oversold and tf_overbought
                        ):
                            alignment_score -= 0.15
                except Exception:
                    continue

            # Boost confidence if multiple timeframes align
            if aligned_timeframes >= 2:
                alignment_score += 0.4

            return max(0.5, min(2.0, alignment_score))

        except Exception:
            return 1.0

    def _adjust_thresholds_for_regime(
        self,
        multi_timeframe_data: Optional[MultiTimeframeData],
        pattern_type: str = "general"
    ) -> RegimeAdjustment:
        """
        Adjust MFI thresholds based on market regime.

        Args:
            multi_timeframe_data: Multi-timeframe data dictionary
            pattern_type: Type of pattern for specific analysis

        Returns:
            Dictionary with adjusted threshold levels
        """
        try:
            # Default thresholds
            adjusted_overbought = self.overbought_level
            adjusted_oversold = self.oversold_level

            # Analyze volume patterns from multi-timeframe data
            if multi_timeframe_data:
                volume_indicators = []

                for tf_df in PatternRecognizer.iter_multi_timeframe_frames(multi_timeframe_data, min_length=21):
                    try:
                        # Calculate volume relative strength
                        avg_volume = tf_df["volume"].rolling(20).mean()
                        if len(avg_volume) > 0:
                            current_volume = tf_df["volume"].iloc[-1]
                            avg_vol = avg_volume.iloc[-1]
                            volume_ratio = current_volume / avg_vol if avg_vol > 0 else 1.0
                            volume_indicators.append(volume_ratio)
                    except Exception:
                        continue

                if volume_indicators:
                    avg_volume_ratio = sum(volume_indicators) / len(volume_indicators)

                    # Adjust thresholds based on volume
                    if avg_volume_ratio > 1.5:  # High volume
                        # Tighten thresholds in high volume (more reliable signals)
                        adjusted_overbought = self.overbought_level * 0.95
                        adjusted_oversold = self.oversold_level * 1.05
                    elif avg_volume_ratio < 0.7:  # Low volume
                        # Widen thresholds in low volume (less reliable signals)
                        adjusted_overbought = self.overbought_level * 1.05
                        adjusted_oversold = self.oversold_level * 0.95

            return {
                'overbought': adjusted_overbought,
                'oversold': adjusted_oversold
            }

        except Exception:
            return {
                'overbought': self.overbought_level,
                'oversold': self.oversold_level
            }
