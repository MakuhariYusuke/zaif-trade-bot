"""
Ichimoku Cloud Pattern Recognizer
既存の一目均衡表特徴量クラスを使用したパターン認識
時間論・波動論・水準論の統合分析
"""

import logging
from typing import TypedDict

import pandas as pd

from ztb.features.generators.technical.trend.ichimoku.ichimoku import (
    compute_ichimoku_cross,
    compute_ichimoku_diff_norm,
)
from ztb.features.generators.technical.trend.ichimoku.ichimoku_cloud_expansion import (
    compute_ichimoku_cloud_expansion,
)
from ztb.features.generators.technical.trend.ichimoku.ichimoku_momentum_confirmation import (
    compute_ichimoku_momentum_confirmation,
)
from ztb.features.generators.technical.trend.ichimoku.ichimoku_sanyaku_kouten import (
    compute_ichimoku_sanyaku_kouten,
)
from ztb.features.generators.technical.trend.ichimoku.ichimoku_time_theory import (
    compute_ichimoku_time_theory,
)
from ztb.features.generators.technical.trend.ichimoku.ichimoku_value_measurement import (
    compute_ichimoku_value_measurement,
)
from ztb.features.generators.technical.trend.ichimoku.ichimoku_wave_theory import (
    compute_ichimoku_wave_theory,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    IndicatorPatternRecognizer,
    MultiTimeframeData,
    SignalResult,
)

LOGGER = logging.getLogger(__name__)

class IchimokuSignals(TypedDict):
    """Structured Ichimoku component values."""

    diff_norm: float
    cross: float
    cloud_expansion: float
    wave_theory: float
    time_theory: float
    value_measurement: float
    momentum_confirmation: float
    sanyaku_kouten: float

DEFAULT_ICHIMOKU_SIGNALS: IchimokuSignals = {
    "diff_norm": 0.0,
    "cross": 0.0,
    "cloud_expansion": 0.0,
    "wave_theory": 0.0,
    "time_theory": 0.0,
    "value_measurement": 0.0,
    "momentum_confirmation": 0.0,
    "sanyaku_kouten": 0.0,
}

class IchimokuPatternRecognizer(IndicatorPatternRecognizer):
    """
    Ichimoku Cloud pattern recognition using existing Ichimoku feature classes.
    既存の一目均衡表特徴量クラスを使用したパターン認識
    時間論・波動論・水準論の統合分析
    """

    def __init__(self, config: dict[str, object] | None = None):
        super().__init__(config)
        self.tenkan_kijun_threshold = float(
            self.config.get("tenkan_kijun_threshold", 0.02)
        )
        self.cloud_expansion_threshold = float(
            self.config.get("cloud_expansion_threshold", 0.1)
        )
        self.wave_theory_threshold = float(self.config.get("wave_theory_threshold", 0.15))
        self.time_theory_threshold = float(self.config.get("time_theory_threshold", 0.2))
        self.value_measurement_threshold = float(
            self.config.get("value_measurement_threshold", 0.25)
        )
        self.momentum_confirmation_threshold = float(
            self.config.get("momentum_confirmation_threshold", 0.3)
        )
        self.sanyaku_kouten_threshold = float(
            self.config.get("sanyaku_kouten_threshold", 0.8)
        )

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """
        Recognize Ichimoku-based patterns using integrated theories.
        一目均衡表ベースのパターン認識（時間論・波動論・水準論の統合）
        """
        min_required_periods = 52  # Minimum periods needed for Ichimoku (26*2)
        resolved_index = self.resolve_indicator_index(
            data,
            index,
            min_required_periods=min_required_periods,
        )
        if resolved_index is None:
            return SignalResult(
                signal_type="ichimoku_insufficient_data",
                strength=0.0,
                direction=0.0,
                description="Insufficient data for Ichimoku (need 52 periods)",
                confidence=0.0,
                risk_level="low",
            )

        analysis_data, local_index = self.build_indicator_view(
            data,
            resolved_index,
            min_required_periods=min_required_periods,
            window_multiplier=12,
            min_window=260,
            max_window=1400,
        )

        lookback_data = analysis_data.iloc[max(0, local_index - 50) : local_index + 1]
        returns = lookback_data["close"].pct_change().dropna()
        current_volatility = float(returns.std()) if not returns.empty else 0.0
        avg_volatility = (
            float(returns.rolling(20).std().mean())
            if len(returns) >= 20
            else current_volatility
        )
        volatility_ratio = (
            current_volatility / avg_volatility if avg_volatility > 0 else 1.0
        )

        # Simple trend strength calculation
        sma_20 = (
            float(lookback_data["close"].rolling(20).mean().iloc[-1])
            if len(lookback_data) >= 20
            else float(lookback_data["close"].mean())
        )
        trend_strength = (
            abs((float(lookback_data["close"].iloc[-1]) - sma_20) / sma_20)
            if sma_20 != 0
            else 0.5
        )

        # Calculate all Ichimoku components using existing features
        try:
            ichimoku_signals = self._calculate_ichimoku_signals(
                analysis_data,
                local_index,
            )
        except Exception as e:
            return SignalResult(
                signal_type="ichimoku_error",
                strength=0.0,
                direction=0.0,
                description=f"Failed to calculate Ichimoku signals: {str(e)}",
                confidence=0.0,
                risk_level="high",
            )

        # Analyze integrated signals with market adaptation
        return self._analyze_integrated_signals(
            ichimoku_signals,
            analysis_data,
            local_index,
            volatility_ratio,
            trend_strength,
        )

    def _calculate_ichimoku_signals(
        self, data: pd.DataFrame, index: int
    ) -> IchimokuSignals:
        """
        Calculate all Ichimoku signals using existing feature functions.
        既存の特徴量関数を使用して全一目均衡表シグナルを計算
        """
        try:
            def _series_value(series: pd.Series) -> float:
                value = float(series.iloc[index])
                return value if pd.notna(value) else 0.0

            # Time Theory - Tenkan/Kijun relationship
            signals: IchimokuSignals = {
                "diff_norm": _series_value(compute_ichimoku_diff_norm(data)),
                "cross": _series_value(compute_ichimoku_cross(data)),
                "cloud_expansion": _series_value(compute_ichimoku_cloud_expansion(data)),
                "wave_theory": _series_value(compute_ichimoku_wave_theory(data)),
                "time_theory": _series_value(compute_ichimoku_time_theory(data)),
                "value_measurement": _series_value(
                    compute_ichimoku_value_measurement(data)
                ),
                "momentum_confirmation": _series_value(
                    compute_ichimoku_momentum_confirmation(data)
                ),
                "sanyaku_kouten": _series_value(compute_ichimoku_sanyaku_kouten(data)),
            }

            return signals

        except Exception as e:
            LOGGER.warning("Ichimoku calculation failed: %s", str(e))
            return DEFAULT_ICHIMOKU_SIGNALS.copy()

    def _analyze_integrated_signals(
        self,
        signals: IchimokuSignals,
        data: pd.DataFrame,
        index: int,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
    ) -> SignalResult | None:
        """
        Analyze integrated Ichimoku signals using multiple theories.
        複数理論による統合一目均衡表シグナル分析
        """
        # Get current price
        current_price = (
            float(data.iloc[index]["close"])
            if index >= 0
            else float(data.iloc[-1]["close"])
        )

        # Primary signal analysis based on time theory (Tenkan-Kijun)
        time_signal = self._analyze_time_theory(
            signals, current_price, volatility_ratio, trend_strength
        )
        if time_signal:
            return time_signal

        # Wave theory analysis for momentum confirmation
        wave_signal = self._analyze_wave_theory(
            signals, current_price, volatility_ratio, trend_strength
        )
        if wave_signal:
            return wave_signal

        # Value measurement for volatility assessment
        value_signal = self._analyze_value_measurement(
            signals, data, index, volatility_ratio, trend_strength
        )
        if value_signal:
            return value_signal

        # Sanyaku Kouten for major reversals
        reversal_signal = self._analyze_sanyaku_kouten(
            signals, current_price, volatility_ratio, trend_strength
        )
        if reversal_signal:
            return reversal_signal

        # Cloud expansion for trend strength
        expansion_signal = self._analyze_cloud_expansion(
            signals, current_price, volatility_ratio, trend_strength
        )
        if expansion_signal:
            return expansion_signal

        return None

    def _analyze_time_theory(
        self,
        signals: IchimokuSignals,
        current_price: float,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
    ) -> SignalResult | None:
        """
        Analyze time theory signals (Tenkan-Kijun relationships).
        時間論シグナル分析（転換線・基準線の関係）
        """
        diff_norm = signals["diff_norm"]
        cross = signals["cross"]

        # Strong bullish signal: Tenkan well above Kijun with positive cross
        if diff_norm > self.tenkan_kijun_threshold and cross > 0:
            base_strength = min(abs(diff_norm) * 2, 0.8)

            # Adaptive direction based on signal strength and market conditions
            signal_strength = abs(diff_norm) / (
                abs(diff_norm) + self.tenkan_kijun_threshold
            )
            direction_factor = signal_strength * (0.8 + trend_strength * 0.2)
            direction = min(1.0, direction_factor)

            # Adaptive strength with volatility boost
            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(1.0, base_strength + volatility_boost)

            return SignalResult(
                signal_type="ichimoku_time_bullish",
                strength=strength,
                direction=direction,
                description=f"Time Theory: Strong bullish (Tenkan-Kijun: {diff_norm:.3f})",
                confidence=min(strength + 0.2, 1.0),
                risk_level="low",
                validity_period=8,
                metadata={
                    "diff_norm": diff_norm,
                    "cross": cross,
                    "signal_strength": signal_strength,
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                },
            )

        # Strong bearish signal: Tenkan well below Kijun with negative cross
        elif diff_norm < -self.tenkan_kijun_threshold and cross < 0:
            base_strength = min(abs(diff_norm) * 2, 0.8)

            # Adaptive direction based on signal strength and market conditions
            signal_strength = abs(diff_norm) / (
                abs(diff_norm) + self.tenkan_kijun_threshold
            )
            direction_factor = -signal_strength * (0.8 + trend_strength * 0.2)
            direction = max(-1.0, direction_factor)

            # Adaptive strength with volatility boost
            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(1.0, base_strength + volatility_boost)

            return SignalResult(
                signal_type="ichimoku_time_bearish",
                strength=strength,
                direction=direction,
                description=f"Time Theory: Strong bearish (Tenkan-Kijun: {diff_norm:.3f})",
                confidence=min(strength + 0.2, 1.0),
                risk_level="low",
                validity_period=8,
                metadata={
                    "diff_norm": diff_norm,
                    "cross": cross,
                    "signal_strength": signal_strength,
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                },
            )

        return None

    def _analyze_wave_theory(
        self,
        signals: IchimokuSignals,
        current_price: float,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
    ) -> SignalResult | None:
        """
        Analyze wave theory signals (cloud wave patterns).
        波動論シグナル分析（雲の波動パターン）
        """
        wave_score = signals["wave_theory"]
        momentum_score = signals["momentum_confirmation"]

        combined_score = (wave_score + momentum_score) / 2

        if combined_score > self.wave_theory_threshold:
            if wave_score > momentum_score:
                # Wave momentum leading
                return SignalResult(
                    signal_type="ichimoku_wave_bullish",
                    strength=min(combined_score, 0.7),
                    direction=1.0,
                    description=f"Wave Theory: Bullish momentum (Wave: {wave_score:.3f}, Momentum: {momentum_score:.3f})",
                    confidence=min(combined_score + 0.1, 0.9),
                    risk_level="medium",
                    validity_period=5,
                )
            else:
                # Momentum confirmation
                return SignalResult(
                    signal_type="ichimoku_wave_bearish",
                    strength=min(combined_score, 0.7),
                    direction=-1.0,
                    description=f"Wave Theory: Bearish momentum (Wave: {wave_score:.3f}, Momentum: {momentum_score:.3f})",
                    confidence=min(combined_score + 0.1, 0.9),
                    risk_level="medium",
                    validity_period=5,
                )

        return None

    def _analyze_value_measurement(
        self,
        signals: IchimokuSignals,
        data: pd.DataFrame,
        index: int,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
    ) -> SignalResult | None:
        """
        Analyze value measurement signals (price fluctuation analysis).
        水準論シグナル分析（価格変動分析）
        """
        value_score = signals["value_measurement"]
        time_score = signals["time_theory"]

        if abs(value_score) > self.value_measurement_threshold:
            # High volatility breakout signal
            if value_score > 0 and time_score > 0:
                return SignalResult(
                    signal_type="ichimoku_value_bullish_breakout",
                    strength=min(abs(value_score), 0.6),
                    direction=1.0,
                    description=f"Value Measurement: Bullish breakout (Value: {value_score:.3f})",
                    confidence=min(abs(value_score) + 0.2, 0.8),
                    risk_level="high",
                    validity_period=3,
                )
            elif value_score < 0 and time_score < 0:
                return SignalResult(
                    signal_type="ichimoku_value_bearish_breakout",
                    strength=min(abs(value_score), 0.6),
                    direction=-1.0,
                    description=f"Value Measurement: Bearish breakout (Value: {value_score:.3f})",
                    confidence=min(abs(value_score) + 0.2, 0.8),
                    risk_level="high",
                    validity_period=3,
                )

        return None

    def _analyze_sanyaku_kouten(
        self,
        signals: IchimokuSignals,
        current_price: float,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
    ) -> SignalResult | None:
        """
        Analyze Sanyaku Kouten signals (three roles reversal).
        三役転換シグナル分析
        """
        sanyaku_score = signals["sanyaku_kouten"]

        if sanyaku_score > self.sanyaku_kouten_threshold:
            return SignalResult(
                signal_type="ichimoku_sanyaku_reversal",
                strength=min(sanyaku_score, 0.9),
                direction=-1.0 if current_price > 0 else 1.0,  # Context-dependent
                description=f"Sanyaku Kouten: Major reversal signal (Score: {sanyaku_score:.3f})",
                confidence=min(sanyaku_score, 0.95),
                risk_level="medium",
                validity_period=10,
            )

        return None

    def _analyze_cloud_expansion(
        self,
        signals: IchimokuSignals,
        current_price: float,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
    ) -> SignalResult | None:
        """
        Analyze cloud expansion signals (trend strength).
        雲の拡大シグナル分析（トレンド強度）
        """
        expansion_score = signals["cloud_expansion"]

        if abs(expansion_score) > self.cloud_expansion_threshold:
            if expansion_score > 0:
                return SignalResult(
                    signal_type="ichimoku_cloud_expansion_bullish",
                    strength=min(expansion_score, 0.5),
                    direction=1.0,
                    description=f"Cloud Expansion: Bullish trend strengthening (Expansion: {expansion_score:.3f})",
                    confidence=min(expansion_score + 0.3, 0.8),
                    risk_level="low",
                    validity_period=12,
                )
            else:
                return SignalResult(
                    signal_type="ichimoku_cloud_expansion_bearish",
                    strength=min(abs(expansion_score), 0.5),
                    direction=-1.0,
                    description=f"Cloud Expansion: Bearish trend strengthening (Expansion: {expansion_score:.3f})",
                    confidence=min(abs(expansion_score) + 0.3, 0.8),
                    risk_level="low",
                    validity_period=12,
                )

        return None
