"""
Ichimoku Cloud Pattern Recognizer
既存の一目均衡表特徴量クラスを使用したパターン認識
時間論・波動論・水準論の統合分析
"""

from typing import Dict, Any, Optional
import pandas as pd

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.features.trend.ichimoku.ichimoku import compute_ichimoku_diff_norm, compute_ichimoku_cross
from ztb.features.trend.ichimoku.ichimoku_cloud_expansion import compute_ichimoku_cloud_expansion
from ztb.features.trend.ichimoku.ichimoku_wave_theory import compute_ichimoku_wave_theory
from ztb.features.trend.ichimoku.ichimoku_time_theory import compute_ichimoku_time_theory
from ztb.features.trend.ichimoku.ichimoku_value_measurement import compute_ichimoku_value_measurement
from ztb.features.trend.ichimoku.ichimoku_momentum_confirmation import compute_ichimoku_momentum_confirmation
from ztb.features.trend.ichimoku.ichimoku_sanyaku_kouten import compute_ichimoku_sanyaku_kouten
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    PatternRecognizer,
    SignalResult,
)


class IchimokuPatternRecognizer(PatternRecognizer):
    """
    Ichimoku Cloud pattern recognition using existing Ichimoku feature classes.
    既存の一目均衡表特徴量クラスを使用したパターン認識
    時間論・波動論・水準論の統合分析
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.tenkan_kijun_threshold = self.config.get("tenkan_kijun_threshold", 0.02)
        self.cloud_expansion_threshold = self.config.get("cloud_expansion_threshold", 0.1)
        self.wave_theory_threshold = self.config.get("wave_theory_threshold", 0.15)
        self.time_theory_threshold = self.config.get("time_theory_threshold", 0.2)
        self.value_measurement_threshold = self.config.get("value_measurement_threshold", 0.25)
        self.momentum_confirmation_threshold = self.config.get("momentum_confirmation_threshold", 0.3)
        self.sanyaku_kouten_threshold = self.config.get("sanyaku_kouten_threshold", 0.8)

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """
        Recognize Ichimoku-based patterns using integrated theories.
        一目均衡表ベースのパターン認識（時間論・波動論・水準論の統合）
        """
        if not self.validate_data(data):
            return None

        if len(data) < 52:  # Minimum periods needed for Ichimoku (26*2)
            return None

        # Calculate all Ichimoku components using existing features
        try:
            ichimoku_signals = self._calculate_ichimoku_signals(data, index)
        except Exception as e:
            return SignalResult(
                signal_type="ichimoku_error",
                strength=0.0,
                direction=ACTION_HOLD,
                description=f"Failed to calculate Ichimoku signals: {str(e)}",
                confidence=0.0,
                risk_level='high'
            )

        # Analyze integrated signals
        return self._analyze_integrated_signals(ichimoku_signals, data, index)

    def _calculate_ichimoku_signals(self, data: pd.DataFrame, index: int) -> Dict[str, float]:
        """
        Calculate all Ichimoku signals using existing feature functions.
        既存の特徴量関数を使用して全一目均衡表シグナルを計算
        """
        signals = {}

        try:
            # Time Theory - Tenkan/Kijun relationship
            signals['diff_norm'] = float(compute_ichimoku_diff_norm(data).iloc[index] if index < len(data) else compute_ichimoku_diff_norm(data).iloc[-1])
            signals['cross'] = float(compute_ichimoku_cross(data).iloc[index] if index < len(data) else compute_ichimoku_cross(data).iloc[-1])

            # Wave Theory - Cloud wave patterns and momentum
            signals['cloud_expansion'] = float(compute_ichimoku_cloud_expansion(data).iloc[index] if index < len(data) else compute_ichimoku_cloud_expansion(data).iloc[-1])
            signals['wave_theory'] = float(compute_ichimoku_wave_theory(data).iloc[index] if index < len(data) else compute_ichimoku_wave_theory(data).iloc[-1])

            # Time Theory - Temporal relationships
            signals['time_theory'] = float(compute_ichimoku_time_theory(data).iloc[index] if index < len(data) else compute_ichimoku_time_theory(data).iloc[-1])

            # Value Measurement - Price fluctuation analysis
            signals['value_measurement'] = float(compute_ichimoku_value_measurement(data).iloc[index] if index < len(data) else compute_ichimoku_value_measurement(data).iloc[-1])

            # Momentum Confirmation - Chikou span momentum
            signals['momentum_confirmation'] = float(compute_ichimoku_momentum_confirmation(data).iloc[index] if index < len(data) else compute_ichimoku_momentum_confirmation(data).iloc[-1])

            # Sanyaku Kouten - Three roles reversal pattern
            signals['sanyaku_kouten'] = float(compute_ichimoku_sanyaku_kouten(data).iloc[index] if index < len(data) else compute_ichimoku_sanyaku_kouten(data).iloc[-1])

        except Exception as e:
            print(f"Debug: Exception in Ichimoku calculation: {e}")
            import traceback
            traceback.print_exc()
            # If any calculation fails, return default values
            print(f"Warning: Ichimoku calculation failed: {str(e)}")
            signals = {
                'diff_norm': 0.0,
                'cross': 0.0,
                'cloud_expansion': 0.0,
                'wave_theory': 0.0,
                'time_theory': 0.0,
                'value_measurement': 0.0,
                'momentum_confirmation': 0.0,
                'sanyaku_kouten': 0.0
            }

        return signals

    def _analyze_integrated_signals(self, signals: Dict[str, float], data: pd.DataFrame, index: int) -> Optional[SignalResult]:
        """
        Analyze integrated Ichimoku signals using multiple theories.
        複数理論による統合一目均衡表シグナル分析
        """
        # Get current price
        current_price = float(data.iloc[index]['close']) if index >= 0 else float(data.iloc[-1]['close'])

        # Primary signal analysis based on time theory (Tenkan-Kijun)
        time_signal = self._analyze_time_theory(signals, current_price)
        if time_signal:
            return time_signal

        # Wave theory analysis for momentum confirmation
        wave_signal = self._analyze_wave_theory(signals, current_price)
        if wave_signal:
            return wave_signal

        # Value measurement for volatility assessment
        value_signal = self._analyze_value_measurement(signals, data, index)
        if value_signal:
            return value_signal

        # Sanyaku Kouten for major reversals
        reversal_signal = self._analyze_sanyaku_kouten(signals, current_price)
        if reversal_signal:
            return reversal_signal

        # Cloud expansion for trend strength
        expansion_signal = self._analyze_cloud_expansion(signals, current_price)
        if expansion_signal:
            return expansion_signal

        return None

    def _analyze_time_theory(self, signals: Dict[str, float], current_price: float) -> Optional[SignalResult]:
        """
        Analyze time theory signals (Tenkan-Kijun relationships).
        時間論シグナル分析（転換線・基準線の関係）
        """
        diff_norm = signals['diff_norm']
        cross = signals['cross']

        # Strong bullish signal: Tenkan well above Kijun with positive cross
        if diff_norm > self.tenkan_kijun_threshold and cross > 0:
            strength = min(abs(diff_norm) * 2, 0.8)
            return SignalResult(
                signal_type="ichimoku_time_bullish",
                strength=strength,
                direction=ACTION_BUY,
                description=f"Time Theory: Strong bullish (Tenkan-Kijun: {diff_norm:.3f})",
                confidence=min(strength + 0.2, 1.0),
                risk_level='low',
                validity_period=8
            )

        # Strong bearish signal: Tenkan well below Kijun with negative cross
        elif diff_norm < -self.tenkan_kijun_threshold and cross < 0:
            strength = min(abs(diff_norm) * 2, 0.8)
            return SignalResult(
                signal_type="ichimoku_time_bearish",
                strength=strength,
                direction=ACTION_SELL,
                description=f"Time Theory: Strong bearish (Tenkan-Kijun: {diff_norm:.3f})",
                confidence=min(strength + 0.2, 1.0),
                risk_level='low',
                validity_period=8
            )

        return None

    def _analyze_wave_theory(self, signals: Dict[str, float], current_price: float) -> Optional[SignalResult]:
        """
        Analyze wave theory signals (cloud wave patterns).
        波動論シグナル分析（雲の波動パターン）
        """
        wave_score = signals['wave_theory']
        momentum_score = signals['momentum_confirmation']

        combined_score = (wave_score + momentum_score) / 2

        if combined_score > self.wave_theory_threshold:
            if wave_score > momentum_score:
                # Wave momentum leading
                return SignalResult(
                    signal_type="ichimoku_wave_bullish",
                    strength=min(combined_score, 0.7),
                    direction=ACTION_BUY,
                    description=f"Wave Theory: Bullish momentum (Wave: {wave_score:.3f}, Momentum: {momentum_score:.3f})",
                    confidence=min(combined_score + 0.1, 0.9),
                    risk_level='medium',
                    validity_period=5
                )
            else:
                # Momentum confirmation
                return SignalResult(
                    signal_type="ichimoku_wave_bearish",
                    strength=min(combined_score, 0.7),
                    direction=ACTION_SELL,
                    description=f"Wave Theory: Bearish momentum (Wave: {wave_score:.3f}, Momentum: {momentum_score:.3f})",
                    confidence=min(combined_score + 0.1, 0.9),
                    risk_level='medium',
                    validity_period=5
                )

        return None

    def _analyze_value_measurement(self, signals: Dict[str, float], data: pd.DataFrame, index: int) -> Optional[SignalResult]:
        """
        Analyze value measurement signals (price fluctuation analysis).
        水準論シグナル分析（価格変動分析）
        """
        value_score = signals['value_measurement']
        time_score = signals['time_theory']

        if abs(value_score) > self.value_measurement_threshold:
            # High volatility breakout signal
            if value_score > 0 and time_score > 0:
                return SignalResult(
                    signal_type="ichimoku_value_bullish_breakout",
                    strength=min(abs(value_score), 0.6),
                    direction=ACTION_BUY,
                    description=f"Value Measurement: Bullish breakout (Value: {value_score:.3f})",
                    confidence=min(abs(value_score) + 0.2, 0.8),
                    risk_level='high',
                    validity_period=3
                )
            elif value_score < 0 and time_score < 0:
                return SignalResult(
                    signal_type="ichimoku_value_bearish_breakout",
                    strength=min(abs(value_score), 0.6),
                    direction=ACTION_SELL,
                    description=f"Value Measurement: Bearish breakout (Value: {value_score:.3f})",
                    confidence=min(abs(value_score) + 0.2, 0.8),
                    risk_level='high',
                    validity_period=3
                )

        return None

    def _analyze_sanyaku_kouten(self, signals: Dict[str, float], current_price: float) -> Optional[SignalResult]:
        """
        Analyze Sanyaku Kouten signals (three roles reversal).
        三役転換シグナル分析
        """
        sanyaku_score = signals['sanyaku_kouten']

        if sanyaku_score > self.sanyaku_kouten_threshold:
            return SignalResult(
                signal_type="ichimoku_sanyaku_reversal",
                strength=min(sanyaku_score, 0.9),
                direction=ACTION_SELL if current_price > 0 else ACTION_BUY,  # Context-dependent
                description=f"Sanyaku Kouten: Major reversal signal (Score: {sanyaku_score:.3f})",
                confidence=min(sanyaku_score, 0.95),
                risk_level='medium',
                validity_period=10
            )

        return None

    def _analyze_cloud_expansion(self, signals: Dict[str, float], current_price: float) -> Optional[SignalResult]:
        """
        Analyze cloud expansion signals (trend strength).
        雲の拡大シグナル分析（トレンド強度）
        """
        expansion_score = signals['cloud_expansion']

        if abs(expansion_score) > self.cloud_expansion_threshold:
            if expansion_score > 0:
                return SignalResult(
                    signal_type="ichimoku_cloud_expansion_bullish",
                    strength=min(expansion_score, 0.5),
                    direction=ACTION_BUY,
                    description=f"Cloud Expansion: Bullish trend strengthening (Expansion: {expansion_score:.3f})",
                    confidence=min(expansion_score + 0.3, 0.8),
                    risk_level='low',
                    validity_period=12
                )
            else:
                return SignalResult(
                    signal_type="ichimoku_cloud_expansion_bearish",
                    strength=min(abs(expansion_score), 0.5),
                    direction=ACTION_SELL,
                    description=f"Cloud Expansion: Bearish trend strengthening (Expansion: {expansion_score:.3f})",
                    confidence=min(abs(expansion_score) + 0.3, 0.8),
                    risk_level='low',
                    validity_period=12
                )

        return None