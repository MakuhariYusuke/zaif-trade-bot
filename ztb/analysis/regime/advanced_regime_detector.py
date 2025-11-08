"""
Advanced Market Regime Detector for SAC v445.

This module provides sophisticated market regime detection capabilities
with 12 distinct market regimes for adaptive trading strategies.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class MarketRegime(Enum):
    """Enumeration of market regimes."""
    STRONG_BULL_TREND = "strong_bull_trend"
    MODERATE_BULL_TREND = "moderate_bull_trend"
    WEAK_BULL_TREND = "weak_bull_trend"
    STRONG_BEAR_TREND = "strong_bear_trend"
    MODERATE_BEAR_TREND = "moderate_bear_trend"
    WEAK_BEAR_TREND = "weak_bear_trend"
    HIGH_VOLATILITY_RANGING = "high_volatility_ranging"
    MODERATE_VOLATILITY_RANGING = "moderate_volatility_ranging"
    LOW_VOLATILITY_RANGING = "low_volatility_ranging"
    EXTREME_VOLATILITY = "extreme_volatility"
    CONSOLIDATION = "consolidation"
    BREAKOUT_SETUP = "breakout_setup"
    BREAKDOWN_SETUP = "breakdown_setup"


@dataclass
class RegimeDetectionResult:
    """Result of regime detection."""
    regime: MarketRegime
    confidence: float
    indicators: Dict[str, float]
    metadata: Dict[str, Any]


class TechnicalIndicators:
    """Static class for technical indicator calculations."""

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    @staticmethod
    def calculate_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Average Directional Index."""
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return 25.0  # Neutral ADX

        # Calculate True Range
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )

        # Calculate Directional Movement
        dm_plus = np.where(
            (highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
            np.maximum(highs[1:] - highs[:-1], 0),
            0
        )
        dm_minus = np.where(
            (lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
            np.maximum(lows[:-1] - lows[1:], 0),
            0
        )

        # Smooth the values
        atr = np.mean(tr[-period:])
        di_plus = 100 * np.mean(dm_plus[-period:]) / atr if atr > 0 else 0
        di_minus = 100 * np.mean(dm_minus[-period:]) / atr if atr > 0 else 0

        # Calculate ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus) if (di_plus + di_minus) > 0 else 0
        adx = np.mean([dx])  # Simplified, should be smoothed

        return float(adx)

    @staticmethod
    def calculate_volatility(prices: np.ndarray, period: int = 20) -> float:
        """Calculate price volatility."""
        if len(prices) < period:
            return 0.0

        returns = np.diff(np.log(prices))
        volatility = np.std(returns[-period:]) * np.sqrt(252)  # Annualized volatility

        return float(volatility)

    @staticmethod
    def calculate_momentum(prices: np.ndarray, period: int = 10) -> float:
        """Calculate momentum indicator."""
        if len(prices) < period + 1:
            return 0.0

        return float((prices[-1] - prices[-period-1]) / prices[-period-1])

    @staticmethod
    def calculate_macd(prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate MACD indicator."""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0

        # Simple EMA calculation (simplified)
        ema12 = np.mean(prices[-12:])
        ema26 = np.mean(prices[-26:])
        macd = ema12 - ema26
        signal = np.mean([macd])  # Simplified
        histogram = macd - signal

        return float(macd), float(signal), float(histogram)


class AdvancedRegimeDetector:
    """
    Advanced market regime detector with 12 distinct regimes.

    This detector uses multiple technical indicators to classify market conditions
    into specific regimes for adaptive trading strategies.
    """

    def __init__(self, detection_window: int = 50, adaptation_frequency: int = 10):
        """
        Initialize the regime detector.

        Args:
            detection_window: Number of data points to use for analysis
            adaptation_frequency: How often to adapt regime detection (in steps)
        """
        self.detection_window = detection_window
        self.adaptation_frequency = adaptation_frequency

        # Data buffers
        self.price_buffer = deque(maxlen=detection_window)
        self.high_buffer = deque(maxlen=detection_window)
        self.low_buffer = deque(maxlen=detection_window)

        # State tracking
        self.current_regime = None
        self.regime_confidence = 0.0
        self.regime_history: List[RegimeDetectionResult] = []
        self.step_counter = 0

        # Thresholds
        self.adx_trend_threshold = 25
        self.trend_strength_threshold = 0.01

    def update_price_data(self, price: float, high: Optional[float] = None,
                         low: Optional[float] = None):
        """
        Update the detector with new price data.

        Args:
            price: Current price
            high: High price (optional, defaults to price)
            low: Low price (optional, defaults to price)
        """
        self.price_buffer.append(price)
        self.high_buffer.append(high if high is not None else price)
        self.low_buffer.append(low if low is not None else price)
        self.step_counter += 1

    def _calculate_indicators(self) -> Dict[str, float]:
        """Calculate all technical indicators."""
        prices = np.array(list(self.price_buffer))
        highs = np.array(list(self.high_buffer))
        lows = np.array(list(self.low_buffer))

        if len(prices) < 14:
            return {}

        # Calculate indicators
        rsi = TechnicalIndicators.calculate_rsi(prices)
        adx = TechnicalIndicators.calculate_adx(highs, lows, prices)
        volatility = TechnicalIndicators.calculate_volatility(prices)
        momentum = TechnicalIndicators.calculate_momentum(prices)
        macd, signal, histogram = TechnicalIndicators.calculate_macd(prices)

        # Calculate percentiles for volatility
        if len(self.price_buffer) > 20:
            recent_volatility = [TechnicalIndicators.calculate_volatility(
                np.array(list(self.price_buffer)[-i-20:-i]) if i > 0 else np.array(list(self.price_buffer)[-20:])
            ) for i in range(min(10, len(self.price_buffer)//20))]
            volatility_percentile = np.percentile(recent_volatility, 70) if recent_volatility else 0.5
        else:
            volatility_percentile = 0.5

        return {
            'rsi': rsi,
            'adx': adx,
            'volatility': volatility,
            'volatility_percentile': volatility_percentile,
            'momentum': momentum,
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': histogram
        }

    def _classify_regime(self, indicators: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Classify the current market regime based on indicators."""
        rsi = indicators.get('rsi', 50.0)
        adx = indicators.get('adx', 25.0)
        trend_strength = abs(indicators.get('momentum', 0.0))
        volatility = indicators.get('volatility', 0.0)
        volatility_percentile = indicators.get('volatility_percentile', 0.5)
        macd_histogram = indicators.get('macd_histogram', 0.0)
        momentum = indicators.get('momentum', 0.0)

        # High volatility regimes
        if volatility_percentile > 0.9:
            return MarketRegime.EXTREME_VOLATILITY, 0.9
        elif volatility_percentile > 0.75:
            return MarketRegime.HIGH_VOLATILITY_RANGING, 0.8

        # Strong trend regimes (high ADX + strong momentum)
        if adx > 40 and trend_strength > self.trend_strength_threshold * 2:
            if momentum > 0.02:  # Strong upward momentum
                return MarketRegime.STRONG_BULL_TREND, 0.85
            elif momentum < -0.02:  # Strong downward momentum
                return MarketRegime.STRONG_BEAR_TREND, 0.85

        # Moderate trend regimes
        if adx > 25 and trend_strength > self.trend_strength_threshold:
            if momentum > 0.01:  # Moderate upward momentum
                return MarketRegime.MODERATE_BULL_TREND, 0.75
            elif momentum < -0.01:  # Moderate downward momentum
                return MarketRegime.MODERATE_BEAR_TREND, 0.75

        # Weak trend regimes
        if adx > 20 and trend_strength > self.trend_strength_threshold * 0.5:
            if momentum > 0.005:  # Weak upward momentum
                return MarketRegime.WEAK_BULL_TREND, 0.65
            elif momentum < -0.005:  # Weak downward momentum
                return MarketRegime.WEAK_BEAR_TREND, 0.65

        # Ranging regimes (low ADX)
        if adx < 20:
            if volatility_percentile > 0.6:
                return MarketRegime.MODERATE_VOLATILITY_RANGING, 0.7
            elif volatility_percentile > 0.4:
                return MarketRegime.LOW_VOLATILITY_RANGING, 0.6
            else:
                return MarketRegime.CONSOLIDATION, 0.5

        # Breakout/Breakdown setups
        if macd_histogram > 0.001 and rsi > 60 and adx > 25:
            return MarketRegime.BREAKOUT_SETUP, 0.7
        elif macd_histogram < -0.001 and rsi < 40 and adx > 25:
            return MarketRegime.BREAKDOWN_SETUP, 0.7

        # Default to consolidation if no clear regime
        return MarketRegime.CONSOLIDATION, 0.5

    def detect_regime(self) -> RegimeDetectionResult:
        """Detect the current market regime."""
        if len(self.price_buffer) < 14:  # Minimum data for indicators
            return RegimeDetectionResult(
                regime=MarketRegime.CONSOLIDATION,
                confidence=0.5,
                indicators={},
                metadata={"reason": "insufficient_data"}
            )

        # Calculate technical indicators
        indicators = self._calculate_indicators()

        # Classify regime
        regime, confidence = self._classify_regime(indicators)

        # Update state
        self.current_regime = regime
        self.regime_confidence = confidence

        result = RegimeDetectionResult(
            regime=regime,
            confidence=confidence,
            indicators=indicators,
            metadata={"detection_time": datetime.now().isoformat()}
        )

        self.regime_history.append(result)

        # Keep history limited
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]

        return result

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime detection."""
        if not self.regime_history:
            return {
                "total_detections": 0,
                "regime_counts": {},
                "average_confidence": 0.0,
                "most_common_regime": None
            }

        regime_counts = {}
        total_confidence = 0.0

        for result in self.regime_history:
            regime_name = result.regime.value
            regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1
            total_confidence += result.confidence

        most_common_regime = max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else None

        return {
            "total_detections": len(self.regime_history),
            "regime_counts": regime_counts,
            "average_confidence": total_confidence / len(self.regime_history),
            "most_common_regime": most_common_regime
        }

    def reset(self):
        """Reset the detector state."""
        self.price_buffer.clear()
        self.high_buffer.clear()
        self.low_buffer.clear()
        self.regime_history.clear()
        self.current_regime = None
        self.regime_confidence = 0.0
        self.step_counter = 0