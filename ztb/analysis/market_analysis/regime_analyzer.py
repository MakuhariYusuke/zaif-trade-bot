"""
Advanced Market Regime Analyzer for SAC v445.

This module provides sophisticated market regime detection capabilities
with improved technical indicators and clearer classification logic.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from ztb.utils.metrics.trading_metrics import sharpe_ratio
from ztb.features.generators.technical.momentum.rsi import compute_rsi
from ztb.features.generators.technical.trend.adx import compute_adx
from ztb.features.generators.technical.volatility.atr import compute_atr
from ztb.features.generators.technical.momentum.roc import compute_roc
from ztb.features.generators.technical.volatility.bollinger import (
    compute_bb_middle, compute_bb_upper, compute_bb_lower
)


class MarketRegime(Enum):
    """Enumeration of market regimes with improved definitions."""
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
    """Result of regime detection with enhanced metadata."""
    regime: MarketRegime
    confidence: float
    indicators: Dict[str, float]
    metadata: Dict[str, Any]
    classification_path: List[str]  # Track which conditions led to this regime


class EnhancedRegimeAnalyzer:
    """
    Enhanced market regime analyzer with improved detection logic.

    This analyzer uses multiple technical indicators with clearer classification
    rules and better statistical foundations.
    """

    def __init__(self, detection_window: int = 50, adaptation_frequency: int = 10):
        """
        Initialize the enhanced regime analyzer.

        Args:
            detection_window: Number of data points for analysis
            adaptation_frequency: How often to adapt regime detection
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

        # Statistical baselines for adaptive thresholds
        self.volatility_history = deque(maxlen=100)
        self.trend_strength_history = deque(maxlen=100)

        # Regime classification thresholds (adaptive)
        self._update_adaptive_thresholds()

    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on historical data."""
        if len(self.volatility_history) < 20:  # Need more data for stable estimates
            # Very conservative default thresholds for initial detection
            self.volatility_percentiles = {
                'p25': 0.002, 'p50': 0.005, 'p75': 0.010, 'p90': 0.020
            }
            self.trend_thresholds = {
                'weak': 0.005, 'moderate': 0.015, 'strong': 0.030  # Higher thresholds
            }
        else:
            # Adaptive thresholds based on historical data
            vol_data = list(self.volatility_history)
            self.volatility_percentiles = {
                'p25': max(np.percentile(vol_data, 25), 0.005),  # Minimum thresholds
                'p50': max(np.percentile(vol_data, 50), 0.012),
                'p75': max(np.percentile(vol_data, 75), 0.020),
                'p90': max(np.percentile(vol_data, 90), 0.030)
            }

            trend_data = [abs(t) for t in self.trend_strength_history]
            self.trend_thresholds = {
                'weak': max(np.percentile(trend_data, 25), 0.003),
                'moderate': max(np.percentile(trend_data, 50), 0.008),
                'strong': max(np.percentile(trend_data, 75), 0.015)
            }

    def update_price_data(self, price: float, high: Optional[float] = None,
                         low: Optional[float] = None):
        """
        Update the analyzer with new price data.

        Args:
            price: Current price
            high: High price (optional)
            low: Low price (optional)
        """
        self.price_buffer.append(price)
        self.high_buffer.append(high if high is not None else price)
        self.low_buffer.append(low if low is not None else price)
        self.step_counter += 1

    def _calculate_indicators(self) -> Dict[str, float]:
        """Calculate all technical indicators using existing feature generators."""
        prices = np.array(list(self.price_buffer))
        highs = np.array(list(self.high_buffer))
        lows = np.array(list(self.low_buffer))

        if len(prices) < 14:
            return {}

        # Create DataFrame for feature generators
        df = pd.DataFrame({
            'close': prices,
            'high': highs,
            'low': lows
        })

        # Core indicators using existing feature generators
        rsi = float(compute_rsi(df).iloc[-1])
        adx = float(compute_adx(df).iloc[-1])

        # For volatility, use ATR-based approach (similar to existing ATR feature)
        atr = float(compute_atr(df).iloc[-1])
        current_price = prices[-1]
        volatility = atr / current_price if current_price > 0 else 0.002

        # For momentum, use ROC (Rate of Change) from existing features
        momentum = float(compute_roc(df, period=10).iloc[-1]) / 100.0

        # MACD using TaLib directly for individual components
        from ztb.utils.talib_wrapper import TaLibWrapper
        talib = TaLibWrapper()
        macd_line, macd_signal, macd_histogram = talib.macd(prices, 12, 26, 9)
        macd = float(macd_line[-1]) if len(macd_line) > 0 else 0.0
        signal = float(macd_signal[-1]) if len(macd_signal) > 0 else 0.0
        histogram = float(macd_histogram[-1]) if len(macd_histogram) > 0 else 0.0

        # Bollinger Bands using existing feature generators
        sma = float(compute_bb_middle(df).iloc[-1])
        bb_upper = float(compute_bb_upper(df).iloc[-1])
        bb_lower = float(compute_bb_lower(df).iloc[-1])

        # ATR using existing feature generator
        atr = float(compute_atr(df).iloc[-1])

        # Bollinger Band position
        bb_position = (prices[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

        # Update statistical baselines
        self.volatility_history.append(volatility)
        self.trend_strength_history.append(abs(momentum))

        # Update adaptive thresholds periodically
        if self.step_counter % 20 == 0:
            self._update_adaptive_thresholds()

        return {
            'rsi': rsi,
            'adx': adx,
            'volatility': volatility,
            'momentum': momentum,
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': histogram,
            'bb_position': bb_position,
            'atr': atr,
            'sma': sma,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower
        }

    def _classify_regime(self, indicators: Dict[str, float]) -> Tuple[MarketRegime, float, List[str]]:
        """
        Classify market regime with clear priority-based logic.

        Returns:
            Tuple of (regime, confidence, classification_path)
        """
        rsi = indicators.get('rsi', 50.0)
        adx = indicators.get('adx', 25.0)
        volatility = indicators.get('volatility', 0.0)
        momentum = indicators.get('momentum', 0.0)
        macd_histogram = indicators.get('macd_histogram', 0.0)
        bb_position = indicators.get('bb_position', 0.5)

        classification_path = []

        # Priority 1: Extreme volatility (highest priority, very strict criteria)
        if volatility > 0.03:  # Fixed threshold of 3% (high but achievable)
            classification_path.append("extreme_volatility_check")
            return MarketRegime.EXTREME_VOLATILITY, 0.95, classification_path

        # Priority 2: Strong trends (high ADX + strong momentum)
        trend_strength = abs(momentum)
        if adx > 30:
            classification_path.append("adx_trend_check")

            if trend_strength > self.trend_thresholds['strong']:
                classification_path.append("strong_trend")
                if momentum > 0:
                    return MarketRegime.STRONG_BULL_TREND, 0.90, classification_path
                else:
                    return MarketRegime.STRONG_BEAR_TREND, 0.90, classification_path

            elif trend_strength > self.trend_thresholds['moderate']:
                classification_path.append("moderate_trend")
                if momentum > 0:
                    return MarketRegime.MODERATE_BULL_TREND, 0.80, classification_path
                else:
                    return MarketRegime.MODERATE_BEAR_TREND, 0.80, classification_path

            elif trend_strength > self.trend_thresholds['weak']:
                classification_path.append("weak_trend")
                if momentum > 0:
                    return MarketRegime.WEAK_BULL_TREND, 0.70, classification_path
                else:
                    return MarketRegime.WEAK_BEAR_TREND, 0.70, classification_path

        # Priority 3: High volatility ranging
        if volatility > self.volatility_percentiles['p75']:
            classification_path.append("high_volatility_ranging")
            return MarketRegime.HIGH_VOLATILITY_RANGING, 0.85, classification_path

        # Priority 4: Breakout/Breakdown setups (moderate ADX, specific conditions)
        if 20 <= adx <= 30 and trend_strength < self.trend_thresholds['weak']:
            classification_path.append("breakout_setup_check")

            # Check for breakout conditions
            if macd_histogram > 0 and rsi > 55 and bb_position > 0.7:
                classification_path.append("breakout_setup")
                return MarketRegime.BREAKOUT_SETUP, 0.75, classification_path
            elif macd_histogram < 0 and rsi < 45 and bb_position < 0.3:
                classification_path.append("breakdown_setup")
                return MarketRegime.BREAKDOWN_SETUP, 0.75, classification_path

        # Priority 5: Ranging markets (low ADX)
        if adx < 20:
            classification_path.append("ranging_market_check")

            if volatility > self.volatility_percentiles['p50']:
                classification_path.append("moderate_volatility_ranging")
                return MarketRegime.MODERATE_VOLATILITY_RANGING, 0.70, classification_path
            elif volatility > self.volatility_percentiles['p25']:
                classification_path.append("low_volatility_ranging")
                return MarketRegime.LOW_VOLATILITY_RANGING, 0.65, classification_path
            else:
                classification_path.append("consolidation")
                return MarketRegime.CONSOLIDATION, 0.60, classification_path

        # Priority 6: Default consolidation
        classification_path.append("default_consolidation")
        return MarketRegime.CONSOLIDATION, 0.50, classification_path

    def detect_regime(self) -> RegimeDetectionResult:
        """Detect the current market regime with enhanced logic."""
        if len(self.price_buffer) < 14:
            return RegimeDetectionResult(
                regime=MarketRegime.CONSOLIDATION,
                confidence=0.5,
                indicators={},
                metadata={"reason": "insufficient_data"},
                classification_path=["insufficient_data"]
            )

        # Calculate technical indicators
        indicators = self._calculate_indicators()

        # Classify regime with clear logic
        regime, confidence, classification_path = self._classify_regime(indicators)

        # Update state
        self.current_regime = regime
        self.regime_confidence = confidence

        # Create result with enhanced metadata
        result = RegimeDetectionResult(
            regime=regime,
            confidence=confidence,
            indicators=indicators,
            metadata={
                "detection_timestamp": datetime.now(),
                "data_points": len(self.price_buffer),
                "adaptive_thresholds": {
                    "volatility_percentiles": self.volatility_percentiles,
                    "trend_thresholds": self.trend_thresholds
                }
            },
            classification_path=classification_path
        )

        # Store in history
        self.regime_history.append(result)

        return result

    def reset(self):
        """Reset the analyzer state."""
        self.price_buffer.clear()
        self.high_buffer.clear()
        self.low_buffer.clear()
        self.regime_history.clear()
        self.volatility_history.clear()
        self.trend_strength_history.clear()
        self.current_regime = None
        self.regime_confidence = 0.0
        self.step_counter = 0
        self._update_adaptive_thresholds()