"""
Market Regime Analysis Components for Action Signal Guide.

This module provides market regime detection and analysis capabilities
to enhance signal generation and validation.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
import logging

from ztb.analysis.market_regime_types import MarketRegime

logger = logging.getLogger(__name__)


from ztb.trading.environment.components.interfaces import IMarketRegimeDetector


class MarketRegimeDetector(IMarketRegimeDetector):
    """
    Detects current market regime using multiple indicators.
    """

    def __init__(self):
        """Initialize market regime detector."""
        self.regime_history = []
        self.transition_threshold = 0.6
        self.stability_window = 20  # periods to consider regime stable

        # Regime detection parameters
        self.trend_threshold = 0.02  # 2% trend strength
        self.volatility_threshold = 0.03  # 3% volatility
        self.range_threshold = 0.015  # 1.5% range bound

    def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime from price data.

        Args:
            market_data: OHLCV market data

        Returns:
            Detected market regime
        """
        if len(market_data) < 50:
            return MarketRegime.RANGING  # Default for insufficient data

        # Calculate regime indicators
        trend_strength = self._calculate_trend_strength(market_data)
        volatility = self._calculate_volatility(market_data)
        range_bound = self._calculate_range_bound(market_data)
        momentum = self._calculate_momentum(market_data)

        # Determine regime based on indicators
        regime = self._classify_regime(
            trend_strength, volatility, range_bound, momentum
        )

        # Store regime for stability analysis
        self.regime_history.append({
            "timestamp": market_data.index[-1] if hasattr(market_data.index, '__getitem__') else pd.Timestamp.now(),
            "regime": regime,
            "indicators": {
                "trend_strength": trend_strength,
                "volatility": volatility,
                "range_bound": range_bound,
                "momentum": momentum,
            }
        })

        # Keep only recent history
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-50:]

        return regime

    def detect_regime(self, current_price: float, step: int) -> str:
        """
        Detect current market regime (IMarketRegimeDetector interface).

        Args:
            current_price: Current market price
            step: Current step number

        Returns:
            Market regime: 'bull', 'bear', 'sideways', 'volatile'
        """
        # For interface compatibility, return a simple regime based on price
        # In a real implementation, this would use historical data
        if len(self.regime_history) > 0:
            return self.regime_history[-1]["regime"].value
        else:
            return "sideways"

    def get_regime_stability(self) -> float:
        """
        Get current regime stability score.

        Returns:
            Stability score (0-1, higher = more stable)
        """
        if len(self.regime_history) < self.stability_window:
            return 0.5

        recent_regimes = [r["regime"] for r in self.regime_history[-self.stability_window:]]
        most_common_regime = max(set(recent_regimes), key=recent_regimes.count)
        stability_ratio = recent_regimes.count(most_common_regime) / len(recent_regimes)

        return stability_ratio

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate trend strength using linear regression slope.

        Args:
            data: Market data

        Returns:
            Trend strength (-1 to 1, positive = bullish)
        """
        prices = data['close'].values
        if len(prices) < 20:
            return 0.0

        # Use recent 20 periods for trend calculation
        recent_prices = prices[-20:]
        x = np.arange(len(recent_prices))

        # Linear regression
        slope, intercept = np.polyfit(x, recent_prices, 1)

        # Normalize slope by average price
        avg_price = np.mean(recent_prices)
        normalized_slope = slope / avg_price if avg_price != 0 else 0

        return normalized_slope

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        Calculate price volatility.

        Args:
            data: Market data

        Returns:
            Volatility (0-1, higher = more volatile)
        """
        returns = data['close'].pct_change().dropna()
        if len(returns) < 10:
            return 0.0

        # Use recent 20 periods
        recent_returns = returns.tail(20)
        volatility = recent_returns.std()

        return min(1.0, volatility * 10)  # Scale for 0-1 range

    def _calculate_range_bound(self, data: pd.DataFrame) -> float:
        """
        Calculate how range-bound the market is.

        Args:
            data: Market data

        Returns:
            Range bound score (0-1, higher = more range-bound)
        """
        if len(data) < 20:
            return 0.0

        recent_data = data.tail(20)
        high_max = recent_data['high'].max()
        low_min = recent_data['low'].min()
        current_price = recent_data['close'].iloc[-1]

        # Calculate range as percentage of current price
        price_range = (high_max - low_min) / current_price

        # Range bound score (inverse of range size, normalized)
        range_bound = max(0.0, 1.0 - (price_range / 0.1))  # 10% range = score of 0

        return range_bound

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """
        Calculate price momentum.

        Args:
            data: Market data

        Returns:
            Momentum (-1 to 1)
        """
        if len(data) < 10:
            return 0.0

        # RSI-style momentum calculation
        returns = data['close'].pct_change().dropna()
        recent_returns = returns.tail(10)

        positive_returns = recent_returns[recent_returns > 0]
        negative_returns = recent_returns[recent_returns < 0]

        if len(negative_returns) == 0:
            return 1.0
        elif len(positive_returns) == 0:
            return -1.0

        avg_gain = positive_returns.mean()
        avg_loss = abs(negative_returns.mean())

        if avg_loss == 0:
            return 1.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Convert RSI to -1 to 1 scale
        momentum = (rsi - 50) / 50

        return momentum

    def _classify_regime(
        self,
        trend_strength: float,
        volatility: float,
        range_bound: float,
        momentum: float
    ) -> MarketRegime:
        """
        Classify market regime based on indicators.

        Args:
            trend_strength: Calculated trend strength
            volatility: Calculated volatility
            range_bound: Calculated range bound score
            momentum: Calculated momentum

        Returns:
            Classified market regime
        """
        # High volatility regime
        if volatility > self.volatility_threshold:
            return MarketRegime.HIGH_VOLATILITY

        # Low volatility regime
        if volatility < self.volatility_threshold * 0.3:
            return MarketRegime.LOW_VOLATILITY

        # Trending regimes
        if abs(trend_strength) > self.trend_threshold:
            if trend_strength > 0 and momentum > 0.2:
                return MarketRegime.TRENDING_BULLISH
            elif trend_strength < 0 and momentum < -0.2:
                return MarketRegime.TRENDING_BEARISH

        # Ranging regime (default)
        if range_bound > 0.6:
            return MarketRegime.RANGING

        # Default to ranging if no clear regime
        return MarketRegime.RANGING


class RegimeAdaptiveSignalProcessor:
    """
    Processes signals adaptively based on detected market regime.
    """

    def __init__(self):
        """Initialize regime adaptive signal processor."""
        self.regime_detector = MarketRegimeDetector()
        self.regime_performance = {regime: [] for regime in MarketRegime}
        self.adaptation_factors = {}

    def process_signals_for_regime(
        self,
        signals: List["ActionSignal"],
        market_data: pd.DataFrame
    ) -> List["ActionSignal"]:
        """
        Process signals based on current market regime.

        Args:
            signals: Input signals
            market_data: Current market data

        Returns:
            Processed signals adapted for regime
        """
        # Detect current regime
        current_regime = self.regime_detector.detect_regime(market_data)
        regime_stability = self.regime_detector.get_regime_stability()

        processed_signals = []

        for signal in signals:
            # Apply regime-specific processing
            processed_signal = self._adapt_signal_for_regime(
                signal, current_regime, regime_stability, market_data
            )

            if processed_signal:
                processed_signals.append(processed_signal)

        return processed_signals

    def _adapt_signal_for_regime(
        self,
        signal: "ActionSignal",
        regime: MarketRegime,
        stability: float,
        market_data: pd.DataFrame
    ) -> Optional["ActionSignal"]:
        """
        Adapt individual signal for specific market regime.

        Args:
            signal: Signal to adapt
            regime: Current market regime
            stability: Regime stability score
            market_data: Market data

        Returns:
            Adapted signal or None if filtered out
        """
        pattern_type = getattr(signal, 'pattern_type', 'unknown')
        original_confidence = getattr(signal, 'confidence', 0.5)

        # Regime-specific pattern preferences and adjustments
        regime_config = self._get_regime_config(regime)

        # Check if pattern is suitable for regime
        if pattern_type not in regime_config["preferred_patterns"]:
            # Reduce confidence for non-preferred patterns
            signal.confidence = original_confidence * regime_config["penalty_factor"]
        else:
            # Boost confidence for preferred patterns
            signal.confidence = min(1.0, original_confidence * regime_config["boost_factor"])

        # Apply stability adjustment
        stability_adjustment = 1.0 + (stability - 0.5) * 0.2  # ±10% based on stability
        signal.confidence = min(1.0, signal.confidence * stability_adjustment)

        # Apply market condition filters
        if not self._passes_regime_filter(signal, regime, market_data):
            return None

        # Add regime metadata
        if not hasattr(signal, 'regime_analysis'):
            signal.regime_analysis = {}

        signal.regime_analysis.update({
            "detected_regime": regime.value,
            "regime_stability": stability,
            "confidence_adjustment": signal.confidence / original_confidence,
            "regime_suitability": pattern_type in regime_config["preferred_patterns"],
        })

        return signal

    def _get_regime_config(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get configuration for specific regime.

        Args:
            regime: Market regime

        Returns:
            Regime configuration
        """
        configs = {
            MarketRegime.TRENDING_BULLISH: {
                "preferred_patterns": ["fibonacci", "harmonic", "gann", "dow_theory", "trend"],
                "boost_factor": 1.3,
                "penalty_factor": 0.7,
            },
            MarketRegime.TRENDING_BEARISH: {
                "preferred_patterns": ["fibonacci", "harmonic", "gann", "dow_theory", "trend"],
                "boost_factor": 1.3,
                "penalty_factor": 0.7,
            },
            MarketRegime.RANGING: {
                "preferred_patterns": ["bollinger", "oscillator", "volume", "candlestick", "support_resistance"],
                "boost_factor": 1.2,
                "penalty_factor": 0.8,
            },
            MarketRegime.HIGH_VOLATILITY: {
                "preferred_patterns": ["volume", "gann", "granville", "breakout"],
                "boost_factor": 1.4,
                "penalty_factor": 0.6,
            },
            MarketRegime.LOW_VOLATILITY: {
                "preferred_patterns": ["fibonacci", "harmonic", "candlestick", "pattern"],
                "boost_factor": 1.1,
                "penalty_factor": 0.9,
            },
        }

        return configs.get(regime, configs[MarketRegime.RANGING])

    def _passes_regime_filter(
        self,
        signal: "ActionSignal",
        regime: MarketRegime,
        market_data: pd.DataFrame
    ) -> bool:
        """
        Check if signal passes regime-specific filters.

        Args:
            signal: Signal to check
            regime: Current regime
            market_data: Market data

        Returns:
            True if signal passes filter
        """
        confidence = getattr(signal, 'confidence', 0.5)

        # Minimum confidence thresholds by regime
        min_confidence = {
            MarketRegime.TRENDING_BULLISH: 0.4,
            MarketRegime.TRENDING_BEARISH: 0.4,
            MarketRegime.RANGING: 0.5,
            MarketRegime.HIGH_VOLATILITY: 0.6,
            MarketRegime.LOW_VOLATILITY: 0.3,
        }

        if confidence < min_confidence.get(regime, 0.4):
            return False

        # Additional regime-specific filters
        if regime == MarketRegime.HIGH_VOLATILITY:
            # In high volatility, require stronger signals
            if confidence < 0.5:
                return False
        elif regime == MarketRegime.LOW_VOLATILITY:
            # In low volatility, be more permissive
            pass  # No additional filter

        return True

    def update_regime_performance(
        self,
        regime: MarketRegime,
        signal_performance: float
    ) -> None:
        """
        Update performance tracking for regime adaptation.

        Args:
            regime: Market regime
            signal_performance: Performance of signal in this regime
        """
        self.regime_performance[regime].append(signal_performance)

        # Keep only recent performance data
        if len(self.regime_performance[regime]) > 100:
            self.regime_performance[regime] = self.regime_performance[regime][-50:]

        # Update adaptation factors based on performance
        self._update_adaptation_factors()

    def _update_adaptation_factors(self) -> None:
        """Update adaptation factors based on performance history."""
        for regime in MarketRegime:
            performances = self.regime_performance[regime]
            if len(performances) >= 10:
                avg_performance = sum(performances[-20:]) / len(performances[-20:])
                self.adaptation_factors[regime] = avg_performance
            else:
                self.adaptation_factors[regime] = 0.5  # Neutral


class MarketConditionAnalyzer:
    """
    Analyzes various market conditions for signal enhancement.
    """

    def __init__(self):
        """Initialize market condition analyzer."""
        self.condition_indicators = {}
        self.analysis_history = []

    def analyze_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze current market conditions.

        Args:
            market_data: Market data to analyze

        Returns:
            Market condition analysis
        """
        if len(market_data) < 20:
            return self._get_default_conditions()

        analysis = {
            "trend": self._analyze_trend(market_data),
            "volatility": self._analyze_volatility(market_data),
            "momentum": self._analyze_momentum(market_data),
            "volume": self._analyze_volume(market_data),
            "support_resistance": self._analyze_support_resistance(market_data),
            "timestamp": market_data.index[-1] if hasattr(market_data.index, '__getitem__') else pd.Timestamp.now(),
        }

        # Store analysis for historical context
        self.analysis_history.append(analysis)
        if len(self.analysis_history) > 50:
            self.analysis_history = self.analysis_history[-25:]

        return analysis

    def _get_default_conditions(self) -> Dict[str, Any]:
        """Get default market conditions when data is insufficient."""
        return {
            "trend": {"direction": "neutral", "strength": 0.0},
            "volatility": {"level": "medium", "value": 0.02},
            "momentum": {"value": 0.0, "strength": "neutral"},
            "volume": {"trend": "neutral", "confirmation": False},
            "support_resistance": {"nearby_levels": []},
            "timestamp": pd.Timestamp.now(),
        }

    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trend."""
        recent_data = data.tail(20)
        prices = recent_data['close'].values

        # Linear regression for trend
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)

        # Determine trend direction and strength
        avg_price = np.mean(prices)
        trend_strength = abs(slope) / avg_price if avg_price != 0 else 0

        if slope > avg_price * 0.001:  # Uptrend
            direction = "bullish"
        elif slope < -avg_price * 0.001:  # Downtrend
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "direction": direction,
            "strength": min(1.0, trend_strength * 100),  # Scale to 0-1
            "slope": slope,
        }

    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price volatility."""
        returns = data['close'].pct_change().dropna().tail(20)
        volatility = returns.std()

        if volatility > 0.04:
            level = "high"
        elif volatility < 0.01:
            level = "low"
        else:
            level = "medium"

        return {
            "level": level,
            "value": volatility,
        }

    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price momentum."""
        if len(data) < 14:
            return {"value": 0.0, "strength": "neutral"}

        # Simple momentum calculation
        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-14]
        momentum_value = (current_price - past_price) / past_price

        if momentum_value > 0.02:
            strength = "strong_bullish"
        elif momentum_value > 0.005:
            strength = "bullish"
        elif momentum_value < -0.02:
            strength = "strong_bearish"
        elif momentum_value < -0.005:
            strength = "bearish"
        else:
            strength = "neutral"

        return {
            "value": momentum_value,
            "strength": strength,
        }

    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns."""
        if 'volume' not in data.columns:
            return {"trend": "neutral", "confirmation": False}

        recent_volume = data['volume'].tail(10)
        avg_volume = recent_volume.mean()
        current_volume = recent_volume.iloc[-1]

        volume_trend = "neutral"
        if current_volume > avg_volume * 1.2:
            volume_trend = "increasing"
        elif current_volume < avg_volume * 0.8:
            volume_trend = "decreasing"

        # Volume confirmation (high volume with price movement)
        price_change = data['close'].pct_change().iloc[-1]
        volume_confirmation = abs(price_change) > 0.005 and current_volume > avg_volume

        return {
            "trend": volume_trend,
            "confirmation": volume_confirmation,
            "current_volume": current_volume,
            "avg_volume": avg_volume,
        }

    def _analyze_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze support and resistance levels."""
        recent_data = data.tail(50)
        current_price = recent_data['close'].iloc[-1]

        # Simple pivot point analysis
        highs = recent_data['high']
        lows = recent_data['low']

        # Find potential support/resistance levels
        resistance_levels = []
        support_levels = []

        # Look for local highs and lows
        for i in range(2, len(recent_data) - 2):
            # Resistance (local high)
            if (highs.iloc[i] > highs.iloc[i-1] and
                highs.iloc[i] > highs.iloc[i-2] and
                highs.iloc[i] > highs.iloc[i+1] and
                highs.iloc[i] > highs.iloc[i+2]):
                resistance_levels.append(highs.iloc[i])

            # Support (local low)
            if (lows.iloc[i] < lows.iloc[i-1] and
                lows.iloc[i] < lows.iloc[i-2] and
                lows.iloc[i] < lows.iloc[i+1] and
                lows.iloc[i] < lows.iloc[i+2]):
                support_levels.append(lows.iloc[i])

        # Find nearby levels (within 2% of current price)
        nearby_levels = []
        tolerance = current_price * 0.02

        for level in resistance_levels[-3:]:  # Last 3 resistance levels
            if abs(level - current_price) <= tolerance:
                nearby_levels.append({"type": "resistance", "price": level})

        for level in support_levels[-3:]:  # Last 3 support levels
            if abs(level - current_price) <= tolerance:
                nearby_levels.append({"type": "support", "price": level})

        return {
            "nearby_levels": nearby_levels,
            "resistance_levels": resistance_levels[-5:],  # Keep last 5
            "support_levels": support_levels[-5:],  # Keep last 5
        }