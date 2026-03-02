"""
Market Regime Analysis Components for Action Signal Guide.

This module provides market regime detection and analysis capabilities
to enhance signal generation and validation.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, TypedDict

import numpy as np
import pandas as pd

from ztb.analysis.regime.market_regime_types import MarketRegime
from ztb.trading.signal.common.utilities import (
    calculate_volatility as calculate_volatility_util,
)

from .history_helpers import append_with_compaction

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..action_signal_guide import ActionSignal

class IMarketRegimeDetector(Protocol):
    def detect_regime(
        self, market_data: pd.DataFrame | float, step: int | None = None
    ) -> MarketRegime | str:
        """Detect market regime from price series or current price."""

class RegimeIndicators(TypedDict):
    trend_strength: float
    volatility: float
    range_bound: float
    momentum: float

class RegimeHistoryEntry(TypedDict):
    timestamp: pd.Timestamp
    regime: MarketRegime
    indicators: RegimeIndicators

class RegimeConfig(TypedDict):
    preferred_patterns: list[str]
    boost_factor: float
    penalty_factor: float

class MarketRegimeDetector(IMarketRegimeDetector):
    """
    Detects current market regime using multiple indicators.
    """

    def __init__(
        self,
        use_relative: bool = False,
        reference_window: int = 1000,
        percentile_threshold: float = 0.8,
    ) -> None:
        """Initialize market regime detector."""
        self.regime_history: list[RegimeHistoryEntry] = []
        self.transition_threshold = 0.6
        self.stability_window = 20  # periods to consider regime stable

        # Regime detection parameters
        self.trend_threshold = 0.02  # 2% trend strength
        self.volatility_threshold = 0.03  # 3% volatility
        self.range_threshold = 0.015  # 1.5% range bound
        # Relative detection settings
        self.use_relative = use_relative
        self.reference_window = (
            int(reference_window) if reference_window is not None else 1000
        )
        self.percentile_threshold = float(percentile_threshold)
        self.current_regime: MarketRegime | None = None
        # Compatibility for detectors used by reward shapers expecting rolling prices.
        self.price_history: list[float] = []

    def _append_price_history(self, price: float) -> None:
        """Track recent prices with bounded history."""
        append_with_compaction(
            self.price_history,
            price,
            high_water=2000,
            retain=1000,
        )

    def _append_regime_history(self, entry: RegimeHistoryEntry) -> None:
        """Track recent regime states with bounded history."""
        append_with_compaction(
            self.regime_history,
            entry,
            high_water=100,
            retain=50,
        )

    def detect_regime(
        self, market_data: pd.DataFrame | float, step: int | None = None
    ) -> MarketRegime | str:
        """
        Detect current market regime from price data.

        Args:
            market_data: OHLCV market data

        Returns:
            Detected market regime
        """
        if isinstance(market_data, (int, float, np.integer, np.floating)):
            return self._detect_regime_from_price(float(market_data), step)

        if not isinstance(market_data, pd.DataFrame):
            return MarketRegime.MODERATE_VOLATILITY_RANGING

        if len(market_data) < 50 or "close" not in market_data.columns:
            self.current_regime = MarketRegime.MODERATE_VOLATILITY_RANGING
            if len(market_data) > 0:
                try:
                    self._append_price_history(float(market_data["close"].iloc[-1]))
                except Exception:
                    pass
            return self.current_regime

        # Calculate regime indicators
        trend_strength = self._calculate_trend_strength(market_data)
        volatility = self._calculate_volatility(market_data)
        range_bound = self._calculate_range_bound(market_data)
        momentum = self._calculate_momentum(market_data)

        # Determine regime based on indicators
        regime = self._classify_regime(
            trend_strength, volatility, range_bound, momentum
        )

        # If relative mode is enabled, override volatility-based classification
        if self.use_relative:
            try:
                vol_percentile = self._calculate_volatility_percentile(
                    market_data, volatility
                )
                if vol_percentile >= self.percentile_threshold:
                    regime = MarketRegime.HIGH_VOLATILITY_RANGING
            except (KeyError, TypeError, ValueError):
                logger.debug("Failed relative volatility percentile evaluation")

        # Store regime for stability analysis
        self._append_price_history(float(market_data["close"].iloc[-1]))
        self._append_regime_history(
            {
                "timestamp": market_data.index[-1]
                if hasattr(market_data.index, "__getitem__")
                else pd.Timestamp.now(),
                "regime": regime,
                "indicators": {
                    "trend_strength": trend_strength,
                    "volatility": volatility,
                    "range_bound": range_bound,
                    "momentum": momentum,
                },
            }
        )

        self.current_regime = regime
        return regime

    def _calculate_volatility_percentile(
        self, data: pd.DataFrame, current_vol: float
    ) -> float:
        """
        Compute the percentile rank (0-1) of the current volatility in a reference window.
        """
        if "close" not in data.columns:
            return 0.0
        if len(data) < 30:
            # Not enough history to compute robust percentile.
            return 0.0

        # Compute volatility for sliding windows over the reference range
        vols: list[float] = []
        window = min(20, len(data))
        start_idx = max(0, len(data) - self.reference_window)
        for i in range(start_idx + window, len(data) + 1):
            sub = data["close"].iloc[i - window : i]
            if len(sub) < window:
                continue
            r = sub.pct_change().dropna()
            if len(r) == 0:
                vols.append(0.0)
            else:
                vols.append(float(r.std()))

        if len(vols) == 0:
            return 0.0

        vols_arr = np.array(vols)
        percentile = float((vols_arr <= current_vol).sum()) / len(vols_arr)
        return percentile

    def _detect_regime_from_price(self, current_price: float, step: int | None) -> str:
        """
        Compatibility path for detectors called via legacy `(current_price, step)` API.
        """
        self._append_price_history(current_price)

        if len(self.price_history) < 20:
            return "sideways"

        recent = self.price_history[-20:]
        returns = [
            (recent[i + 1] / recent[i] - 1.0) for i in range(len(recent) - 1) if recent[i]
        ]
        if not returns:
            return "sideways"

        volatility = float(np.std(returns))
        if volatility > 0.03:
            return "volatile"

        start_price = recent[0]
        if start_price == 0:
            return "sideways"
        trend = (recent[-1] - start_price) / start_price
        if trend > 0.02:
            return "bull"
        if trend < -0.02:
            return "bear"
        return "sideways"

    def detect_regime_from_data(self, market_data: pd.DataFrame) -> str:
        """
        Detect current market regime (IMarketRegimeDetector interface).

        Args:
            current_price: Current market price
            step: Current step number

        Returns:
            Market regime: 'bull', 'bear', 'sideways', 'volatile'
        """
        if len(market_data) > 0 and "close" in market_data.columns:
            detected = self.detect_regime(market_data)
            if isinstance(detected, MarketRegime):
                return detected.value
            return str(detected)
        if self.current_regime is not None:
            return self.current_regime.value
        return MarketRegime.MODERATE_VOLATILITY_RANGING.value

    def get_regime_stability(self) -> float:
        """
        Get current regime stability score.

        Returns:
            Stability score (0-1, higher = more stable)
        """
        if len(self.regime_history) < self.stability_window:
            return 0.5

        recent_regimes = [
            r["regime"] for r in self.regime_history[-self.stability_window :]
        ]
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
        if "close" not in data.columns:
            return 0.0
        prices = data["close"].values
        if len(prices) < 20:
            return 0.0

        # Use recent 20 periods for trend calculation
        recent_prices = prices[-20:]
        x = np.arange(len(recent_prices))

        # Linear regression
        slope, _ = np.polyfit(x, recent_prices, 1)

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
            Volatility (raw std of returns, e.g. 0.02 = 2%)
        """
        if "close" not in data.columns:
            return 0.0
        returns = data["close"].pct_change().dropna()
        if len(returns) < 10:
            return 0.0

        # Use recent 20 periods
        recent_returns = returns.tail(20)
        try:
            vol = calculate_volatility_util(
                recent_returns, window=min(20, len(recent_returns)), method="std"
            )
            return max(0.0, float(vol))
        except (TypeError, ValueError):
            volatility = recent_returns.std()
            return max(0.0, float(volatility))

    def _calculate_range_bound(self, data: pd.DataFrame) -> float:
        """
        Calculate how range-bound the market is.

        Args:
            data: Market data

        Returns:
            Range bound score (0-1, higher = more range-bound)
        """
        if len(data) < 20 or not {"close", "high", "low"}.issubset(data.columns):
            return 0.0

        recent_data = data.tail(20)
        high_max = recent_data["high"].max()
        low_min = recent_data["low"].min()
        current_price = recent_data["close"].iloc[-1]

        # Calculate range as percentage of current price
        if current_price == 0:
            return 0.0
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
        if len(data) < 10 or "close" not in data.columns:
            return 0.0

        # RSI-style momentum calculation
        returns = data["close"].pct_change().dropna()
        recent_returns = returns.tail(10)
        if recent_returns.empty:
            return 0.0

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
        momentum: float,
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
            return MarketRegime.HIGH_VOLATILITY_RANGING

        # Low volatility regime
        if volatility < self.volatility_threshold * 0.3:
            return MarketRegime.LOW_VOLATILITY_RANGING

        # Trending regimes
        if abs(trend_strength) > self.trend_threshold:
            if trend_strength > 0 and momentum > 0.2:
                return MarketRegime.MODERATE_BULL_TREND
            elif trend_strength < 0 and momentum < -0.2:
                return MarketRegime.MODERATE_BEAR_TREND

        # Ranging regime (default)
        if range_bound > 0.6:
            return MarketRegime.MODERATE_VOLATILITY_RANGING

        # Default to ranging if no clear regime
        return MarketRegime.MODERATE_VOLATILITY_RANGING

class RegimeAdaptiveSignalProcessor:
    """
    Processes signals adaptively based on detected market regime.
    """

    def __init__(self) -> None:
        """Initialize regime adaptive signal processor."""
        self.regime_detector = MarketRegimeDetector()
        self.regime_performance: dict[MarketRegime, list[float]] = {
            regime: [] for regime in MarketRegime
        }
        self.adaptation_factors: dict[MarketRegime, float] = {}

    def process_signals_for_regime(
        self, signals: list["ActionSignal"], market_data: pd.DataFrame
    ) -> list["ActionSignal"]:
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

        processed_signals: list["ActionSignal"] = []

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
        market_data: pd.DataFrame,
    ) -> ActionSignal | None:
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
        pattern_type_raw = str(
            getattr(signal, "signal_type", getattr(signal, "pattern_type", "unknown"))
        )
        pattern_type = self._normalize_pattern_family(pattern_type_raw)
        original_confidence = float(getattr(signal, "confidence", 0.5))

        # Regime-specific pattern preferences and adjustments
        regime_config = self._get_regime_config(regime)

        # Check if pattern is suitable for regime
        if pattern_type not in regime_config["preferred_patterns"]:
            # Reduce confidence for non-preferred patterns
            signal.confidence = original_confidence * regime_config["penalty_factor"]
        else:
            # Boost confidence for preferred patterns
            signal.confidence = min(
                1.0, original_confidence * regime_config["boost_factor"]
            )

        # Apply stability adjustment
        stability_adjustment = 1.0 + (stability - 0.5) * 0.2  # ±10% based on stability
        signal.confidence = min(1.0, max(0.0, signal.confidence * stability_adjustment))

        # Apply market condition filters
        if not self._passes_regime_filter(signal, regime, market_data):
            return None

        # Add regime metadata (store in metadata for dataclass compatibility)
        if not hasattr(signal, "metadata") or not isinstance(signal.metadata, dict):
            signal.metadata = {}
        confidence_adjustment = (
            signal.confidence / original_confidence if original_confidence > 0 else 0.0
        )
        regime_analysis = {
            "detected_regime": regime.value,
            "regime_stability": stability,
            "confidence_adjustment": confidence_adjustment,
            "regime_suitability": pattern_type in regime_config["preferred_patterns"],
            "pattern_type_raw": pattern_type_raw,
            "pattern_type_normalized": pattern_type,
        }
        signal.metadata["regime_analysis"] = regime_analysis
        # Keep legacy dynamic attribute for compatibility with external callers.
        signal.regime_analysis = regime_analysis

        return signal

    @staticmethod
    def _normalize_pattern_family(pattern_type: str) -> str:
        """Map recognizer-specific signal types to regime-compat pattern families."""
        normalized = pattern_type.lower().replace("_", "")

        if "fibonacci" in normalized:
            return "fibonacci"
        if any(key in normalized for key in ["gartley", "butterfly", "crab", "bat", "harmonic"]):
            return "harmonic"
        if "gann" in normalized:
            return "gann"
        if "dow" in normalized:
            return "dow_theory"
        if "granville" in normalized:
            return "granville"
        if any(key in normalized for key in ["bollinger"]):
            return "bollinger"
        if any(
            key in normalized
            for key in [
                "rsi",
                "macd",
                "stochastic",
                "cci",
                "williams",
                "mfi",
                "oscillator",
            ]
        ):
            return "oscillator"
        if any(
            key in normalized
            for key in [
                "hammer",
                "engulfing",
                "morningstar",
                "eveningstar",
                "piercing",
                "threewhite",
                "threeblack",
                "candlestick",
            ]
        ):
            return "candlestick"
        if any(key in normalized for key in ["chaikin", "volume"]):
            return "volume"
        if any(key in normalized for key in ["support", "resistance"]):
            return "support_resistance"
        if any(key in normalized for key in ["breakout", "breakdown"]):
            return "breakout"
        if any(key in normalized for key in ["wave", "trend", "heikin"]):
            return "trend"

        return normalized

    def _get_regime_config(self, regime: MarketRegime) -> RegimeConfig:
        """
        Get configuration for specific regime.

        Args:
            regime: Market regime

        Returns:
            Regime configuration
        """
        configs: dict[MarketRegime, RegimeConfig] = {
            MarketRegime.MODERATE_BULL_TREND: {
                "preferred_patterns": [
                    "fibonacci",
                    "harmonic",
                    "gann",
                    "dow_theory",
                    "trend",
                ],
                "boost_factor": 1.3,
                "penalty_factor": 0.7,
            },
            MarketRegime.MODERATE_BEAR_TREND: {
                "preferred_patterns": [
                    "fibonacci",
                    "harmonic",
                    "gann",
                    "dow_theory",
                    "trend",
                ],
                "boost_factor": 1.3,
                "penalty_factor": 0.7,
            },
            MarketRegime.MODERATE_VOLATILITY_RANGING: {
                "preferred_patterns": [
                    "bollinger",
                    "oscillator",
                    "volume",
                    "candlestick",
                    "support_resistance",
                ],
                "boost_factor": 1.2,
                "penalty_factor": 0.8,
            },
            MarketRegime.HIGH_VOLATILITY_RANGING: {
                "preferred_patterns": ["volume", "gann", "granville", "breakout"],
                "boost_factor": 1.4,
                "penalty_factor": 0.6,
            },
            MarketRegime.LOW_VOLATILITY_RANGING: {
                "preferred_patterns": [
                    "fibonacci",
                    "harmonic",
                    "candlestick",
                    "pattern",
                ],
                "boost_factor": 1.1,
                "penalty_factor": 0.9,
            },
        }

        return configs.get(regime, configs[MarketRegime.MODERATE_VOLATILITY_RANGING])

    def _passes_regime_filter(
        self, signal: "ActionSignal", regime: MarketRegime, market_data: pd.DataFrame
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
        confidence = getattr(signal, "confidence", 0.5)

        # Minimum confidence thresholds by regime
        min_confidence = {
            MarketRegime.MODERATE_BULL_TREND: 0.4,
            MarketRegime.MODERATE_BEAR_TREND: 0.4,
            MarketRegime.MODERATE_VOLATILITY_RANGING: 0.5,
            MarketRegime.HIGH_VOLATILITY_RANGING: 0.6,
            MarketRegime.LOW_VOLATILITY_RANGING: 0.3,
        }

        if confidence < min_confidence.get(regime, 0.4):
            return False

        # Additional regime-specific filters
        if regime == MarketRegime.HIGH_VOLATILITY_RANGING:
            # In high volatility, require stronger signals
            if confidence < 0.5:
                return False
        elif regime == MarketRegime.LOW_VOLATILITY_RANGING:
            # In low volatility, be more permissive
            pass  # No additional filter

        return True

    def update_regime_performance(
        self, regime: MarketRegime, signal_performance: float
    ) -> None:
        """
        Update performance tracking for regime adaptation.

        Args:
            regime: Market regime
            signal_performance: Performance of signal in this regime
        """
        append_with_compaction(
            self.regime_performance[regime],
            signal_performance,
            high_water=100,
            retain=50,
        )

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

    def __init__(self) -> None:
        """Initialize market condition analyzer."""
        self.condition_indicators: dict[str, object] = {}
        self.analysis_history: list[dict[str, object]] = []

    def analyze_market_conditions(self, market_data: pd.DataFrame) -> dict[str, object]:
        """
        Analyze current market conditions.

        Args:
            market_data: Market data to analyze

        Returns:
            Market condition analysis
        """
        if len(market_data) < 20:
            return self._get_default_conditions()

        analysis: dict[str, object] = {
            "trend": self._analyze_trend(market_data),
            "volatility": self._analyze_volatility(market_data),
            "momentum": self._analyze_momentum(market_data),
            "volume": self._analyze_volume(market_data),
            "support_resistance": self._analyze_support_resistance(market_data),
            "timestamp": market_data.index[-1]
            if hasattr(market_data.index, "__getitem__")
            else pd.Timestamp.now(),
        }

        # Store analysis for historical context
        append_with_compaction(
            self.analysis_history,
            analysis,
            high_water=50,
            retain=25,
        )

        return analysis

    def _get_default_conditions(self) -> dict[str, object]:
        """Get default market conditions when data is insufficient."""
        return {
            "trend": {"direction": "neutral", "strength": 0.0},
            "volatility": {"level": "medium", "value": 0.02},
            "momentum": {"value": 0.0, "strength": "neutral"},
            "volume": {"trend": "neutral", "confirmation": False},
            "support_resistance": {"nearby_levels": []},
            "timestamp": pd.Timestamp.now(),
        }

    def _analyze_trend(self, data: pd.DataFrame) -> dict[str, object]:
        """Analyze price trend."""
        if "close" not in data.columns or len(data) < 3:
            return {"direction": "neutral", "strength": 0.0, "slope": 0.0}
        recent_data = data.tail(20)
        prices = recent_data["close"].values

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

    def _analyze_volatility(self, data: pd.DataFrame) -> dict[str, object]:
        """Analyze price volatility."""
        if "close" not in data.columns:
            return {"level": "medium", "value": 0.0}
        returns = data["close"].pct_change().dropna().tail(20)
        if returns.empty:
            return {"level": "medium", "value": 0.0}
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

    def _analyze_momentum(self, data: pd.DataFrame) -> dict[str, object]:
        """Analyze price momentum."""
        if len(data) < 14 or "close" not in data.columns:
            return {"value": 0.0, "strength": "neutral"}

        # Simple momentum calculation
        current_price = data["close"].iloc[-1]
        past_price = data["close"].iloc[-14]
        if past_price == 0:
            return {"value": 0.0, "strength": "neutral"}
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

    def _analyze_volume(self, data: pd.DataFrame) -> dict[str, object]:
        """Analyze volume patterns."""
        if "volume" not in data.columns:
            return {"trend": "neutral", "confirmation": False}

        recent_volume = data["volume"].tail(10)
        avg_volume = recent_volume.mean()
        current_volume = recent_volume.iloc[-1]

        volume_trend = "neutral"
        if current_volume > avg_volume * 1.2:
            volume_trend = "increasing"
        elif current_volume < avg_volume * 0.8:
            volume_trend = "decreasing"

        # Volume confirmation (high volume with price movement)
        price_change = data["close"].pct_change().iloc[-1]
        volume_confirmation = abs(price_change) > 0.005 and current_volume > avg_volume

        return {
            "trend": volume_trend,
            "confirmation": volume_confirmation,
            "current_volume": current_volume,
            "avg_volume": avg_volume,
        }

    def _analyze_support_resistance(self, data: pd.DataFrame) -> dict[str, object]:
        """Analyze support and resistance levels."""
        if len(data) < 5 or not {"close", "high", "low"}.issubset(data.columns):
            return {
                "nearby_levels": [],
                "resistance_levels": [],
                "support_levels": [],
            }
        recent_data = data.tail(50)
        current_price = recent_data["close"].iloc[-1]
        if current_price == 0:
            return {
                "nearby_levels": [],
                "resistance_levels": [],
                "support_levels": [],
            }

        # Simple pivot point analysis
        highs = recent_data["high"]
        lows = recent_data["low"]

        # Find potential support/resistance levels
        resistance_levels = []
        support_levels = []

        # Look for local highs and lows
        for i in range(2, len(recent_data) - 2):
            # Resistance (local high)
            if (
                highs.iloc[i] > highs.iloc[i - 1]
                and highs.iloc[i] > highs.iloc[i - 2]
                and highs.iloc[i] > highs.iloc[i + 1]
                and highs.iloc[i] > highs.iloc[i + 2]
            ):
                resistance_levels.append(highs.iloc[i])

            # Support (local low)
            if (
                lows.iloc[i] < lows.iloc[i - 1]
                and lows.iloc[i] < lows.iloc[i - 2]
                and lows.iloc[i] < lows.iloc[i + 1]
                and lows.iloc[i] < lows.iloc[i + 2]
            ):
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
