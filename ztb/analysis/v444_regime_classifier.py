"""
SAC v444 Advanced Regime Classification System

This module implements the 12-regime classification system for SAC v444,
providing sophisticated market regime detection and adaptation capabilities.

Regimes:
1. strong_bull_trend - Strong upward momentum with high conviction
2. moderate_bull_trend - Moderate upward trend with steady gains
3. weak_bull_trend - Weak upward movement with low momentum
4. strong_bear_trend - Strong downward momentum with high conviction
5. moderate_bear_trend - Moderate downward trend with steady losses
6. weak_bear_trend - Weak downward movement with low momentum
7. high_volatility_ranging - High volatility sideways movement
8. moderate_volatility_ranging - Moderate volatility consolidation
9. low_volatility_ranging - Low volatility tight range
10. extreme_volatility - Extreme market volatility conditions
11. consolidation - Market consolidation with balanced forces
12. breakout_setup - Potential breakout formation
13. breakdown_setup - Potential breakdown formation
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


ConfigDict = Dict[str, Union[str, int, float, bool, Dict[str, Any], List[Any]]]


class TimeFrame(Enum):
    """Time frame definitions for multi-timeframe analysis"""
    SHORT = "short"      # 5-15 minutes
    MEDIUM = "medium"    # 1-4 hours
    LONG = "long"        # Daily


class RegimeType(Enum):
    """Enumeration of all 12 market regime types in SAC v444"""

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
class RegimeMetrics:
    """Container for regime detection metrics"""

    trend_strength: float
    bull_strength: float
    bear_strength: float
    volatility: float
    momentum: float
    volume_trend: float
    price_range_ratio: float
    adx: float
    rsi: float
    macd_signal: float
    bollinger_position: float
    support_resistance_strength: float


@dataclass
class RegimeDetectionResult:
    """Result of regime detection analysis"""

    primary_regime: RegimeType
    confidence: float
    secondary_regimes: List[Tuple[RegimeType, float]]
    metrics: RegimeMetrics
    detection_timestamp: pd.Timestamp
    lookback_period: int


@dataclass
class MultiTimeFrameMetrics:
    """Metrics from multiple timeframes"""
    short_term: RegimeMetrics
    medium_term: RegimeMetrics
    long_term: RegimeMetrics
    timeframe_weights: Dict[str, float]
    integrated_regime: RegimeType
    integration_confidence: float


class V444RegimeClassifier:
    """
    Advanced regime classifier for SAC v444 with 12-regime detection

    This classifier uses multi-timeframe analysis and sophisticated
    technical indicators to accurately identify market regimes.
    """

    def __init__(self, config: Optional[ConfigDict] = None):
        """
        Initialize the regime classifier

        Args:
            config: Configuration dictionary with regime parameters
        """
        self.config = config or {}
        self.lookback_periods: Dict[str, int] = self.config.get(
            "lookback_periods", {"short": 20, "medium": 50, "long": 100}
        )  # type: ignore

        # Regime detection thresholds
        self.thresholds: Dict[str, float] = self.config.get(
            "thresholds",
            {
                "strong_trend_threshold": 3.0,
                "moderate_trend_threshold": 2.0,
                "weak_trend_threshold": 1.0,
                "high_volatility_threshold": 0.15,  # Adjusted for log returns volatility levels
                "moderate_volatility_threshold": 0.10,  # Adjusted for log returns volatility levels
                "extreme_volatility_threshold": 0.20,  # Adjusted for log returns volatility levels
                "consolidation_range_threshold": 0.05,  # Adjusted for very low volatility
                "breakout_setup_threshold": 0.15,
            },
        )  # type: ignore

        logger.info("V444 Regime Classifier initialized")

        # Dynamic threshold adaptation
        self.dynamic_thresholds_enabled = self.config.get("dynamic_thresholds", True)
        self.adaptation_window = self.config.get("adaptation_window", 100)
        self.market_stats_history = []

        # Debug metrics
        self.metrics = {
            "candidate_bull": 0,
            "candidate_bear": 0,
            "consolidation_fallback": 0,
        }

    def detect_regime(
        self, data: pd.DataFrame, current_index: int = -1
    ) -> RegimeDetectionResult:
        """
        Detect the current market regime from price data

        Args:
            data: OHLCV DataFrame
            current_index: Index to analyze (default: latest)

        Returns:
            RegimeDetectionResult with detected regime and confidence

        Raises:
            ValueError: If data is invalid or insufficient
            RuntimeError: If regime detection fails
        """
        try:
            if data is None or data.empty:
                raise ValueError("Input data cannot be None or empty")

            if current_index == -1:
                current_index = len(data) - 1

            if current_index < 0 or current_index >= len(data):
                raise ValueError(f"Invalid current_index: {current_index}")

            # Calculate regime metrics
            metrics = self._calculate_regime_metrics(data, current_index)

            # Adapt thresholds based on market conditions if enabled
            if self.dynamic_thresholds_enabled:
                self._adapt_thresholds(data, current_index, metrics)

            # Determine primary regime
            primary_regime, confidence = self._classify_regime(metrics)

            # Calculate secondary regimes
            secondary_regimes = self._calculate_secondary_regimes(metrics, primary_regime)

            return RegimeDetectionResult(
                primary_regime=primary_regime,
                confidence=confidence,
                secondary_regimes=secondary_regimes,
                metrics=metrics,
                detection_timestamp=data.index[current_index] if hasattr(data, 'index') else pd.Timestamp.now(),
                lookback_period=self.lookback_periods["medium"],
            )

        except Exception as e:
            logger.error(f"Regime detection failed at index {current_index}: {e}")
            # Return fallback result
            fallback_metrics = self._get_default_metrics()
            return RegimeDetectionResult(
                primary_regime=RegimeType.CONSOLIDATION,
                confidence=0.0,
                secondary_regimes=[],
                metrics=fallback_metrics,
                detection_timestamp=pd.Timestamp.now(),
                lookback_period=self.lookback_periods["medium"],
            )

    def _calculate_regime_metrics(
        self, data: pd.DataFrame, index: int
    ) -> RegimeMetrics:
        """
        Calculate comprehensive regime detection metrics

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze

        Returns:
            RegimeMetrics object with calculated indicators
        """
        # Ensure we have enough data
        min_periods = max(self.lookback_periods.values())
        if index < min_periods:
            return self._get_default_metrics()

        # Extract price data
        close = data["close"].iloc[
            max(0, index - self.lookback_periods["long"]) : index + 1
        ]
        high = data["high"].iloc[
            max(0, index - self.lookback_periods["long"]) : index + 1
        ]
        low = data["low"].iloc[
            max(0, index - self.lookback_periods["long"]) : index + 1
        ]
        volume = (
            data["volume"].iloc[
                max(0, index - self.lookback_periods["long"]) : index + 1
            ]
            if "volume" in data.columns
            else pd.Series([1.0] * len(close))
        )

        # Calculate trend strength (ADX-like)
        trend_strength, bull_strength, bear_strength = self._calculate_trend_strength(
            high, low, close
        )

        # Calculate volatility
        volatility = self._calculate_volatility(close)

        # Calculate momentum
        momentum = self._calculate_momentum(close)

        # Calculate volume trend
        volume_trend = self._calculate_volume_trend(volume)

        # Calculate price range ratio
        price_range_ratio = self._calculate_price_range_ratio(high, low, close)

        # Calculate ADX
        adx = self._calculate_adx(high, low, close)

        # Calculate RSI
        rsi = self._calculate_rsi(close)

        # Calculate MACD signal
        macd_signal = self._calculate_macd_signal(close)

        # Calculate Bollinger Band position
        bollinger_position = self._calculate_bollinger_position(close)

        # Calculate support/resistance strength
        support_resistance_strength = self._calculate_support_resistance_strength(
            high, low, close
        )

        return RegimeMetrics(
            trend_strength=trend_strength,
            bull_strength=bull_strength,
            bear_strength=bear_strength,
            volatility=volatility,
            momentum=momentum,
            volume_trend=volume_trend,
            price_range_ratio=price_range_ratio,
            adx=adx,
            rsi=rsi,
            macd_signal=macd_signal,
            bollinger_position=bollinger_position,
            support_resistance_strength=support_resistance_strength,
        )

    def _calculate_trend_strength(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> Tuple[float, float, float]:
        """Calculate trend strength with directional components"""
        try:
            # Convert to pandas Series if not already
            if not isinstance(high, pd.Series):
                high = pd.Series(high)
            if not isinstance(low, pd.Series):
                low = pd.Series(low)
            if not isinstance(close, pd.Series):
                close = pd.Series(close)

            # Ensure we have enough data
            if len(high) < 14:
                return 0.0, 0.0, 0.0

            # Calculate price momentum over 10 periods (relative change)
            momentum = close / close.shift(10) - 1  # More numerically stable

            # Calculate volatility (standard deviation of log returns, scaled)
            log_returns = np.log(close / close.shift())
            volatility = log_returns.rolling(10).std() * np.sqrt(
                10
            )  # Scale to match volatility calculation

            # Calculate raw trend strength with direction
            trend_strength_raw = momentum / (volatility + 1e-8)

            # Smooth the trend strength without clipping
            trend_strength_smooth = trend_strength_raw.ewm(span=3).mean()

            # Separate bull and bear strength
            bull_strength = trend_strength_smooth.clip(lower=0)
            bear_strength = (-trend_strength_smooth).clip(lower=0)

            result_trend = (
                float(trend_strength_smooth.iloc[-1])
                if not trend_strength_smooth.empty
                else 0.0
            )
            result_bull = (
                float(bull_strength.iloc[-1]) if not bull_strength.empty else 0.0
            )
            result_bear = (
                float(bear_strength.iloc[-1]) if not bear_strength.empty else 0.0
            )

            # Debug output
            scaled_volatility = volatility.iloc[-1] if not volatility.empty else 0.0
            print(
                f"DEBUG TREND: momentum={momentum.iloc[-1]:.6f}, volatility={scaled_volatility:.6f}, trend_strength={result_trend:.6f}, bull_strength={result_bull:.6f}, bear_strength={result_bear:.6f}"
            )

            return result_trend, result_bull, result_bear

        except Exception as e:
            logger.warning(f"Error calculating trend strength: {e}")
            return 0.0, 0.0, 0.0

    def _calculate_volatility(self, close: pd.Series) -> float:
        """Calculate normalized volatility"""
        try:
            returns = close.pct_change().fillna(0)
            volatility = returns.rolling(20).std()

            # Scale volatility appropriately for trend strength calculation
            # Use more reasonable scaling to prevent over-amplification
            scaled_volatility = volatility * 10  # Reduced scaling

            return (
                float(scaled_volatility.iloc[-1])
                if not scaled_volatility.empty
                else 0.0
            )

        except Exception as e:
            logger.warning(f"Error calculating volatility: {e}")
            return 0.0

    def _calculate_momentum(self, close: pd.Series) -> float:
        """Calculate momentum indicator"""
        try:
            # ROC (Rate of Change) - keep raw values for classification
            roc = (close - close.shift(10)) / close.shift(10)
            momentum = roc.rolling(5).mean()

            return float(momentum.iloc[-1]) if not momentum.empty else 0.0

        except Exception as e:
            logger.warning(f"Error calculating momentum: {e}")
            return 0.0

    def _calculate_volume_trend(self, volume: pd.Series) -> float:
        """Calculate volume trend"""
        try:
            volume_ma_short = volume.rolling(10).mean()
            volume_ma_long = volume.rolling(30).mean()
            volume_trend = (volume_ma_short - volume_ma_long) / volume_ma_long

            return float(volume_trend.iloc[-1]) if not volume_trend.empty else 0.0

        except Exception as e:
            logger.warning(f"Error calculating volume trend: {e}")
            return 0.0

    def _calculate_price_range_ratio(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> float:
        """Calculate price range ratio (volatility measure)"""
        try:
            price_range = (high - low) / close.shift(1)
            range_ratio = price_range.rolling(20).mean()

            return float(range_ratio.iloc[-1]) if not range_ratio.empty else 0.0

        except Exception as e:
            logger.warning(f"Error calculating price range ratio: {e}")
            return 0.0

    def _calculate_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> float:
        """Calculate ADX (Average Directional Index)"""
        try:
            # Convert to pandas Series if not already
            if not isinstance(high, pd.Series):
                high = pd.Series(high)
            if not isinstance(low, pd.Series):
                low = pd.Series(low)
            if not isinstance(close, pd.Series):
                close = pd.Series(close)

            # Calculate True Range
            tr = np.maximum(
                high - low,
                np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
            )

            # Calculate Directional Movement
            dm_plus = np.where(
                (high - high.shift(1)) > (low.shift(1) - low),
                np.maximum(high - high.shift(1), 0),
                0,
            )
            dm_minus = np.where(
                (low.shift(1) - low) > (high - high.shift(1)),
                np.maximum(low.shift(1) - low, 0),
                0,
            )

            # Convert to pandas Series for rolling operations
            tr_series = pd.Series(tr)
            dm_plus_series = pd.Series(dm_plus)
            dm_minus_series = pd.Series(dm_minus)

            period = 14
            atr = tr_series.rolling(period).mean()
            di_plus = (dm_plus_series.rolling(period).mean() / atr).fillna(0)
            di_minus = (dm_minus_series.rolling(period).mean() / atr).fillna(0)

            dx = abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
            adx = dx.rolling(period).mean()

            return float(adx.iloc[-1]) if not adx.empty else 0.0

        except Exception as e:
            logger.warning(f"Error calculating ADX: {e}")
            return 0.0

    def _calculate_rsi(self, close: pd.Series) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi.iloc[-1]) if not rsi.empty else 50.0

        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            return 50.0

    def _calculate_macd_signal(self, close: pd.Series) -> float:
        """Calculate MACD signal strength"""
        try:
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_signal = macd - signal

            return float(macd_signal.iloc[-1]) if not macd_signal.empty else 0.0

        except Exception as e:
            logger.warning(f"Error calculating MACD signal: {e}")
            return 0.0

    def _calculate_bollinger_position(self, close: pd.Series) -> float:
        """Calculate position within Bollinger Bands"""
        try:
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            upper = sma + 2 * std
            lower = sma - 2 * std

            position = (close - lower) / (upper - lower + 1e-10)
            position = position.clip(0, 1)  # Normalize to 0-1

            return float(position.iloc[-1]) if not position.empty else 0.5

        except Exception as e:
            logger.warning(f"Error calculating Bollinger position: {e}")
            return 0.5

    def _calculate_support_resistance_strength(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> float:
        """Calculate support/resistance strength"""
        try:
            # Simple support/resistance calculation
            recent_high = high.rolling(20).max()
            recent_low = low.rolling(20).min()

            resistance_distance = (recent_high - close) / close
            support_distance = (close - recent_low) / close

            strength = 1.0 - min(
                resistance_distance.iloc[-1], support_distance.iloc[-1]
            )

            return float(strength) if not np.isnan(strength) else 0.5

        except Exception as e:
            logger.warning(f"Error calculating support/resistance strength: {e}")
            return 0.5

    def _classify_regime(self, metrics: RegimeMetrics) -> Tuple[RegimeType, float]:
        """
        Classify the market regime based on calculated metrics

        Args:
            metrics: Calculated regime metrics

        Returns:
            Tuple of (regime_type, confidence_score)
        """
        # Strong trend classification first (highest priority)
        if (
            metrics.bull_strength > self.thresholds["strong_trend_threshold"]
            and metrics.momentum > 0.001
        ):
            self.metrics["candidate_bull"] += 1
            return RegimeType.STRONG_BULL_TREND, min(metrics.bull_strength, 0.95)
        elif (
            metrics.bear_strength > self.thresholds["strong_trend_threshold"]
            and metrics.momentum < -0.001
        ):
            self.metrics["candidate_bear"] += 1
            return RegimeType.STRONG_BEAR_TREND, min(metrics.bear_strength, 0.95)

        # Check for ranging regimes (high volatility with moderate trend strength)
        trend_strength_abs = abs(metrics.trend_strength)
        if (
            metrics.volatility > self.thresholds["extreme_volatility_threshold"]
            and trend_strength_abs < 6.0
        ):
            return RegimeType.EXTREME_VOLATILITY, min(metrics.volatility * 0.8, 0.9)
        elif (
            metrics.volatility > self.thresholds["high_volatility_threshold"]
            and trend_strength_abs < 5.0
        ):
            return RegimeType.HIGH_VOLATILITY_RANGING, min(
                metrics.volatility * 0.7, 0.8
            )
        elif (
            metrics.volatility > self.thresholds["moderate_volatility_threshold"]
            and trend_strength_abs < 4.0
        ):
            return RegimeType.MODERATE_VOLATILITY_RANGING, min(
                metrics.volatility * 0.6, 0.7
            )
        elif (
            metrics.volatility > self.thresholds["consolidation_range_threshold"]
            and trend_strength_abs < 3.0
        ):
            return RegimeType.LOW_VOLATILITY_RANGING, min(metrics.volatility * 0.5, 0.6)

        # Moderate/weak trend classification
        if (
            metrics.bull_strength > self.thresholds["moderate_trend_threshold"]
            and metrics.momentum > 0.0005
        ):
            self.metrics["candidate_bull"] += 1
            return RegimeType.MODERATE_BULL_TREND, min(
                metrics.bull_strength * 0.9, 0.85
            )
        elif (
            metrics.bull_strength > self.thresholds["weak_trend_threshold"]
            and metrics.momentum > 0.0002
        ):
            self.metrics["candidate_bull"] += 1
            return RegimeType.WEAK_BULL_TREND, min(metrics.bull_strength * 0.8, 0.75)

        if (
            metrics.bear_strength > self.thresholds["moderate_trend_threshold"]
            and metrics.momentum < -0.0005
        ):
            self.metrics["candidate_bear"] += 1
            return RegimeType.MODERATE_BEAR_TREND, min(
                metrics.bear_strength * 0.9, 0.85
            )
        elif (
            metrics.bear_strength > self.thresholds["weak_trend_threshold"]
            and metrics.momentum < -0.0002
        ):
            self.metrics["candidate_bear"] += 1
            return RegimeType.WEAK_BEAR_TREND, min(metrics.bear_strength * 0.8, 0.75)

        # Check for breakout/breakdown setups (lowest priority)
        if (
            metrics.bollinger_position > 0.8
            and metrics.rsi > 70
            and metrics.macd_signal > 0
        ):
            return RegimeType.BREAKOUT_SETUP, 0.8
        elif (
            metrics.bollinger_position < 0.2
            and metrics.rsi < 30
            and metrics.macd_signal < 0
        ):
            return RegimeType.BREAKDOWN_SETUP, 0.8

        # Default to consolidation
        self.metrics["consolidation_fallback"] += 1
        return RegimeType.CONSOLIDATION, 0.5

    def _calculate_secondary_regimes(
        self, metrics: RegimeMetrics, primary_regime: RegimeType
    ) -> List[Tuple[RegimeType, float]]:
        """
        Calculate secondary regime possibilities with confidence scores

        Args:
            metrics: Calculated regime metrics
            primary_regime: Already determined primary regime

        Returns:
            List of (regime, confidence) tuples for secondary regimes
        """
        secondary_candidates = []

        # Calculate confidence for each regime type
        regime_scores = self._calculate_all_regime_scores(metrics)

        # Sort by confidence and exclude primary regime
        sorted_regimes = sorted(regime_scores.items(), key=lambda x: x[1], reverse=True)
        secondary_regimes = [
            (regime, score)
            for regime, score in sorted_regimes
            if regime != primary_regime
        ][:3]  # Top 3 secondary regimes

        return secondary_regimes

    def _calculate_all_regime_scores(
        self, metrics: RegimeMetrics
    ) -> Dict[RegimeType, float]:
        """Calculate confidence scores for all regime types"""
        scores = {}

        # Trend regimes
        scores[RegimeType.STRONG_BULL_TREND] = min(metrics.bull_strength, 0.95)
        scores[RegimeType.MODERATE_BULL_TREND] = min(metrics.bull_strength * 0.9, 0.85)
        scores[RegimeType.WEAK_BULL_TREND] = min(metrics.bull_strength * 0.8, 0.75)
        scores[RegimeType.STRONG_BEAR_TREND] = min(metrics.bear_strength, 0.95)
        scores[RegimeType.MODERATE_BEAR_TREND] = min(metrics.bear_strength * 0.9, 0.85)
        scores[RegimeType.WEAK_BEAR_TREND] = min(metrics.bear_strength * 0.8, 0.75)

        # Volatility regimes
        vol_factor = metrics.volatility
        scores[RegimeType.EXTREME_VOLATILITY] = min(vol_factor * 0.8, 0.9)
        scores[RegimeType.HIGH_VOLATILITY_RANGING] = min(vol_factor * 0.7, 0.8)
        scores[RegimeType.MODERATE_VOLATILITY_RANGING] = min(vol_factor * 0.6, 0.7)
        scores[RegimeType.LOW_VOLATILITY_RANGING] = max(0, 0.6 - vol_factor * 0.5)

        # Consolidation and setup regimes
        range_factor = 1 - metrics.price_range_ratio
        scores[RegimeType.CONSOLIDATION] = min(range_factor * 0.8, 0.7)

        breakout_factor = abs(metrics.momentum) * metrics.support_resistance_strength
        scores[RegimeType.BREAKOUT_SETUP] = (
            min(breakout_factor * 1.5, 0.8) if metrics.momentum > 0 else 0
        )
        scores[RegimeType.BREAKDOWN_SETUP] = (
            min(breakout_factor * 1.5, 0.8) if metrics.momentum < 0 else 0
        )

        return scores

    def _get_default_metrics(self) -> RegimeMetrics:
        """Return default metrics when insufficient data"""
        return RegimeMetrics(
            trend_strength=0.0,
            bull_strength=0.0,
            bear_strength=0.0,
            volatility=0.0,
            momentum=0.0,
            volume_trend=0.0,
            price_range_ratio=0.0,
            adx=0.0,
            rsi=50.0,
            macd_signal=0.0,
            bollinger_position=0.5,
            support_resistance_strength=0.5,
        )

    def get_regime_config(self, regime: RegimeType) -> Dict[str, Any]:
        """
        Get configuration parameters for a specific regime

        Args:
            regime: Regime type to get config for

        Returns:
            Dictionary with regime-specific parameters
        """
        regime_configs = self.config.get("regime_configs", {})

        # Default configurations for each regime
        default_configs = {
            RegimeType.STRONG_BULL_TREND: {
                "action_balance_target": 0.95,
                "entropy_regularization": 0.005,
                "feature_weights": {"momentum": 1.2, "trend": 1.1, "volume": 0.9},
            },
            RegimeType.MODERATE_BULL_TREND: {
                "action_balance_target": 0.85,
                "entropy_regularization": 0.01,
                "feature_weights": {"momentum": 1.1, "trend": 1.0, "volume": 0.95},
            },
            RegimeType.WEAK_BULL_TREND: {
                "action_balance_target": 0.75,
                "entropy_regularization": 0.015,
                "feature_weights": {"momentum": 1.0, "trend": 0.9, "volume": 1.0},
            },
            RegimeType.STRONG_BEAR_TREND: {
                "action_balance_target": 0.05,
                "entropy_regularization": 0.005,
                "feature_weights": {"momentum": 1.2, "trend": 1.1, "volume": 0.9},
            },
            RegimeType.MODERATE_BEAR_TREND: {
                "action_balance_target": 0.15,
                "entropy_regularization": 0.01,
                "feature_weights": {"momentum": 1.1, "trend": 1.0, "volume": 0.95},
            },
            RegimeType.WEAK_BEAR_TREND: {
                "action_balance_target": 0.25,
                "entropy_regularization": 0.015,
                "feature_weights": {"momentum": 1.0, "trend": 0.9, "volume": 1.0},
            },
            RegimeType.HIGH_VOLATILITY_RANGING: {
                "action_balance_target": 0.5,
                "entropy_regularization": 0.02,
                "feature_weights": {"volatility": 1.3, "momentum": 0.8, "trend": 0.7},
            },
            RegimeType.MODERATE_VOLATILITY_RANGING: {
                "action_balance_target": 0.5,
                "entropy_regularization": 0.018,
                "feature_weights": {"volatility": 1.2, "momentum": 0.9, "trend": 0.8},
            },
            RegimeType.LOW_VOLATILITY_RANGING: {
                "action_balance_target": 0.5,
                "entropy_regularization": 0.025,
                "feature_weights": {"volatility": 0.8, "momentum": 1.1, "trend": 1.0},
            },
            RegimeType.EXTREME_VOLATILITY: {
                "action_balance_target": 0.5,
                "entropy_regularization": 0.03,
                "feature_weights": {"volatility": 1.5, "momentum": 0.6, "trend": 0.5},
            },
            RegimeType.CONSOLIDATION: {
                "action_balance_target": 0.5,
                "entropy_regularization": 0.02,
                "feature_weights": {"volatility": 0.9, "momentum": 0.9, "trend": 1.1},
            },
            RegimeType.BREAKOUT_SETUP: {
                "action_balance_target": 0.7,
                "entropy_regularization": 0.012,
                "feature_weights": {
                    "momentum": 1.3,
                    "support_resistance": 1.2,
                    "volume": 1.1,
                },
            },
            RegimeType.BREAKDOWN_SETUP: {
                "action_balance_target": 0.3,
                "entropy_regularization": 0.012,
                "feature_weights": {
                    "momentum": 1.3,
                    "support_resistance": 1.2,
                    "volume": 1.1,
                },
            },
        }

        # Merge with any custom config
        regime_config = default_configs.get(regime, {})
        custom_config = regime_configs.get(regime.value, {})
        regime_config.update(custom_config)

        return regime_config

    def get_adaptive_feature_weights(
        self, regime: RegimeType, base_features: List[str]
    ) -> Dict[str, float]:
        """
        Get adaptive feature weights based on detected regime

        Args:
            regime: Detected market regime
            base_features: List of base feature names

        Returns:
            Dictionary mapping feature categories to weights
        """
        try:
            regime_config = self.get_regime_config(regime)

            # Get base weights from regime config
            base_weights = regime_config.get("feature_weights", {})

            # Create adaptive weights for all feature categories
            adaptive_weights = {}

            # Map feature names to categories
            feature_category_map = self._map_features_to_categories(base_features)

            # Apply regime-specific weights
            for feature, category in feature_category_map.items():
                if category in base_weights:
                    adaptive_weights[feature] = base_weights[category]
                else:
                    # Default weight for unmapped categories
                    adaptive_weights[feature] = 1.0

            # Apply market condition adjustments
            adaptive_weights = self._apply_market_condition_adjustments(
                adaptive_weights, regime
            )

            logger.debug(f"Adaptive feature weights for {regime.value}: {adaptive_weights}")
            return adaptive_weights

        except Exception as e:
            logger.warning(f"Error getting adaptive feature weights: {e}")
            return {feature: 1.0 for feature in base_features}

    def _map_features_to_categories(self, features: List[str]) -> Dict[str, str]:
        """
        Map feature names to their categories

        Args:
            features: List of feature names

        Returns:
            Dictionary mapping feature names to categories
        """
        category_map = {}

        for feature in features:
            feature_lower = feature.lower()

            # Momentum indicators
            if any(keyword in feature_lower for keyword in ['rsi', 'stoch', 'williams', 'momentum', 'roc', 'macd']):
                category_map[feature] = "momentum"

            # Trend indicators
            elif any(keyword in feature_lower for keyword in ['adx', 'trend', 'dmi', 'slope', 'linear']):
                category_map[feature] = "trend"

            # Volatility indicators
            elif any(keyword in feature_lower for keyword in ['atr', 'bollinger', 'std', 'volatility', 'range']):
                category_map[feature] = "volatility"

            # Volume indicators
            elif any(keyword in feature_lower for keyword in ['volume', 'obv', 'vwap', 'money_flow']):
                category_map[feature] = "volume"

            # Support/Resistance
            elif any(keyword in feature_lower for keyword in ['support', 'resistance', 'pivot']):
                category_map[feature] = "support_resistance"

            # Default category
            else:
                category_map[feature] = "general"

        return category_map

    def _apply_market_condition_adjustments(
        self, weights: Dict[str, float], regime: RegimeType
    ) -> Dict[str, float]:
        """
        Apply additional market condition adjustments to feature weights

        Args:
            weights: Base feature weights
            regime: Current market regime

        Returns:
            Adjusted feature weights
        """
        adjusted_weights = weights.copy()

        # Get recent market statistics for additional adaptation
        if len(self.market_stats_history) >= 10:
            recent_stats = self.market_stats_history[-10:]
            avg_volatility = np.mean([s["volatility"] for s in recent_stats])
            avg_trend_strength = np.mean([s["trend_strength"] for s in recent_stats])

            # Adjust weights based on recent market conditions
            if regime in [RegimeType.STRONG_BULL_TREND, RegimeType.STRONG_BEAR_TREND]:
                # In strong trends, emphasize trend-following features
                for feature, category in self._map_features_to_categories(list(weights.keys())).items():
                    if category == "trend":
                        adjusted_weights[feature] *= 1.1
                    elif category == "volatility":
                        adjusted_weights[feature] *= 0.9

            elif regime in [RegimeType.HIGH_VOLATILITY_RANGING, RegimeType.EXTREME_VOLATILITY]:
                # In high volatility, emphasize volatility and momentum features
                for feature, category in self._map_features_to_categories(list(weights.keys())).items():
                    if category == "volatility":
                        adjusted_weights[feature] *= 1.2
                    elif category == "momentum":
                        adjusted_weights[feature] *= 1.1

            elif regime == RegimeType.CONSOLIDATION:
                # In consolidation, balance all features
                for feature in adjusted_weights:
                    adjusted_weights[feature] = 1.0

        return adjusted_weights

    def detect_multi_timeframe_regime(
        self, data: pd.DataFrame, current_index: int = -1
    ) -> MultiTimeFrameMetrics:
        """
        Detect regime using multi-timeframe analysis

        Args:
            data: OHLCV DataFrame (should contain multiple timeframes)
            current_index: Current index to analyze

        Returns:
            MultiTimeFrameMetrics with integrated analysis
        """
        try:
            if current_index == -1:
                current_index = len(data) - 1

            # Calculate metrics for each timeframe
            short_metrics = self._calculate_timeframe_metrics(data, current_index, TimeFrame.SHORT)
            medium_metrics = self._calculate_timeframe_metrics(data, current_index, TimeFrame.MEDIUM)
            long_metrics = self._calculate_timeframe_metrics(data, current_index, TimeFrame.LONG)

            # Determine regime for each timeframe
            short_regime, short_conf = self._classify_regime(short_metrics)
            medium_regime, medium_conf = self._classify_regime(medium_metrics)
            long_regime, long_conf = self._classify_regime(long_metrics)

            # Integrate regimes across timeframes
            integrated_regime, integration_confidence, timeframe_weights = self._integrate_timeframe_regimes(
                short_regime, short_conf, medium_regime, medium_conf, long_regime, long_conf
            )

            return MultiTimeFrameMetrics(
                short_term=short_metrics,
                medium_term=medium_metrics,
                long_term=long_metrics,
                timeframe_weights=timeframe_weights,
                integrated_regime=integrated_regime,
                integration_confidence=integration_confidence
            )

        except Exception as e:
            logger.warning(f"Multi-timeframe analysis failed: {e}")
            # Return fallback with medium-term only
            fallback_metrics = self._calculate_regime_metrics(data, current_index)
            return MultiTimeFrameMetrics(
                short_term=fallback_metrics,
                medium_term=fallback_metrics,
                long_term=fallback_metrics,
                timeframe_weights={"short": 0.2, "medium": 0.6, "long": 0.2},
                integrated_regime=RegimeType.CONSOLIDATION,
                integration_confidence=0.5
            )

    def _calculate_timeframe_metrics(
        self, data: pd.DataFrame, index: int, timeframe: TimeFrame
    ) -> RegimeMetrics:
        """
        Calculate regime metrics for a specific timeframe

        Args:
            data: OHLCV DataFrame
            index: Current index
            timeframe: Time frame to analyze

        Returns:
            RegimeMetrics for the specified timeframe
        """
        # Adjust lookback periods based on timeframe
        timeframe_multipliers = {
            TimeFrame.SHORT: 0.5,   # Shorter lookback for short-term
            TimeFrame.MEDIUM: 1.0,  # Standard lookback for medium-term
            TimeFrame.LONG: 2.0     # Longer lookback for long-term
        }

        multiplier = timeframe_multipliers[timeframe]

        # Temporarily adjust lookback periods
        original_periods = self.lookback_periods.copy()
        self.lookback_periods = {
            "short": int(original_periods["short"] * multiplier),
            "medium": int(original_periods["medium"] * multiplier),
            "long": int(original_periods["long"] * multiplier)
        }

        try:
            # Calculate metrics with adjusted periods
            metrics = self._calculate_regime_metrics(data, index)
            return metrics
        finally:
            # Restore original periods
            self.lookback_periods = original_periods

    def _integrate_timeframe_regimes(
        self,
        short_regime: RegimeType, short_conf: float,
        medium_regime: RegimeType, medium_conf: float,
        long_regime: RegimeType, long_conf: float
    ) -> Tuple[RegimeType, float, Dict[str, float]]:
        """
        Integrate regime classifications from multiple timeframes

        Args:
            short_regime, medium_regime, long_regime: Regimes from each timeframe
            short_conf, medium_conf, long_conf: Confidences from each timeframe

        Returns:
            Tuple of (integrated_regime, confidence, weights)
        """
        # Define timeframe weights (long-term has highest weight for stability)
        base_weights = {
            "short": 0.2,   # Short-term: 20% (entry/exit timing)
            "medium": 0.3,  # Medium-term: 30% (trend direction)
            "long": 0.5     # Long-term: 50% (market environment)
        }

        # Adjust weights based on regime stability
        stability_scores = self._calculate_regime_stability_scores({
            short_regime.value: {short_regime.value: short_conf},
            medium_regime.value: {medium_regime.value: medium_conf},
            long_regime.value: {long_regime.value: long_conf}
        })

        # Boost weight for more stable regimes
        adjusted_weights = {}
        for tf, base_weight in base_weights.items():
            regime_name = locals()[f"{tf}_regime"].value
            stability = stability_scores.get(regime_name, 0.5)
            adjusted_weights[tf] = base_weight * (1 + stability * 0.5)

        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        normalized_weights = {tf: w / total_weight for tf, w in adjusted_weights.items()}

        # Calculate weighted regime scores
        regime_scores = {}
        for regime in RegimeType:
            score = (
                normalized_weights["short"] * (1.0 if short_regime == regime else 0.0) * short_conf +
                normalized_weights["medium"] * (1.0 if medium_regime == regime else 0.0) * medium_conf +
                normalized_weights["long"] * (1.0 if long_regime == regime else 0.0) * long_conf
            )
            regime_scores[regime] = score

        # Select regime with highest score
        integrated_regime = max(regime_scores.keys(), key=lambda r: regime_scores[r])
        integration_confidence = regime_scores[integrated_regime]

        return integrated_regime, integration_confidence, normalized_weights

    def _adapt_thresholds(
        self, data: pd.DataFrame, current_index: int, metrics: RegimeMetrics
    ) -> None:
        """
        Dynamically adapt classification thresholds based on market conditions

        Args:
            data: OHLCV DataFrame
            current_index: Current index
            metrics: Current regime metrics
        """
        try:
            # Store market statistics for adaptation
            market_stats = {
                "volatility": metrics.volatility,
                "trend_strength": abs(metrics.trend_strength),
                "timestamp": data.index[current_index] if hasattr(data, 'index') else None,
            }
            self.market_stats_history.append(market_stats)

            # Keep only recent history
            if len(self.market_stats_history) > self.adaptation_window:
                self.market_stats_history = self.market_stats_history[-self.adaptation_window:]

            # Calculate adaptive thresholds based on recent market conditions
            if len(self.market_stats_history) >= 20:  # Need minimum history
                recent_volatilities = [s["volatility"] for s in self.market_stats_history[-50:]]
                recent_trend_strengths = [s["trend_strength"] for s in self.market_stats_history[-50:]]

                # Calculate percentile-based thresholds
                vol_p25 = np.percentile(recent_volatilities, 25)
                vol_p75 = np.percentile(recent_volatilities, 75)
                trend_p75 = np.percentile(recent_trend_strengths, 75)

                # Adapt thresholds based on market regime
                volatility_regime = "normal"
                if metrics.volatility > vol_p75 * 1.5:
                    volatility_regime = "high"
                elif metrics.volatility < vol_p25 * 0.7:
                    volatility_regime = "low"

                # Adjust thresholds based on volatility regime
                if volatility_regime == "high":
                    # In high volatility, require stronger signals
                    self.thresholds["strong_trend_threshold"] = max(4.0, trend_p75 * 1.2)
                    self.thresholds["moderate_trend_threshold"] = max(2.5, trend_p75 * 0.8)
                    self.thresholds["weak_trend_threshold"] = max(1.5, trend_p75 * 0.4)
                    self.thresholds["high_volatility_threshold"] = vol_p75 * 0.8
                    self.thresholds["extreme_volatility_threshold"] = vol_p75 * 1.2

                elif volatility_regime == "low":
                    # In low volatility, be more sensitive to smaller signals
                    self.thresholds["strong_trend_threshold"] = max(2.0, trend_p75 * 0.8)
                    self.thresholds["moderate_trend_threshold"] = max(1.5, trend_p75 * 0.6)
                    self.thresholds["weak_trend_threshold"] = max(0.8, trend_p75 * 0.3)
                    self.thresholds["consolidation_range_threshold"] = vol_p25 * 0.8

                else:  # normal volatility
                    # Reset to baseline with slight adaptation
                    base_strong = 3.0
                    base_moderate = 2.0
                    base_weak = 1.0

                    self.thresholds["strong_trend_threshold"] = base_strong * (1 + (trend_p75 - 2.0) * 0.1)
                    self.thresholds["moderate_trend_threshold"] = base_moderate * (1 + (trend_p75 - 1.5) * 0.1)
                    self.thresholds["weak_trend_threshold"] = base_weak * (1 + (trend_p75 - 1.0) * 0.1)

                logger.debug(
                    f"Adapted thresholds for {volatility_regime} volatility regime: "
                    f"strong={self.thresholds['strong_trend_threshold']:.2f}, "
                    f"moderate={self.thresholds['moderate_trend_threshold']:.2f}, "
                    f"weak={self.thresholds['weak_trend_threshold']:.2f}"
                )

        except Exception as e:
            logger.warning(f"Error adapting thresholds: {e}")
            # Keep default thresholds on error

    def _calculate_regime_stability_scores(
        self, transition_probabilities: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """レジーム安定性スコア計算"""
        stability_scores = {}
        for regime, probabilities in transition_probabilities.items():
            # 自己遷移確率が高いほど安定性が高い
            self_transition_prob = probabilities.get(regime, 0.0)
            stability_scores[regime] = self_transition_prob

        return stability_scores
