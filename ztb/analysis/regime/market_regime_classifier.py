"""
Generic Market Regime Classification System

This module provides a flexible, configurable market regime classification system
that can be used across different trading models and strategies.

Supports multiple regime classification schemes and adaptive parameter tuning.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ztb.analysis.regime.market_regime_types import MarketRegime
from ztb.metrics.technical import calculate_adx
from ztb.utils.error_utils import safe_execute

logger = logging.getLogger(__name__)

# Backward-compatible alias: RegimeType → MarketRegime
# Both share the same string values; short names (e.g. STRONG_BULL) are available
# via MarketRegime's backward-compat aliases defined in market_regime_types.py.
RegimeType = MarketRegime


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
class RegimeDefinition:
    """Definition of a market regime with classification rules"""

    name: str
    regime_type: RegimeType
    conditions: Dict[str, Any]
    priority: int = 1
    description: str = ""


class MarketRegimeClassifier:
    """
    Generic market regime classifier with configurable regime definitions

    This classifier provides flexible regime detection that can be adapted
    for different market conditions and trading strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the regime classifier

        Args:
            config: Configuration dictionary with regime parameters
        """
        self.config = config or self._get_default_config()
        self.lookback_periods = self.config.get(
            "lookback_periods", {"short": 20, "medium": 50, "long": 100}
        )

        # Thresholds for regime detection
        default_thresholds = self._get_default_thresholds()
        config_thresholds = self.config.get("thresholds", {})
        self.thresholds = {**default_thresholds, **config_thresholds}

        # Initialize regime definitions
        self.regime_definitions = self._load_regime_definitions()

        logger.info(
            f"Market Regime Classifier initialized with {len(self.regime_definitions)} regime definitions"
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "lookback_periods": {"short": 20, "medium": 50, "long": 100},
            "regime_scheme": "comprehensive",  # Options: 'basic', 'comprehensive', 'custom'
            "use_multi_timeframe": True,
            "confidence_threshold": 0.6,
        }

    def _get_default_thresholds(self) -> Dict[str, float]:
        """Get default threshold values"""
        return {
            "strong_trend_threshold": 3.0,
            "moderate_trend_threshold": 2.0,
            "weak_trend_threshold": 1.0,
            "high_volatility_threshold": 0.0005,  # Lowered for 1m data
            "moderate_volatility_threshold": 0.0002,
            "extreme_volatility_threshold": 0.0010,
            "consolidation_range_threshold": 0.0001,
            "breakout_setup_threshold": 0.0005,
        }

    def _load_regime_definitions(self) -> List[RegimeDefinition]:
        """Load regime definitions based on configuration"""
        scheme = self.config.get("regime_scheme", "comprehensive")

        if scheme == "basic":
            return self._get_basic_regime_definitions()
        elif scheme == "comprehensive":
            return self._get_comprehensive_regime_definitions()
        else:
            # Custom regime definitions from config
            return self._load_custom_regime_definitions()

    def _get_basic_regime_definitions(self) -> List[RegimeDefinition]:
        """Get basic 4-regime definitions"""
        return [
            RegimeDefinition(
                name="Strong Bull Trend",
                regime_type=RegimeType.STRONG_BULL,
                conditions={
                    "trend_strength": {"min": 3.0},
                    "bull_strength": {"min": 2.0},
                },
                priority=10,
                description="Strong upward trend",
            ),
            RegimeDefinition(
                name="Strong Bear Trend",
                regime_type=RegimeType.STRONG_BEAR,
                conditions={
                    "trend_strength": {"max": -3.0},
                    "bear_strength": {"min": 2.0},
                },
                priority=10,
                description="Strong downward trend",
            ),
            RegimeDefinition(
                name="High Volatility Range",
                regime_type=RegimeType.HIGH_VOLATILITY_RANGE,
                conditions={
                    "volatility": {"min": 0.15},
                    "trend_strength": {"min": -2.0, "max": 2.0},
                },
                priority=5,
                description="High volatility sideways movement",
            ),
            RegimeDefinition(
                name="Consolidation",
                regime_type=RegimeType.CONSOLIDATION,
                conditions={
                    "volatility": {"max": 0.05},
                    "trend_strength": {"min": -1.0, "max": 1.0},
                },
                priority=1,
                description="Low volatility consolidation",
            ),
        ]

    def _get_comprehensive_regime_definitions(self) -> List[RegimeDefinition]:
        """Get comprehensive 16-regime definitions (Symmetric for BUY/SELL balance)"""
        t = self.thresholds

        return [
            # BUY Specialized Regimes (High Priority)
            RegimeDefinition(
                name="Buy Breakout",
                regime_type=RegimeType.BUY_BREAKOUT,
                conditions={
                    "price_range_ratio": {"min": t["breakout_setup_threshold"]},
                    "bull_strength": {"min": 1.5},
                    "volatility": {"min": t["moderate_volatility_threshold"]},
                },
                priority=20,
                description="Strong upward breakout from range",
            ),
            RegimeDefinition(
                name="Buy Divergence",
                regime_type=RegimeType.BUY_DIVERGENCE,
                conditions={
                    "rsi": {"max": 40},
                    "bull_strength": {"min": 1.0},
                    "momentum": {"min": 0.5},
                },
                priority=19,
                description="Bullish divergence pattern",
            ),
            RegimeDefinition(
                name="Buy Momentum Strong",
                regime_type=RegimeType.BUY_MOMENTUM_STRONG,
                conditions={
                    "adx": {"min": 25},
                    "bull_strength": {"min": 2.0},
                    "momentum": {"min": 1.0},
                },
                priority=18,
                description="Strong bullish momentum",
            ),
            RegimeDefinition(
                name="Buy Volume Surge",
                regime_type=RegimeType.BUY_VOLUME_SURGE,
                conditions={
                    "volume_trend": {"min": 1.5},
                    "bull_strength": {"min": 1.0},
                    "price_range_ratio": {"min": t["consolidation_range_threshold"]},
                },
                priority=17,
                description="High volume bullish move",
            ),
            # SELL Specialized Regimes (High Priority)
            RegimeDefinition(
                name="Sell Breakdown",
                regime_type=RegimeType.SELL_BREAKDOWN,
                conditions={
                    "price_range_ratio": {"min": t["breakout_setup_threshold"]},
                    "bear_strength": {"min": 1.5},
                    "volatility": {"min": t["moderate_volatility_threshold"]},
                },
                priority=20,
                description="Strong downward breakdown from range",
            ),
            RegimeDefinition(
                name="Sell Divergence",
                regime_type=RegimeType.SELL_DIVERGENCE,
                conditions={
                    "rsi": {"min": 60},
                    "bear_strength": {"min": 1.0},
                    "momentum": {"max": -0.5},
                },
                priority=19,
                description="Bearish divergence pattern",
            ),
            RegimeDefinition(
                name="Sell Momentum Weak",
                regime_type=RegimeType.SELL_MOMENTUM_WEAK,
                conditions={
                    "adx": {"min": 25},
                    "bear_strength": {"min": 2.0},
                    "momentum": {"max": -1.0},
                },
                priority=18,
                description="Strong bearish momentum",
            ),
            RegimeDefinition(
                name="Sell Volume Surge",
                regime_type=RegimeType.SELL_VOLUME_SURGE,
                conditions={
                    "volume_trend": {"min": 1.5},
                    "bear_strength": {"min": 1.0},
                    "price_range_ratio": {"min": t["consolidation_range_threshold"]},
                },
                priority=17,
                description="High volume bearish move",
            ),
            # Bull trends
            RegimeDefinition(
                name="Strong Bull Trend",
                regime_type=RegimeType.STRONG_BULL,
                conditions={
                    "trend_strength": {"min": t["strong_trend_threshold"]},
                    "bull_strength": {"min": 2.5},
                    "volatility": {"max": t["high_volatility_threshold"]},
                },
                priority=12,
                description="Strong upward momentum with high conviction",
            ),
            RegimeDefinition(
                name="Moderate Bull Trend",
                regime_type=RegimeType.MODERATE_BULL,
                conditions={
                    "trend_strength": {
                        "min": t["moderate_trend_threshold"],
                        "max": t["strong_trend_threshold"],
                    },
                    "bull_strength": {"min": 1.5, "max": 2.5},
                    "volatility": {"max": t["extreme_volatility_threshold"]},
                },
                priority=11,
                description="Moderate upward trend with steady gains",
            ),
            RegimeDefinition(
                name="Weak Bull Trend",
                regime_type=RegimeType.WEAK_BULL,
                conditions={
                    "trend_strength": {
                        "min": t["weak_trend_threshold"],
                        "max": t["moderate_trend_threshold"],
                    },
                    "bull_strength": {"min": 0.5, "max": 1.5},
                    "volatility": {"max": t["extreme_volatility_threshold"]},
                },
                priority=10,
                description="Weak upward movement with low momentum",
            ),
            # Bear trends
            RegimeDefinition(
                name="Strong Bear Trend",
                regime_type=RegimeType.STRONG_BEAR,
                conditions={
                    "trend_strength": {"max": -t["strong_trend_threshold"]},
                    "bear_strength": {"min": 2.2},
                    "volatility": {"max": t["high_volatility_threshold"]},
                },
                priority=12,
                description="Strong downward momentum with high conviction",
            ),
            RegimeDefinition(
                name="Moderate Bear Trend",
                regime_type=RegimeType.MODERATE_BEAR,
                conditions={
                    "trend_strength": {
                        "max": -t["moderate_trend_threshold"],
                        "min": -t["strong_trend_threshold"],
                    },
                    "bear_strength": {"min": 1.3, "max": 2.2},
                    "volatility": {"max": t["extreme_volatility_threshold"]},
                },
                priority=11,
                description="Moderate downward trend with steady losses",
            ),
            RegimeDefinition(
                name="Weak Bear Trend",
                regime_type=RegimeType.WEAK_BEAR,
                conditions={
                    "trend_strength": {
                        "max": -t["weak_trend_threshold"],
                        "min": -t["moderate_trend_threshold"],
                    },
                    "bear_strength": {"min": 0.3, "max": 1.3},
                    "volatility": {"max": t["extreme_volatility_threshold"]},
                },
                priority=10,
                description="Weak downward movement with low momentum",
            ),
            # Range conditions
            RegimeDefinition(
                name="High Volatility Range",
                regime_type=RegimeType.HIGH_VOLATILITY_RANGE,
                conditions={
                    "volatility": {"min": t["high_volatility_threshold"]},
                    "trend_strength": {
                        "min": -t["moderate_trend_threshold"],
                        "max": t["moderate_trend_threshold"],
                    },
                },
                priority=6,
                description="High volatility sideways movement",
            ),
            RegimeDefinition(
                name="Moderate Volatility Range",
                regime_type=RegimeType.MODERATE_VOLATILITY_RANGE,
                conditions={
                    "volatility": {
                        "min": t["moderate_volatility_threshold"],
                        "max": t["high_volatility_threshold"],
                    },
                    "trend_strength": {
                        "min": -t["weak_trend_threshold"],
                        "max": t["weak_trend_threshold"],
                    },
                },
                priority=5,
                description="Moderate volatility consolidation",
            ),
            RegimeDefinition(
                name="Low Volatility Range",
                regime_type=RegimeType.LOW_VOLATILITY_RANGE,
                conditions={
                    "volatility": {"max": t["moderate_volatility_threshold"]},
                    "trend_strength": {
                        "min": -t["weak_trend_threshold"],
                        "max": t["weak_trend_threshold"],
                    },
                },
                priority=4,
                description="Low volatility tight range",
            ),
            # Special conditions
            RegimeDefinition(
                name="Extreme Volatility",
                regime_type=RegimeType.EXTREME_VOLATILITY,
                conditions={"volatility": {"min": t["extreme_volatility_threshold"]}},
                priority=3,
                description="Extreme market volatility conditions",
            ),
            RegimeDefinition(
                name="Consolidation",
                regime_type=RegimeType.CONSOLIDATION,
                conditions={
                    "volatility": {"max": t["consolidation_range_threshold"]},
                    "trend_strength": {"min": -0.8, "max": 0.8},
                },
                priority=2,
                description="Tight consolidation with minimal movement",
            ),
            RegimeDefinition(
                name="Breakout Setup",
                regime_type=RegimeType.BREAKOUT_SETUP,
                conditions={
                    "volatility": {"max": t["breakout_setup_threshold"]},
                    "trend_strength": {"min": -1.2, "max": 1.2},
                    "support_resistance_strength": {"min": 0.7},
                },
                priority=1,
                description="Potential breakout from consolidation",
            ),
            RegimeDefinition(
                name="Breakdown Setup",
                regime_type=RegimeType.BREAKDOWN_SETUP,
                conditions={
                    "volatility": {"max": t["breakout_setup_threshold"]},
                    "trend_strength": {"min": -1.2, "max": 1.2},
                    "support_resistance_strength": {"min": 0.7},
                },
                priority=1,
                description="Potential breakdown from consolidation",
            ),
        ]

    def _load_custom_regime_definitions(self) -> List[RegimeDefinition]:
        """Load custom regime definitions from config"""
        custom_defs = self.config.get("custom_regime_definitions", [])
        regime_definitions = []

        for def_config in custom_defs:
            regime_definitions.append(
                RegimeDefinition(
                    name=def_config["name"],
                    regime_type=RegimeType(def_config["regime_type"]),
                    conditions=def_config["conditions"],
                    priority=def_config.get("priority", 1),
                    description=def_config.get("description", ""),
                )
            )

    def _load_custom_regime_definitions(self) -> List[RegimeDefinition]:
        """Load custom regime definitions from config"""
        custom_defs = self.config.get("custom_regime_definitions", [])
        regime_definitions = []

        for def_config in custom_defs:
            regime_definitions.append(
                RegimeDefinition(
                    name=def_config["name"],
                    regime_type=RegimeType(def_config["regime_type"]),
                    conditions=def_config["conditions"],
                    priority=def_config.get("priority", 1),
                    description=def_config.get("description", ""),
                )
            )

        return regime_definitions

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
        """
        if current_index == -1:
            current_index = len(data) - 1

        # Calculate regime metrics
        metrics = self._calculate_regime_metrics(data, current_index)

        # Determine primary regime
        primary_regime, confidence = self._classify_regime(metrics)

        # Calculate secondary regimes
        secondary_regimes = self._calculate_secondary_regimes(metrics, primary_regime)

        return RegimeDetectionResult(
            primary_regime=primary_regime,
            confidence=confidence,
            secondary_regimes=secondary_regimes,
            metrics=metrics,
            detection_timestamp=data.index[current_index],
            lookback_period=self.lookback_periods["medium"],
        )

    def get_regime_multiplier(
        self, regime: RegimeType, multiplier_type: str = "reward"
    ) -> float:
        """
        Get regime-specific multiplier for rewards/penalties

        Args:
            regime: Market regime
            multiplier_type: 'reward' or 'penalty'

        Returns:
            Multiplier value (default 1.0)
        """
        adaptation_config = self.config.get("adaptation", {})
        if not adaptation_config.get("enabled", False):
            return 1.0

        if multiplier_type == "reward":
            multipliers = adaptation_config.get("regime_reward_multipliers", {})
            # Try direct lookup (if key is Enum) or string lookup
            if regime in multipliers:
                return multipliers[regime]
            return multipliers.get(
                regime.value if hasattr(regime, "value") else str(regime), 1.0
            )
        elif multiplier_type == "penalty":
            multipliers = adaptation_config.get("regime_penalty_multipliers", {})
            if regime in multipliers:
                return multipliers[regime]
            return multipliers.get(
                regime.value if hasattr(regime, "value") else str(regime), 1.0
            )

        return 1.0

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

        # Calculate all metrics
        trend_strength, bull_strength, bear_strength = self._calculate_trend_strength(
            high, low, close
        )
        volatility = self._calculate_volatility(close)
        momentum = self._calculate_momentum(close)
        volume_trend = self._calculate_volume_trend(volume)
        price_range_ratio = self._calculate_price_range_ratio(high, low, close)
        adx = self._calculate_adx(high, low, close)
        rsi = self._calculate_rsi(close)
        macd_signal = self._calculate_macd_signal(close)
        bollinger_position = self._calculate_bollinger_position(close)
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

    def _get_default_metrics(self) -> RegimeMetrics:
        """Get default metrics when insufficient data"""
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
            support_resistance_strength=0.0,
        )

    def _calculate_trend_strength(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> Tuple[float, float, float]:
        """Calculate trend strength with directional components"""
        def calculate_trend():
            # Ensure we have enough data
            if len(high) < 14:
                return 0.0, 0.0, 0.0

            # Calculate price momentum over 10 periods
            momentum = close / close.shift(10) - 1

            # Calculate volatility (standard deviation of log returns)
            log_returns = np.log(close / close.shift())
            from ztb.metrics.technical import calculate_rolling_volatility

            volatility = calculate_rolling_volatility(
                log_returns, window=10, annualize=True, trading_days=10
            )

            # Calculate raw trend strength
            trend_strength_raw = momentum / (volatility + 1e-8)

            # Smooth the trend strength
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

            return result_trend, result_bull, result_bear

        return safe_execute(calculate_trend, operation_name="Error calculating trend strength", default_result=(0.0, 0.0, 0.0))

    def _calculate_volatility(self, close: pd.Series) -> float:
        """Calculate normalized volatility"""
        def calculate_vol():
            from ztb.metrics.technical import calculate_volatility_from_returns

            returns = close.pct_change().fillna(0)
            # Original logic: returns.rolling(20).std() * 10
            # We can use calculate_volatility_from_returns(returns, 20) * 10
            vol = calculate_volatility_from_returns(returns, 20)
            return vol * 10

        return safe_execute(calculate_vol, operation_name="Error calculating volatility", default_result=0.0)

    def _calculate_momentum(self, close: pd.Series) -> float:
        """Calculate momentum indicator"""
        def calculate_mom():
            roc = (close - close.shift(10)) / close.shift(10)
            momentum = roc.rolling(5).mean()

            return float(momentum.iloc[-1]) if not momentum.empty else 0.0

        return safe_execute(calculate_mom, operation_name="Error calculating momentum", default_result=0.0)

    def _calculate_volume_trend(self, volume: pd.Series) -> float:
        """Calculate volume trend"""
        def calculate_vol_trend():
            volume_ma_short = volume.rolling(10).mean()
            volume_ma_long = volume.rolling(30).mean()
            volume_trend = (volume_ma_short - volume_ma_long) / volume_ma_long

            return float(volume_trend.iloc[-1]) if not volume_trend.empty else 0.0

        return safe_execute(calculate_vol_trend, operation_name="Error calculating volume trend", default_result=0.0)

    def _calculate_price_range_ratio(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> float:
        """Calculate price range ratio"""
        def calculate_range_ratio():
            price_range = (high - low) / close.shift(1)
            range_ratio = price_range.rolling(10).mean()

            return float(range_ratio.iloc[-1]) if not range_ratio.empty else 0.0

        return safe_execute(calculate_range_ratio, operation_name="Error calculating price range ratio", default_result=0.0)

    def _calculate_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> float:
        """Calculate ADX (Average Directional Index)"""
        return calculate_adx(high, low, close, period=14)

    def _calculate_rsi(self, close: pd.Series) -> float:
        """Calculate RSI (Relative Strength Index)"""
        from ztb.metrics.technical import calculate_rsi

        return calculate_rsi(close, period=14)

    def _calculate_macd_signal(self, close: pd.Series) -> float:
        """Calculate MACD signal"""
        try:
            from ztb.metrics.technical import calculate_macd

            _, _, hist = calculate_macd(close)
            return float(hist.iloc[-1]) if not hist.empty else 0.0

        except Exception as e:
            logger.warning(f"Error calculating MACD signal: {e}")
            return 0.0

    def _calculate_bollinger_position(self, close: pd.Series) -> float:
        """Calculate Bollinger Band position"""
        try:
            from ztb.metrics.technical import calculate_bollinger_bands

            upper_band, _, lower_band = calculate_bollinger_bands(close, period=20)

            band_width = upper_band - lower_band
            position = (close - lower_band) / band_width

            return float(position.iloc[-1]) if not position.empty else 0.5

        except Exception as e:
            logger.warning(f"Error calculating Bollinger position: {e}")
            return 0.5

    def _calculate_support_resistance_strength(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> float:
        """Calculate support/resistance strength"""
        try:
            # Simplified calculation - distance from recent highs/lows
            recent_high = high.rolling(20).max()
            recent_low = low.rolling(20).min()

            strength = 1 - ((close - recent_low) / (recent_high - recent_low))

            return float(strength.iloc[-1]) if not strength.empty else 0.0

        except Exception as e:
            logger.warning(f"Error calculating support/resistance strength: {e}")
            return 0.0

    def _classify_regime(self, metrics: RegimeMetrics) -> Tuple[RegimeType, float]:
        """
        Classify regime based on metrics and regime definitions

        Args:
            metrics: Calculated regime metrics

        Returns:
            Tuple of (regime_type, confidence)
        """
        # Convert metrics to dict for easier condition checking
        metrics_dict = {
            "trend_strength": metrics.trend_strength,
            "bull_strength": metrics.bull_strength,
            "bear_strength": metrics.bear_strength,
            "volatility": metrics.volatility,
            "momentum": metrics.momentum,
            "volume_trend": metrics.volume_trend,
            "price_range_ratio": metrics.price_range_ratio,
            "adx": metrics.adx,
            "rsi": metrics.rsi,
            "macd_signal": metrics.macd_signal,
            "bollinger_position": metrics.bollinger_position,
            "support_resistance_strength": metrics.support_resistance_strength,
        }

        # Evaluate each regime definition
        best_regime = None
        best_confidence = 0.0
        best_score = 0

        for regime_def in self.regime_definitions:
            score, confidence = self._evaluate_regime_conditions(
                metrics_dict, regime_def
            )

            # Use priority as tiebreaker
            if confidence > best_confidence or (
                confidence == best_confidence and regime_def.priority > best_score
            ):
                best_regime = regime_def.regime_type
                best_confidence = confidence
                best_score = regime_def.priority

        # Default to consolidation if no regime matches well
        if best_confidence < self.config.get("confidence_threshold", 0.6):
            best_regime = RegimeType.CONSOLIDATION
            best_confidence = 0.5

        return best_regime, best_confidence

    def _evaluate_regime_conditions(
        self, metrics: Dict[str, float], regime_def: RegimeDefinition
    ) -> Tuple[int, float]:
        """
        Evaluate how well metrics match regime conditions

        Args:
            metrics: Dictionary of metric values
            regime_def: Regime definition to evaluate

        Returns:
            Tuple of (score, confidence)
        """
        score = 0
        total_conditions = len(regime_def.conditions)
        matched_conditions = 0

        for metric_name, conditions in regime_def.conditions.items():
            if metric_name not in metrics:
                continue

            metric_value = metrics[metric_name]
            condition_met = True

            # Check min/max conditions
            if "min" in conditions and metric_value < conditions["min"]:
                condition_met = False
            if "max" in conditions and metric_value > conditions["max"]:
                condition_met = False

            if condition_met:
                matched_conditions += 1
                score += 1

        confidence = (
            matched_conditions / total_conditions if total_conditions > 0 else 0.0
        )

        return score, confidence

    def _calculate_secondary_regimes(
        self, metrics: RegimeMetrics, primary_regime: RegimeType
    ) -> List[Tuple[RegimeType, float]]:
        """
        Calculate secondary regime candidates

        Args:
            metrics: Regime metrics
            primary_regime: Primary detected regime

        Returns:
            List of (regime_type, confidence) tuples for secondary regimes
        """
        # Convert metrics to dict
        metrics_dict = {
            "trend_strength": metrics.trend_strength,
            "bull_strength": metrics.bull_strength,
            "bear_strength": metrics.bear_strength,
            "volatility": metrics.volatility,
            "momentum": metrics.momentum,
            "volume_trend": metrics.volume_trend,
            "price_range_ratio": metrics.price_range_ratio,
            "adx": metrics.adx,
            "rsi": metrics.rsi,
            "macd_signal": metrics.macd_signal,
            "bollinger_position": metrics.bollinger_position,
            "support_resistance_strength": metrics.support_resistance_strength,
        }

        secondary_regimes = []

        for regime_def in self.regime_definitions:
            if regime_def.regime_type == primary_regime:
                continue

            score, confidence = self._evaluate_regime_conditions(
                metrics_dict, regime_def
            )

            if confidence > 0.3:  # Lower threshold for secondary regimes
                secondary_regimes.append((regime_def.regime_type, confidence))

        # Sort by confidence and return top 3
        secondary_regimes.sort(key=lambda x: x[1], reverse=True)
        return secondary_regimes[:3]

    def get_regime_definitions(self) -> List[RegimeDefinition]:
        """Get current regime definitions"""
        return self.regime_definitions.copy()

    def update_regime_definitions(self, new_definitions: List[RegimeDefinition]):
        """Update regime definitions"""
        self.regime_definitions = new_definitions
        logger.info(f"Updated regime definitions: {len(new_definitions)} regimes")

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()

    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration"""
        self.config.update(new_config)
        self.thresholds = self.config.get("thresholds", self._get_default_thresholds())
        self.lookback_periods = self.config.get(
            "lookback_periods", {"short": 20, "medium": 50, "long": 100}
        )
        logger.info("Configuration updated")
