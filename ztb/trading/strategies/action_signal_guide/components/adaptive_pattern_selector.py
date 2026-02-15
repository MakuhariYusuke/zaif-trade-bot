"""
Adaptive Pattern Selector for Action Signal Guide.

This component dynamically selects and activates patterns based on:
- Market regime detection
- Historical pattern performance
- Signal quality metrics
- Computational resource constraints
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List, Mapping, Optional, Set, Tuple, TypedDict

import numpy as np

from ztb.analysis.regime.market_regime_types import MarketRegime
from ztb.utils.logging_utils import get_logger


class PatternCategory(Enum):
    """Pattern categories for adaptive selection."""

    TREND = "trend"
    OSCILLATOR = "oscillator"
    VOLUME = "volume"
    CANDLESTICK = "candlestick"
    HARMONIC = "harmonic"
    FIBONACCI = "fibonacci"
    WAVE = "wave"
    GANN = "gann"


@dataclass
class PatternPerformance:
    """Performance metrics for a pattern."""

    pattern_name: str
    category: PatternCategory
    success_rate: float
    avg_strength: float
    avg_confidence: float
    execution_time: float
    memory_usage: float
    last_used: float
    usage_count: int


@dataclass
class MarketCondition:
    """Current market condition assessment."""

    regime: MarketRegime
    volatility: float
    trend_strength: float
    volume_trend: float
    timestamp: float


class PatternPerformanceSample(TypedDict):
    success: float
    strength: float
    confidence: float
    execution_time: float
    memory_usage: float
    timestamp: float


class AdaptivePatternSelector:
    """
    Dynamically selects optimal patterns based on market conditions and performance.

    This class implements:
    - Market regime-based pattern selection
    - Performance-weighted pattern activation
    - Computational resource management
    - Adaptive threshold adjustment
    """

    def __init__(self, config: Mapping[str, object]):
        """
        Initialize adaptive pattern selector.

        Args:
            config: Configuration dictionary
        """
        self.logger = get_logger("ztb.trading.strategies.adaptive_selector")
        self.config = dict(config)

        # Performance tracking
        self.pattern_performance: Dict[str, PatternPerformance] = {}
        self.performance_history: Dict[str, Deque[PatternPerformanceSample]] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Market condition tracking
        self.market_history: Deque[MarketCondition] = deque(maxlen=100)
        self.current_condition: Optional[MarketCondition] = None

        # Selection parameters
        self.min_success_rate = self._coerce_float(config.get("min_success_rate"), 0.4)
        self.max_patterns_per_category = self._coerce_int(
            config.get("max_patterns_per_category"), 2
        )
        self.performance_decay_factor = self._coerce_float(
            config.get("performance_decay_factor"), 0.95
        )
        self.adaptation_interval = self._coerce_float(
            config.get("adaptation_interval"), 300.0
        )  # 5 minutes

        # Resource constraints
        self.max_execution_time = self._coerce_float(
            config.get("max_execution_time"), 1.0
        )  # seconds
        self.max_memory_usage = self._coerce_float(
            config.get("max_memory_usage"), 100.0
        )  # MB

        # Pattern category mappings
        self.pattern_categories = self._initialize_pattern_categories()

        # Adaptive thresholds
        self.success_thresholds = self._initialize_success_thresholds()

        self.last_adaptation = time.time()
        self.logger.info("AdaptivePatternSelector initialized")

    @staticmethod
    def _coerce_float(value: object, default: float) -> float:
        """Coerce config values to float with safe fallback."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int(value: object, default: int) -> int:
        """Coerce config values to int with safe fallback."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _initialize_pattern_categories(self) -> Dict[str, PatternCategory]:
        """Initialize pattern to category mappings."""
        return {
            # Trend patterns
            "adx": PatternCategory.TREND,
            "ADXRecognizer": PatternCategory.TREND,
            "dow_theory": PatternCategory.TREND,
            "DowTheoryRecognizer": PatternCategory.TREND,
            "hierarchical_trend": PatternCategory.TREND,
            "heikin_ashi": PatternCategory.TREND,
            "HeikinAshiRecognizer": PatternCategory.TREND,
            # Oscillator patterns
            "rsi": PatternCategory.OSCILLATOR,
            "RSIPatternRecognizer": PatternCategory.OSCILLATOR,
            "macd": PatternCategory.OSCILLATOR,
            "MACDPatternRecognizer": PatternCategory.OSCILLATOR,
            "stochastic": PatternCategory.OSCILLATOR,
            "StochasticRecognizer": PatternCategory.OSCILLATOR,
            "cci": PatternCategory.OSCILLATOR,
            "CCIRecognizer": PatternCategory.OSCILLATOR,
            "williams_r": PatternCategory.OSCILLATOR,
            "WilliamsRRecognizer": PatternCategory.OSCILLATOR,
            "mfi": PatternCategory.OSCILLATOR,
            "MFIRecognizer": PatternCategory.OSCILLATOR,
            "bollinger_bands": PatternCategory.OSCILLATOR,
            "BollingerBandsRecognizer": PatternCategory.OSCILLATOR,
            # Volume patterns
            "volume_price_trend": PatternCategory.VOLUME,
            "chaikin_ad": PatternCategory.VOLUME,
            "ChaikinADRecognizer": PatternCategory.VOLUME,
            "ease_of_movement": PatternCategory.VOLUME,
            "granville_law": PatternCategory.VOLUME,
            "GranvilleLawRecognizer": PatternCategory.VOLUME,
            # Candlestick patterns
            "doji": PatternCategory.CANDLESTICK,
            "hammer": PatternCategory.CANDLESTICK,
            "HammerRecognizer": PatternCategory.CANDLESTICK,
            "engulfing": PatternCategory.CANDLESTICK,
            "BearishEngulfingRecognizer": PatternCategory.CANDLESTICK,
            "BullishEngulfingRecognizer": PatternCategory.CANDLESTICK,
            "morning_star": PatternCategory.CANDLESTICK,
            "MorningStarRecognizer": PatternCategory.CANDLESTICK,
            "evening_star": PatternCategory.CANDLESTICK,
            "EveningStarRecognizer": PatternCategory.CANDLESTICK,
            "hanging_man": PatternCategory.CANDLESTICK,
            "HangingManRecognizer": PatternCategory.CANDLESTICK,
            "piercing_pattern": PatternCategory.CANDLESTICK,
            "PiercingPatternRecognizer": PatternCategory.CANDLESTICK,
            "rising_three_methods": PatternCategory.CANDLESTICK,
            "RisingThreeMethodsRecognizer": PatternCategory.CANDLESTICK,
            "sakata_five_methods": PatternCategory.CANDLESTICK,
            "SakataFiveMethodsRecognizer": PatternCategory.CANDLESTICK,
            "three_black_crows": PatternCategory.CANDLESTICK,
            "ThreeBlackCrowsRecognizer": PatternCategory.CANDLESTICK,
            "three_white_soldiers": PatternCategory.CANDLESTICK,
            "ThreeWhiteSoldiersRecognizer": PatternCategory.CANDLESTICK,
            # Harmonic patterns (high cost)
            "gartley": PatternCategory.HARMONIC,
            "GartleyRecognizer": PatternCategory.HARMONIC,
            "butterfly": PatternCategory.HARMONIC,
            "ButterflyRecognizer": PatternCategory.HARMONIC,
            "bat": PatternCategory.HARMONIC,
            "BatRecognizer": PatternCategory.HARMONIC,
            "crab": PatternCategory.HARMONIC,
            "CrabRecognizer": PatternCategory.HARMONIC,
            # Fibonacci patterns
            "fibonacci_retracement": PatternCategory.FIBONACCI,
            "FibonacciRetracementRecognizer": PatternCategory.FIBONACCI,
            "fibonacci_extension": PatternCategory.FIBONACCI,
            "FibonacciExtensionRecognizer": PatternCategory.FIBONACCI,
            "fibonacci_projection": PatternCategory.FIBONACCI,
            "FibonacciProjectionRecognizer": PatternCategory.FIBONACCI,
            # Wave patterns (high cost)
            "elliott_wave": PatternCategory.WAVE,
            "impulse_wave": PatternCategory.WAVE,
            "ImpulseWaveRecognizer": PatternCategory.WAVE,
            "corrective_wave": PatternCategory.WAVE,
            "CorrectiveWaveRecognizer": PatternCategory.WAVE,
            "wave_extension": PatternCategory.WAVE,
            "WaveExtensionRecognizer": PatternCategory.WAVE,
            "wave_i": PatternCategory.WAVE,
            "WaveIRecognizer": PatternCategory.WAVE,
            "wave_v": PatternCategory.WAVE,
            "WaveVRecognizer": PatternCategory.WAVE,
            "wave_y": PatternCategory.WAVE,
            "WaveYRecognizer": PatternCategory.WAVE,
            "wave_p": PatternCategory.WAVE,
            "WavePRecognizer": PatternCategory.WAVE,
            "wave_n": PatternCategory.WAVE,
            "WaveNRecognizer": PatternCategory.WAVE,
            "wave_s": PatternCategory.WAVE,
            "WaveSRecognizer": PatternCategory.WAVE,
            # Gann patterns (high cost)
            "gann_square": PatternCategory.GANN,
            "GannSquareRecognizer": PatternCategory.GANN,
            "gann_fan": PatternCategory.GANN,
            "gann_angle": PatternCategory.GANN,
            "GannAngleRecognizer": PatternCategory.GANN,
            "gann_time_cluster": PatternCategory.GANN,
            "GannTimeClusterRecognizer": PatternCategory.GANN,
        }

    def _initialize_success_thresholds(
        self,
    ) -> Dict[MarketRegime, Dict[PatternCategory, float]]:
        """Initialize success rate thresholds by regime and category."""
        return {
            MarketRegime.STRONG_BULL_TREND: {
                PatternCategory.TREND: 0.6,
                PatternCategory.OSCILLATOR: 0.3,
                PatternCategory.VOLUME: 0.4,
                PatternCategory.CANDLESTICK: 0.4,
                PatternCategory.HARMONIC: 0.7,
                PatternCategory.FIBONACCI: 0.6,
                PatternCategory.WAVE: 0.7,
                PatternCategory.GANN: 0.7,
            },
            MarketRegime.MODERATE_BULL_TREND: {
                PatternCategory.TREND: 0.5,
                PatternCategory.OSCILLATOR: 0.3,
                PatternCategory.VOLUME: 0.4,
                PatternCategory.CANDLESTICK: 0.4,
                PatternCategory.HARMONIC: 0.6,
                PatternCategory.FIBONACCI: 0.5,
                PatternCategory.WAVE: 0.6,
                PatternCategory.GANN: 0.6,
            },
            MarketRegime.STRONG_BEAR_TREND: {
                PatternCategory.TREND: 0.6,
                PatternCategory.OSCILLATOR: 0.3,
                PatternCategory.VOLUME: 0.4,
                PatternCategory.CANDLESTICK: 0.4,
                PatternCategory.HARMONIC: 0.7,
                PatternCategory.FIBONACCI: 0.6,
                PatternCategory.WAVE: 0.7,
                PatternCategory.GANN: 0.7,
            },
            MarketRegime.MODERATE_BEAR_TREND: {
                PatternCategory.TREND: 0.5,
                PatternCategory.OSCILLATOR: 0.3,
                PatternCategory.VOLUME: 0.4,
                PatternCategory.CANDLESTICK: 0.4,
                PatternCategory.HARMONIC: 0.6,
                PatternCategory.FIBONACCI: 0.5,
                PatternCategory.WAVE: 0.6,
                PatternCategory.GANN: 0.6,
            },
            MarketRegime.HIGH_VOLATILITY_RANGING: {
                PatternCategory.TREND: 0.2,
                PatternCategory.OSCILLATOR: 0.6,
                PatternCategory.VOLUME: 0.5,
                PatternCategory.CANDLESTICK: 0.5,
                PatternCategory.HARMONIC: 0.4,
                PatternCategory.FIBONACCI: 0.4,
                PatternCategory.WAVE: 0.3,
                PatternCategory.GANN: 0.3,
            },
            MarketRegime.MODERATE_VOLATILITY_RANGING: {
                PatternCategory.TREND: 0.2,
                PatternCategory.OSCILLATOR: 0.5,
                PatternCategory.VOLUME: 0.4,
                PatternCategory.CANDLESTICK: 0.5,
                PatternCategory.HARMONIC: 0.4,
                PatternCategory.FIBONACCI: 0.4,
                PatternCategory.WAVE: 0.3,
                PatternCategory.GANN: 0.3,
            },
            MarketRegime.LOW_VOLATILITY_RANGING: {
                PatternCategory.TREND: 0.2,
                PatternCategory.OSCILLATOR: 0.4,
                PatternCategory.VOLUME: 0.3,
                PatternCategory.CANDLESTICK: 0.4,
                PatternCategory.HARMONIC: 0.3,
                PatternCategory.FIBONACCI: 0.3,
                PatternCategory.WAVE: 0.2,
                PatternCategory.GANN: 0.2,
            },
            MarketRegime.EXTREME_VOLATILITY: {
                PatternCategory.TREND: 0.3,
                PatternCategory.OSCILLATOR: 0.4,
                PatternCategory.VOLUME: 0.5,
                PatternCategory.CANDLESTICK: 0.4,
                PatternCategory.HARMONIC: 0.3,
                PatternCategory.FIBONACCI: 0.4,
                PatternCategory.WAVE: 0.3,
                PatternCategory.GANN: 0.3,
            },
            MarketRegime.CONSOLIDATION: {
                PatternCategory.TREND: 0.2,
                PatternCategory.OSCILLATOR: 0.5,
                PatternCategory.VOLUME: 0.4,
                PatternCategory.CANDLESTICK: 0.5,
                PatternCategory.HARMONIC: 0.4,
                PatternCategory.FIBONACCI: 0.4,
                PatternCategory.WAVE: 0.3,
                PatternCategory.GANN: 0.3,
            },
            MarketRegime.BREAKOUT_SETUP: {
                PatternCategory.TREND: 0.6,
                PatternCategory.OSCILLATOR: 0.4,
                PatternCategory.VOLUME: 0.6,
                PatternCategory.CANDLESTICK: 0.5,
                PatternCategory.HARMONIC: 0.5,
                PatternCategory.FIBONACCI: 0.6,
                PatternCategory.WAVE: 0.5,
                PatternCategory.GANN: 0.5,
            },
            MarketRegime.BREAKDOWN_SETUP: {
                PatternCategory.TREND: 0.6,
                PatternCategory.OSCILLATOR: 0.4,
                PatternCategory.VOLUME: 0.6,
                PatternCategory.CANDLESTICK: 0.5,
                PatternCategory.HARMONIC: 0.5,
                PatternCategory.FIBONACCI: 0.6,
                PatternCategory.WAVE: 0.5,
                PatternCategory.GANN: 0.5,
            },
        }

    def update_pattern_performance(
        self,
        pattern_name: str,
        success: bool,
        strength: float,
        confidence: float,
        execution_time: float,
        memory_usage: float,
    ) -> None:
        """
        Update performance metrics for a pattern.

        Args:
            pattern_name: Name of the pattern
            success: Whether the pattern signal was successful
            strength: Signal strength (0-1)
            confidence: Signal confidence (0-1)
            execution_time: Execution time in seconds
            memory_usage: Memory usage in MB
        """
        current_time = time.time()

        if pattern_name not in self.pattern_performance:
            category = self.pattern_categories.get(pattern_name, PatternCategory.TREND)
            self.pattern_performance[pattern_name] = PatternPerformance(
                pattern_name=pattern_name,
                category=category,
                success_rate=0.5,  # Initial neutral value
                avg_strength=0.5,
                avg_confidence=0.5,
                execution_time=execution_time,
                memory_usage=memory_usage,
                last_used=current_time,
                usage_count=1,
            )

        perf = self.pattern_performance[pattern_name]

        # Update performance history
        self.performance_history[pattern_name].append(
            {
                "success": 1.0 if success else 0.0,
                "strength": strength,
                "confidence": confidence,
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "timestamp": current_time,
            }
        )

        # Recalculate metrics
        history = self.performance_history[pattern_name]
        if history:
            success_sum = 0.0
            strength_sum = 0.0
            confidence_sum = 0.0
            execution_sum = 0.0
            memory_sum = 0.0
            for sample in history:
                success_sum += sample["success"]
                strength_sum += sample["strength"]
                confidence_sum += sample["confidence"]
                execution_sum += sample["execution_time"]
                memory_sum += sample["memory_usage"]

            sample_count = float(len(history))
            perf.success_rate = success_sum / sample_count
            perf.avg_strength = strength_sum / sample_count
            perf.avg_confidence = confidence_sum / sample_count
            perf.execution_time = execution_sum / sample_count
            perf.memory_usage = memory_sum / sample_count

        perf.last_used = current_time
        perf.usage_count += 1

        # Apply performance decay for unused patterns
        self._apply_performance_decay()

    def update_market_condition(
        self,
        regime: MarketRegime,
        volatility: float,
        trend_strength: float,
        volume_trend: float,
    ) -> None:
        """
        Update current market condition.

        Args:
            regime: Current market regime
            volatility: Current volatility level
            trend_strength: Current trend strength
            volume_trend: Current volume trend
        """
        self.current_condition = MarketCondition(
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            volume_trend=volume_trend,
            timestamp=time.time(),
        )

        self.market_history.append(self.current_condition)

    def select_active_patterns(self, available_patterns: List[str]) -> Set[str]:
        """
        Select patterns to activate based on current conditions.

        Args:
            available_patterns: List of available pattern names

        Returns:
            Set of pattern names to activate
        """
        if not self.current_condition:
            # Default selection if no market condition available
            return self._default_pattern_selection(available_patterns)

        # Check if adaptation is needed
        current_time = time.time()
        if current_time - self.last_adaptation > self.adaptation_interval:
            self._adapt_thresholds()
            self.last_adaptation = current_time

        # Select patterns by category priorities
        selected_patterns = set()

        # Get regime-specific thresholds
        regime_thresholds = self.success_thresholds.get(
            self.current_condition.regime,
            self.success_thresholds[MarketRegime.MODERATE_VOLATILITY_RANGING],
        )

        # Select patterns by category
        for category in PatternCategory:
            category_patterns = [
                p
                for p in available_patterns
                if self.pattern_categories.get(p) == category
            ]

            if not category_patterns:
                continue

            # Sort by performance and resource efficiency
            sorted_patterns = self._rank_patterns_by_performance(
                category_patterns, regime_thresholds.get(category, 0.4)
            )

            # Select top patterns within resource constraints
            selected = self._select_within_constraints(
                sorted_patterns, category, self.max_patterns_per_category
            )

            selected_patterns.update(selected)

        # Ensure minimum pattern coverage
        selected_patterns = self._ensure_minimum_coverage(
            selected_patterns, available_patterns
        )

        self.logger.debug(
            f"Selected {len(selected_patterns)} patterns: {sorted(selected_patterns)}"
        )
        return selected_patterns

    def _default_pattern_selection(self, available_patterns: List[str]) -> Set[str]:
        """Default pattern selection when no market condition is available."""
        # Select high-reliability patterns
        reliable_patterns = {
            "rsi",
            "macd",
            "adx",
            "fibonacci_retracement",
            "doji",
            "hammer",
            "volume_price_trend",
        }

        return set(available_patterns) & reliable_patterns

    def _rank_patterns_by_performance(
        self, patterns: List[str], threshold: float
    ) -> List[Tuple[str, float]]:
        """
        Rank patterns by performance score.

        Args:
            patterns: List of pattern names
            threshold: Minimum success rate threshold

        Returns:
            List of (pattern_name, score) tuples, sorted by score descending
        """
        pattern_scores = []

        for pattern in patterns:
            if pattern not in self.pattern_performance:
                # New pattern - give neutral score
                score = 0.5
            else:
                perf = self.pattern_performance[pattern]

                # Skip if below threshold
                if perf.success_rate < threshold:
                    continue

                # Calculate composite score
                success_weight = 0.4
                strength_weight = 0.3
                confidence_weight = 0.2
                efficiency_weight = 0.1

                success_score = perf.success_rate
                strength_score = perf.avg_strength
                confidence_score = perf.avg_confidence

                # Efficiency score (lower execution time is better)
                efficiency_score = max(
                    0, 1 - (perf.execution_time / self.max_execution_time)
                )

                score = (
                    success_weight * success_score
                    + strength_weight * strength_score
                    + confidence_weight * confidence_score
                    + efficiency_weight * efficiency_score
                )

            pattern_scores.append((pattern, score))

        # Sort by score descending
        return sorted(pattern_scores, key=lambda x: x[1], reverse=True)

    def _select_within_constraints(
        self,
        ranked_patterns: List[Tuple[str, float]],
        _category: PatternCategory,
        max_count: int,
    ) -> List[str]:
        """
        Select patterns within resource constraints.

        Args:
            ranked_patterns: List of (pattern, score) tuples
            category: Pattern category
            max_count: Maximum number of patterns to select

        Returns:
            List of selected pattern names
        """
        selected = []
        total_time = 0
        total_memory = 0

        for pattern, _score in ranked_patterns:
            if len(selected) >= max_count:
                break

            if pattern not in self.pattern_performance:
                # New pattern - assume reasonable resource usage
                exec_time = 0.1
                memory = 10
            else:
                perf = self.pattern_performance[pattern]
                exec_time = perf.execution_time
                memory = perf.memory_usage

            # Check resource constraints
            if (
                total_time + exec_time > self.max_execution_time
                or total_memory + memory > self.max_memory_usage
            ):
                continue

            selected.append(pattern)
            total_time += exec_time
            total_memory += memory

        return selected

    def _ensure_minimum_coverage(
        self, selected: Set[str], available: List[str]
    ) -> Set[str]:
        """Ensure minimum pattern coverage across categories."""
        # Ensure at least one pattern from each major category
        major_categories = {PatternCategory.TREND, PatternCategory.OSCILLATOR}

        for category in major_categories:
            category_patterns = [
                p for p in available if self.pattern_categories.get(p) == category
            ]

            if not any(p in selected for p in category_patterns):
                # Add the best available pattern from this category
                ranked = self._rank_patterns_by_performance(category_patterns, 0.0)
                if ranked:
                    selected.add(ranked[0][0])

        return selected

    def _apply_performance_decay(self) -> None:
        """Apply time-based decay to pattern performance for unused patterns."""
        current_time = time.time()
        decay_threshold = 3600  # 1 hour

        for _, perf in self.pattern_performance.items():
            time_since_used = current_time - perf.last_used
            if time_since_used > decay_threshold:
                # Apply decay
                decay_factor = self.performance_decay_factor ** (
                    time_since_used / decay_threshold
                )
                perf.success_rate *= decay_factor
                perf.avg_strength *= decay_factor
                perf.avg_confidence *= decay_factor

    def _adapt_thresholds(self) -> None:
        """Adapt success thresholds based on recent performance."""
        if len(self.market_history) < 10:
            return

        # Analyze recent performance by regime
        regime_performance = defaultdict(list)

        for condition in self.market_history:
            regime_performance[condition.regime].append(condition)

        # Adjust thresholds based on market conditions
        for regime, conditions in regime_performance.items():
            if len(conditions) < 5:
                continue

            # Adjust thresholds based on market conditions
            regime_name = regime.name if hasattr(regime, "name") else str(regime)
            if "VOLATILITY" in regime_name and "HIGH" in regime_name:  # High volatility
                self.success_thresholds[regime][PatternCategory.OSCILLATOR] *= 0.9
                self.success_thresholds[regime][PatternCategory.TREND] *= 1.1
            elif "TREND" in regime_name and "STRONG" in regime_name:  # Strong trend
                self.success_thresholds[regime][PatternCategory.TREND] *= 0.9
                self.success_thresholds[regime][PatternCategory.OSCILLATOR] *= 1.1

    def get_pattern_statistics(self) -> dict[str, object]:
        """Get comprehensive pattern performance statistics."""
        total_patterns = len(self.pattern_performance)
        active_patterns = 0
        category_distribution: Dict[str, int] = defaultdict(int)
        performance_summary: Dict[str, dict[str, float | int]] = {}
        weighted_execution_time = 0.0
        weighted_memory_usage = 0.0
        total_usage_count = 0

        for perf in self.pattern_performance.values():
            category_distribution[perf.category.value] += 1

            if perf.usage_count > 0:
                active_patterns += 1
                performance_summary[perf.pattern_name] = {
                    "success_rate": perf.success_rate,
                    "avg_strength": perf.avg_strength,
                    "usage_count": perf.usage_count,
                    "execution_time": perf.execution_time,
                }
                weighted_execution_time += perf.execution_time * perf.usage_count
                weighted_memory_usage += perf.memory_usage * perf.usage_count
                total_usage_count += perf.usage_count

        avg_execution_time = (
            weighted_execution_time / total_usage_count if total_usage_count > 0 else 0.0
        )
        avg_memory_usage = (
            weighted_memory_usage / total_usage_count if total_usage_count > 0 else 0.0
        )

        return {
            "total_patterns": total_patterns,
            "active_patterns": active_patterns,
            "category_distribution": dict(category_distribution),
            "performance_summary": performance_summary,
            "resource_usage": {
                "avg_execution_time": avg_execution_time,
                "avg_memory_usage": avg_memory_usage,
                "total_usage_count": total_usage_count,
            },
        }
