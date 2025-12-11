"""
Dynamic Adapter for Action Signal Guide.

This component integrates adaptive pattern selection and signal quality filtering
to create a comprehensive dynamic adaptation system for trading signals.
"""

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from ztb.analysis.market_regime_types import MarketRegime
from ztb.utils.logging_utils import get_logger

from .adaptive_pattern_selector import AdaptivePatternSelector
from .signal_quality_filter import SignalQualityFilter


@dataclass
class AdaptationMetrics:
    """Metrics for adaptation system performance."""

    patterns_selected: int
    signals_filtered: int
    avg_quality_score: float
    adaptation_time: float
    market_regime: str
    timestamp: float


class DynamicAdapter:
    """
    Dynamic adaptation system that integrates pattern selection and signal filtering.

    This class provides:
    - Real-time pattern activation/deactivation
    - Quality-based signal filtering
    - Performance-driven adaptation
    - Market regime-aware optimization
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dynamic adapter.

        Args:
            config: Configuration dictionary
        """
        self.logger = get_logger("ztb.trading.strategies.dynamic_adapter")
        self.config = config

        # Core components
        self.pattern_selector = AdaptivePatternSelector(
            config.get("pattern_selector", {})
        )
        self.quality_filter = SignalQualityFilter(config.get("quality_filter", {}))

        # Adaptation state
        self.active_patterns: Set[str] = set()
        self.market_regime: Optional[MarketRegime] = None
        self.last_adaptation = time.time()

        # Performance tracking
        self.adaptation_history = []
        self.performance_metrics = {
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "avg_adaptation_time": 0,
            "pattern_activation_rate": 0,
            "signal_quality_improvement": 0,
        }

        # Adaptation parameters
        self.adaptation_interval = config.get("adaptation_interval", 300)  # 5 minutes
        self.min_adaptation_signals = config.get("min_adaptation_signals", 10)
        self.quality_improvement_threshold = config.get(
            "quality_improvement_threshold", 0.05
        )

        # Market regime tracking
        self.regime_stability_counter = 0
        self.last_regime_change = time.time()

        self.logger.info("DynamicAdapter initialized")

    def adapt_and_filter(
        self,
        available_patterns: List[str],
        signals: List[Any],
        market_data: pd.DataFrame,
        market_regime: MarketRegime,
        force_adaptation: bool = False,
    ) -> List[Any]:
        """
        Perform dynamic adaptation and signal filtering.

        Args:
            available_patterns: List of available pattern names
            signals: Raw signals to filter
            market_data: Current market data
            market_regime: Current market regime
            force_adaptation: Force pattern adaptation regardless of timing

        Returns:
            Filtered and optimized signals
        """
        start_time = time.time()

        # Update market regime
        regime_changed = self._update_market_regime(market_regime)

        # Check if adaptation is needed
        should_adapt = (
            force_adaptation
            or time.time() - self.last_adaptation > self.adaptation_interval
            or regime_changed
        )

        if should_adapt:
            # Perform pattern selection
            old_active_count = len(self.active_patterns)
            self.active_patterns = self.pattern_selector.select_active_patterns(
                available_patterns
            )
            new_active_count = len(self.active_patterns)

            self.last_adaptation = time.time()

            if new_active_count != old_active_count:
                self.logger.info(
                    f"Pattern adaptation: {old_active_count} -> {new_active_count} active patterns"
                )

        # DEBUG: Print active patterns and signal pattern names
        # self.logger.info(f"DEBUG: Active patterns: {self.active_patterns}")
        # for signal in signals:
        #     self.logger.info(f"DEBUG: Signal pattern name: {self._get_signal_pattern_name(signal)}")

        # Filter signals based on active patterns and quality
        # If active_patterns is empty, allow permissive mode (useful for backtests)
        pattern_filtered_signals = []
        permissive_on_empty = self.config.get("permissive_on_empty_patterns", True)
        if not self.active_patterns and permissive_on_empty:
            # Treat as if all patterns are active when no patterns were selected
            pattern_filtered_signals = list(signals)
        else:
            for signal in signals:
                pattern_name = self._get_signal_pattern_name(signal)
                # Special handling for aggregated signals - check if any source pattern is active
                if pattern_name == "aggregated":
                    if hasattr(signal, "source_patterns"):
                        is_active = False
                        for source in signal.source_patterns:
                            # Normalize source pattern name
                            normalized_source = self._normalize_pattern_name(source)
                            if normalized_source in self.active_patterns:
                                is_active = True
                                break
                        if is_active:
                            pattern_filtered_signals.append(signal)
                        else:
                            # Dropped aggregated signals if none of the source patterns are active
                            self.logger.debug(
                                f"Dropped aggregated signal. Sources: {signal.source_patterns}, Active: {self.active_patterns}"
                            )
                    else:
                        pattern_filtered_signals.append(signal)
                elif pattern_name in self.active_patterns:
                    pattern_filtered_signals.append(signal)
                else:
                    self.logger.debug(
                        f"Dropped signal {pattern_name}. Active: {self.active_patterns}"
                    )

        # Apply quality filtering
        quality_filtered_signals = self.quality_filter.filter_signals(
            pattern_filtered_signals, market_data, market_regime
        )

        # Update performance tracking
        adaptation_time = time.time() - start_time
        self._update_adaptation_metrics(
            len(self.active_patterns),
            len(quality_filtered_signals),
            self._calculate_avg_quality(quality_filtered_signals),
            adaptation_time,
            market_regime,
        )

        # Periodic threshold adaptation
        if len(self.adaptation_history) % 10 == 0:  # Every 10 adaptations
            self.quality_filter.adapt_thresholds()

        self.logger.debug(
            f"Dynamic adaptation completed: {len(quality_filtered_signals)} signals from {len(signals)} raw signals"
        )
        return quality_filtered_signals

    def update_pattern_performance(
        self,
        pattern_name: str,
        success: bool,
        strength: float,
        confidence: float,
        execution_time: float,
        memory_usage: float,
    ):
        """
        Update pattern performance metrics.

        Args:
            pattern_name: Name of the pattern
            success: Whether the signal was successful
            strength: Signal strength (0-1)
            confidence: Signal confidence (0-1)
            execution_time: Execution time in seconds
            memory_usage: Memory usage in MB
        """
        self.pattern_selector.update_pattern_performance(
            pattern_name, success, strength, confidence, execution_time, memory_usage
        )

    def update_market_condition(
        self,
        regime: MarketRegime,
        volatility: float,
        trend_strength: float,
        volume_trend: float,
    ):
        """
        Update market condition for adaptation.

        Args:
            regime: Current market regime
            volatility: Current volatility level
            trend_strength: Current trend strength
            volume_trend: Current volume trend
        """
        self.pattern_selector.update_market_condition(
            regime, volatility, trend_strength, volume_trend
        )

    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics."""
        stats = {
            "performance_metrics": self.performance_metrics.copy(),
            "active_patterns": sorted(list(self.active_patterns)),
            "current_regime": str(self.market_regime)
            if self.market_regime
            else "unknown",
            "regime_stability": self.regime_stability_counter,
            "last_adaptation": self.last_adaptation,
            "adaptation_history_size": len(self.adaptation_history),
        }

        # Add component statistics
        stats["pattern_selector"] = self.pattern_selector.get_pattern_statistics()
        stats["quality_filter"] = self.quality_filter.get_quality_statistics()

        # Recent adaptation history
        if self.adaptation_history:
            recent_adaptations = self.adaptation_history[
                -min(10, len(self.adaptation_history)) :
            ]
            stats["recent_adaptations"] = [
                {
                    "patterns_selected": a.patterns_selected,
                    "signals_filtered": a.signals_filtered,
                    "avg_quality_score": a.avg_quality_score,
                    "market_regime": a.market_regime,
                    "timestamp": a.timestamp,
                }
                for a in recent_adaptations
            ]

        return stats

    def force_adaptation(
        self, available_patterns: List[str], market_regime: MarketRegime
    ):
        """Force immediate pattern adaptation."""
        old_patterns = self.active_patterns.copy()
        self.active_patterns = self.pattern_selector.select_active_patterns(
            available_patterns
        )
        self._update_market_regime(market_regime)
        self.last_adaptation = time.time()

        changed_patterns = old_patterns.symmetric_difference(self.active_patterns)
        if changed_patterns:
            self.logger.info(
                f"Forced adaptation changed patterns: {sorted(changed_patterns)}"
            )

    def _update_market_regime(self, new_regime: MarketRegime) -> bool:
        """
        Update market regime and return whether it changed.

        Args:
            new_regime: New market regime

        Returns:
            True if regime changed, False otherwise
        """
        if self.market_regime != new_regime:
            old_regime = self.market_regime
            self.market_regime = new_regime
            self.regime_stability_counter = 0
            self.last_regime_change = time.time()

            self.logger.info(f"Market regime changed: {old_regime} -> {new_regime}")
            return True
        else:
            self.regime_stability_counter += 1
            return False

    def _normalize_pattern_name(self, name: str) -> str:
        """Normalize pattern name to match active_patterns format."""
        # Explicit mappings for known inconsistencies
        mapping = {
            "MACDPatternRecognizer": "macd",
            "RSIPatternRecognizer": "rsi",
            "BollingerBandsRecognizer": "bollinger_bands",
            "HeikinAshiRecognizer": "heikin_ashi",
            "ADXRecognizer": "adx",
            "StochasticRecognizer": "stochastic",
            "CCIRecognizer": "cci",
            "WilliamsRRecognizer": "williams_r",
            "MFIRecognizer": "mfi",
            "ChaikinADRecognizer": "chaikin_ad",
            "GranvilleLawRecognizer": "granville_law",
            "HammerRecognizer": "hammer",
            "MorningStarRecognizer": "morning_star",
            "EveningStarRecognizer": "evening_star",
            "ThreeWhiteSoldiersRecognizer": "three_white_soldiers",
            "ThreeBlackCrowsRecognizer": "three_black_crows",
            "FibonacciRetracementRecognizer": "fibonacci_retracement",
            "FibonacciExtensionRecognizer": "fibonacci_extension",
            "GartleyRecognizer": "gartley",
            "ButterflyRecognizer": "butterfly",
            "ImpulseWaveRecognizer": "impulse_wave",
            "CorrectiveWaveRecognizer": "corrective_wave",
        }

        if name in mapping:
            return mapping[name]

        # Remove 'Recognizer' suffix if present
        if name.endswith("Recognizer"):
            name = name[:-10]

        # Convert CamelCase to snake_case
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def _get_signal_pattern_name(self, signal: Any) -> str:
        """Extract pattern name from signal object."""
        # Try different attribute names
        for attr in ["pattern_name", "pattern", "type", "signal_type"]:
            if hasattr(signal, attr):
                value = getattr(signal, attr)
                if isinstance(value, str):
                    return value

        # Fallback to class name
        return signal.__class__.__name__.lower()

    def _calculate_avg_quality(self, signals: List[Any]) -> float:
        """Calculate average quality score of filtered signals."""
        if not signals:
            return 0.0

        total_quality = 0.0
        count = 0

        for signal in signals:
            # Try to get quality score from signal
            if hasattr(signal, "quality_score"):
                total_quality += signal.quality_score
                count += 1
            elif hasattr(signal, "strength") and hasattr(signal, "confidence"):
                # Fallback quality calculation
                quality = (signal.strength + signal.confidence) / 2.0
                total_quality += quality
                count += 1

        return total_quality / count if count > 0 else 0.0

    def _update_adaptation_metrics(
        self,
        patterns_selected: int,
        signals_filtered: int,
        avg_quality_score: float,
        adaptation_time: float,
        market_regime: MarketRegime,
    ):
        """Update adaptation performance metrics."""
        # Create adaptation record
        metrics = AdaptationMetrics(
            patterns_selected=patterns_selected,
            signals_filtered=signals_filtered,
            avg_quality_score=avg_quality_score,
            adaptation_time=adaptation_time,
            market_regime=str(market_regime),
            timestamp=time.time(),
        )

        self.adaptation_history.append(metrics)

        # Update running averages
        self.performance_metrics["total_adaptations"] += 1

        # Simple success criteria: produced some signals
        if signals_filtered > 0:
            self.performance_metrics["successful_adaptations"] += 1

        # Update average adaptation time
        current_avg_time = self.performance_metrics["avg_adaptation_time"]
        total_adaptations = self.performance_metrics["total_adaptations"]
        self.performance_metrics["avg_adaptation_time"] = (
            (current_avg_time * (total_adaptations - 1)) + adaptation_time
        ) / total_adaptations

        # Calculate pattern activation rate
        if hasattr(self.pattern_selector, "pattern_performance"):
            total_patterns = len(self.pattern_selector.pattern_performance)
            if total_patterns > 0:
                self.performance_metrics["pattern_activation_rate"] = (
                    patterns_selected / total_patterns
                )

        # Keep history bounded
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-500:]

    def get_optimal_config_suggestions(self) -> Dict[str, Any]:
        """Get suggestions for optimal configuration based on adaptation history."""
        if len(self.adaptation_history) < 20:
            return {}

        suggestions = {}

        # Analyze adaptation time
        adaptation_times = [a.adaptation_time for a in self.adaptation_history]
        avg_time = sum(adaptation_times) / len(adaptation_times)

        if avg_time > 1.0:
            suggestions["adaptation_interval"] = max(60, self.adaptation_interval * 0.8)
        elif avg_time < 0.1:
            suggestions["adaptation_interval"] = min(
                1800, self.adaptation_interval * 1.2
            )

        # Analyze signal quality
        quality_scores = [
            a.avg_quality_score
            for a in self.adaptation_history
            if a.signals_filtered > 0
        ]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            if avg_quality < 0.5:
                suggestions["quality_improvement_needed"] = True
                suggestions["suggested_min_confidence"] = max(0.3, avg_quality - 0.1)

        # Analyze pattern selection
        pattern_counts = [a.patterns_selected for a in self.adaptation_history]
        if pattern_counts:
            avg_patterns = sum(pattern_counts) / len(pattern_counts)
            if avg_patterns < 3:
                suggestions["increase_pattern_coverage"] = True
            elif avg_patterns > 8:
                suggestions["reduce_pattern_coverage"] = True

        return suggestions
