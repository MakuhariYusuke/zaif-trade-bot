"""
PerformanceTracker Component.

This component is responsible for tracking performance metrics.
Follows Single Responsibility Principle by focusing only on performance tracking.
"""

import statistics
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd
from ztb.utils.logging_utils import get_logger

from .interfaces import IPerformanceTracker

if TYPE_CHECKING:
    from ..types import PerformanceStats


class PerformanceTracker(IPerformanceTracker):
    """
    Tracks performance metrics for ActionSignalGuide operations.

    This class monitors:
    - Signal generation times
    - Pattern recognition performance
    - Cache hit rates
    - Memory usage
    - Error rates
    """

    def __init__(self, enable_detailed_tracking: bool = True):
        """
        Initialize PerformanceTracker.

        Args:
            enable_detailed_tracking: Whether to track detailed metrics
        """
        self.enable_detailed_tracking = enable_detailed_tracking
        self.logger = get_logger("ztb.trading.strategies.performance_tracker")

        # Timing metrics
        self.signal_generation_times: List[float] = []
        self.pattern_recognition_times: Dict[str, List[float]] = defaultdict(list)
        self.cache_operation_times: List[float] = []

        # Performance counters
        self.total_signals_generated = 0
        self.total_patterns_recognized = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0

        # Pattern-specific metrics
        self.pattern_success_rates: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"success": 0, "total": 0}
        )
        self.pattern_strengths: Dict[str, List[float]] = defaultdict(list)
        self.pattern_confidences: Dict[str, List[float]] = defaultdict(list)

        # Memory and resource metrics
        self.memory_usage_samples: List[float] = []
        self.start_time = time.time()

        # SAC correlation tracking
        self.sac_action_history: List[Dict[str, Union[str, int, float]]] = []
        self.signal_sac_correlation_data: List[Dict[str, Union[str, int, float]]] = []
        self.regime_performance_correlation: Dict[str, List[float]] = defaultdict(list)

    def record_signal_generation(self, duration: float) -> None:
        """
        Record signal generation performance.

        Args:
            duration: Time taken for signal generation
        """
        self.signal_generation_times.append(duration)
        self.total_signals_generated += 1

        # Keep only recent samples for memory efficiency
        if len(self.signal_generation_times) > 1000:
            self.signal_generation_times = self.signal_generation_times[-500:]

    def record_pattern_recognition(
        self, pattern_type: str, duration: float, success: bool
    ) -> None:
        """
        Record pattern recognition performance.

        Args:
            pattern_type: Type of pattern recognized
            duration: Time taken for recognition
            success: Whether recognition was successful
        """
        if self.enable_detailed_tracking:
            self.pattern_recognition_times[pattern_type].append(duration)

            # Keep only recent samples
            if len(self.pattern_recognition_times[pattern_type]) > 100:
                self.pattern_recognition_times[
                    pattern_type
                ] = self.pattern_recognition_times[pattern_type][-50:]

        self.total_patterns_recognized += 1

        # Update success rates
        self.pattern_success_rates[pattern_type]["total"] += 1
        if success:
            self.pattern_success_rates[pattern_type]["success"] += 1

    def record_pattern_signal(
        self, pattern_type: str, strength: float, confidence: float
    ) -> None:
        """
        Record pattern signal metrics.

        Args:
            pattern_type: Type of pattern
            strength: Signal strength
            confidence: Signal confidence
        """
        if self.enable_detailed_tracking:
            self.pattern_strengths[pattern_type].append(strength)
            self.pattern_confidences[pattern_type].append(confidence)

            # Keep only recent samples
            if len(self.pattern_strengths[pattern_type]) > 100:
                self.pattern_strengths[pattern_type] = self.pattern_strengths[
                    pattern_type
                ][-50:]
            if len(self.pattern_confidences[pattern_type]) > 100:
                self.pattern_confidences[pattern_type] = self.pattern_confidences[
                    pattern_type
                ][-50:]

    def record_cache_operation(self, duration: float, hit: bool) -> None:
        """
        Record cache operation performance.

        Args:
            duration: Time taken for cache operation
            hit: Whether it was a cache hit
        """
        self.cache_operation_times.append(duration)

        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        # Keep only recent samples
        if len(self.cache_operation_times) > 500:
            self.cache_operation_times = self.cache_operation_times[-250:]

    def record_error(self, error_type: str, error_message: str) -> None:
        """
        Record error occurrence.

        Args:
            error_type: Type of error
            error_message: Error message
        """
        self.errors += 1
        self.logger.warning(
            f"Performance error recorded: {error_type} - {error_message}"
        )

    def record_memory_usage(self, memory_mb: float) -> None:
        """
        Record memory usage sample.

        Args:
            memory_mb: Memory usage in MB
        """
        self.memory_usage_samples.append(memory_mb)

        # Keep only recent samples
        if len(self.memory_usage_samples) > 100:
            self.memory_usage_samples = self.memory_usage_samples[-50:]

    def track_sac_correlation(
        self,
        signal: "ActionSignal",
        sac_action: Optional[Union[int, float]] = None,
        market_regime: Optional[str] = None,
        portfolio_state: Optional[Dict[str, Union[int, float]]] = None
    ) -> None:
        """
        Track correlation between Action Signal Guide signals and SAC actions.

        Args:
            signal: Generated Action Signal Guide signal
            sac_action: Corresponding SAC action (-1 to 1)
            market_regime: Current market regime
            portfolio_state: Current portfolio state
        """
        if not self.enable_detailed_tracking:
            return

        correlation_entry = {
            "timestamp": time.time(),
            "signal_action": signal.action.value if hasattr(signal.action, 'value') else str(signal.action),
            "signal_confidence": signal.confidence,
            "signal_reason": signal.reason,
            "sac_action": sac_action,
            "market_regime": market_regime or "unknown",
            "portfolio_value": portfolio_state.get("value", 0.0) if portfolio_state else 0.0,
            "position_size": portfolio_state.get("position", 0.0) if portfolio_state else 0.0,
        }

        self.signal_sac_correlation_data.append(correlation_entry)

        # Track regime-specific performance
        if market_regime:
            regime_key = f"{market_regime}_{signal.action}"
            if sac_action is not None:
                # Calculate correlation strength (simplified)
                correlation_strength = abs(signal.confidence - abs(sac_action)) if signal.confidence else 0.0
                self.regime_performance_correlation[regime_key].append(correlation_strength)

        # Maintain history size limit
        if len(self.signal_sac_correlation_data) > self.max_history_size:
            self.signal_sac_correlation_data.pop(0)

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.

        Returns:
            Dictionary with performance metrics
        """
        summary = {
            "total_signals_generated": self.total_signals_generated,
            "total_patterns_recognized": self.total_patterns_recognized,
            "total_errors": self.errors,
            "uptime_seconds": time.time() - self.start_time,
            "cache_hit_rate": self._calculate_cache_hit_rate(),
        }

        # Signal generation metrics
        if self.signal_generation_times:
            summary["signal_generation"] = {
                "avg_time": statistics.mean(self.signal_generation_times),
                "min_time": min(self.signal_generation_times),
                "max_time": max(self.signal_generation_times),
                "median_time": statistics.median(self.signal_generation_times),
                "samples": len(self.signal_generation_times),
            }

        # Pattern recognition metrics
        if self.enable_detailed_tracking and self.pattern_recognition_times:
            pattern_metrics = {}
            for pattern_type, times in self.pattern_recognition_times.items():
                if times:
                    pattern_metrics[pattern_type] = {
                        "avg_time": statistics.mean(times),
                        "success_rate": self._calculate_pattern_success_rate(
                            pattern_type
                        ),
                        "samples": len(times),
                    }
            summary["pattern_recognition"] = pattern_metrics

        # Cache metrics
        if self.cache_operation_times:
            summary["cache_operations"] = {
                "avg_time": statistics.mean(self.cache_operation_times),
                "total_operations": len(self.cache_operation_times),
                "hit_rate": self._calculate_cache_hit_rate(),
            }

        # Memory metrics
        if self.memory_usage_samples:
            summary["memory"] = {
                "avg_usage_mb": statistics.mean(self.memory_usage_samples),
                "max_usage_mb": max(self.memory_usage_samples),
                "current_usage_mb": self.memory_usage_samples[-1]
                if self.memory_usage_samples
                else 0,
            }

        return summary

    def get_pattern_performance(
        self, pattern_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed pattern performance metrics.

        Args:
            pattern_type: Specific pattern type, or None for all

        Returns:
            Dictionary with pattern performance metrics
        """
        if pattern_type:
            return self._get_single_pattern_performance(pattern_type)
        else:
            return {
                pt: self._get_single_pattern_performance(pt)
                for pt in self.pattern_success_rates.keys()
            }

    def reset_metrics(self) -> None:
        """
        Reset all performance metrics.
        """
        self.signal_generation_times.clear()
        self.pattern_recognition_times.clear()
        self.cache_operation_times.clear()

        self.total_signals_generated = 0
        self.total_patterns_recognized = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0

        self.pattern_success_rates.clear()
        self.pattern_strengths.clear()
        self.pattern_confidences.clear()
        self.memory_usage_samples.clear()

        self.start_time = time.time()
        self.logger.info("Performance metrics reset")

    def _calculate_cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Cache hit rate as percentage
        """
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0

    def _calculate_pattern_success_rate(self, pattern_type: str) -> float:
        """
        Calculate success rate for a pattern type.

        Args:
            pattern_type: Pattern type

        Returns:
            Success rate as percentage
        """
        stats = self.pattern_success_rates.get(pattern_type, {"success": 0, "total": 0})
        return (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0.0

    def _get_single_pattern_performance(self, pattern_type: str) -> Dict[str, Any]:
        """
        Get performance metrics for a single pattern type.

        Args:
            pattern_type: Pattern type

        Returns:
            Dictionary with pattern metrics
        """
        metrics = {
            "success_rate": self._calculate_pattern_success_rate(pattern_type),
            "total_attempts": self.pattern_success_rates[pattern_type]["total"],
            "successful_attempts": self.pattern_success_rates[pattern_type]["success"],
        }

        # Add timing metrics if available
        if pattern_type in self.pattern_recognition_times:
            times = self.pattern_recognition_times[pattern_type]
            if times:
                metrics["recognition_time"] = {
                    "avg": statistics.mean(times),
                    "min": min(times),
                    "max": max(times),
                }

        # Add signal quality metrics if available
        if (
            pattern_type in self.pattern_strengths
            and self.pattern_strengths[pattern_type]
        ):
            strengths = self.pattern_strengths[pattern_type]
            metrics["signal_strength"] = {
                "avg": statistics.mean(strengths),
                "min": min(strengths),
                "max": max(strengths),
            }

        if (
            pattern_type in self.pattern_confidences
            and self.pattern_confidences[pattern_type]
        ):
            confidences = self.pattern_confidences[pattern_type]
            metrics["signal_confidence"] = {
                "avg": statistics.mean(confidences),
                "min": min(confidences),
                "max": max(confidences),
            }

        return metrics

    def record_cache_hit(self) -> None:
        """Record cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record cache miss."""
        self.cache_misses += 1

    def get_performance_stats(self) -> "PerformanceStats":
        """
        Get performance statistics.

        Returns:
            Dictionary of performance statistics
        """
        total_cache_operations = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            (self.cache_hits / total_cache_operations)
            if total_cache_operations > 0
            else 0.0
        )

        stats = {
            "total_signals_generated": self.total_signals_generated,
            "total_patterns_recognized": self.total_patterns_recognized,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "errors": self.errors,
            "uptime_seconds": time.time() - self.start_time,
        }

        # Add timing statistics if available
        if self.signal_generation_times:
            stats["signal_generation_time"] = {
                "avg": statistics.mean(self.signal_generation_times),
                "min": min(self.signal_generation_times),
                "max": max(self.signal_generation_times),
                "count": len(self.signal_generation_times),
            }

        if self.cache_operation_times:
            stats["cache_operation_time"] = {
                "avg": statistics.mean(self.cache_operation_times),
                "min": min(self.cache_operation_times),
                "max": max(self.cache_operation_times),
                "count": len(self.cache_operation_times),
            }

        return stats

    def get_sac_correlation_analysis(self) -> Dict[str, Union[int, float, dict, list]]:
        """
        Get comprehensive SAC correlation analysis.

        Returns:
            Dictionary containing correlation analysis results
        """
        if not self.signal_sac_correlation_data:
            return {"error": "No SAC correlation data available"}

        df = pd.DataFrame(self.signal_sac_correlation_data)

        analysis = {
            "total_correlation_points": len(df),
            "correlation_period_days": (df["timestamp"].max() - df["timestamp"].min()) / 86400,
            "signal_distribution": self._analyze_signal_distribution(df),
            "sac_action_distribution": self._analyze_sac_action_distribution(df),
            "regime_correlation": self._analyze_regime_correlation(df),
            "performance_correlation": self._analyze_performance_correlation(df),
            "temporal_patterns": self._analyze_temporal_correlation_patterns(df),
        }

        return analysis

    def _analyze_signal_distribution(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """Analyze signal distribution in correlation data."""
        signal_counts = df["signal_action"].value_counts().to_dict()
        total_signals = len(df)

        return {
            "signal_counts": signal_counts,
            "signal_percentages": {k: v/total_signals for k, v in signal_counts.items()},
            "most_common_signal": max(signal_counts.items(), key=lambda x: x[1])[0] if signal_counts else None,
        }

    def _analyze_sac_action_distribution(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """Analyze SAC action distribution."""
        sac_actions = df["sac_action"].dropna()

        if len(sac_actions) == 0:
            return {"error": "No SAC action data available"}

        return {
            "mean_sac_action": float(sac_actions.mean()),
            "std_sac_action": float(sac_actions.std()),
            "sac_action_range": [float(sac_actions.min()), float(sac_actions.max())],
            "sac_action_skewness": float(sac_actions.skew()),
            "strong_buy_signals": int((sac_actions > 0.7).sum()),
            "strong_sell_signals": int((sac_actions < -0.7).sum()),
        }

    def _analyze_regime_correlation(self, df: pd.DataFrame) -> Dict[str, Union[int, float, dict]]:
        """Analyze correlation by market regime."""
        regime_groups = df.groupby("market_regime")

        regime_analysis = {}
        for regime, group in regime_groups:
            valid_data = group.dropna(subset=["sac_action", "signal_confidence"])

            if len(valid_data) > 1:
                correlation = valid_data["signal_confidence"].corr(valid_data["sac_action"])
                regime_analysis[regime] = {
                    "correlation_coefficient": float(correlation) if not pd.isna(correlation) else 0.0,
                    "sample_size": len(valid_data),
                    "avg_signal_confidence": float(valid_data["signal_confidence"].mean()),
                    "avg_sac_action": float(valid_data["sac_action"].mean()),
                }
            else:
                regime_analysis[regime] = {"error": "Insufficient data for correlation analysis"}

        return regime_analysis

    def _analyze_performance_correlation(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """Analyze correlation between signals and portfolio performance."""
        # Calculate forward returns (simplified approach)
        df_sorted = df.sort_values("timestamp").copy()

        # Calculate correlation between signal confidence and subsequent portfolio changes
        valid_data = df_sorted.dropna(subset=["signal_confidence", "portfolio_value"])

        if len(valid_data) < 2:
            return {"error": "Insufficient data for performance correlation"}

        # Simple correlation analysis (in practice, you'd want more sophisticated forward-looking analysis)
        confidence_correlation = valid_data["signal_confidence"].corr(valid_data["portfolio_value"])

        return {
            "signal_confidence_portfolio_correlation": float(confidence_correlation) if not pd.isna(confidence_correlation) else 0.0,
            "sample_size": len(valid_data),
            "correlation_strength": self._interpret_correlation_strength(confidence_correlation),
        }

    def _analyze_temporal_correlation_patterns(self, df: pd.DataFrame) -> Dict[str, Union[int, float, list]]:
        """Analyze temporal patterns in correlation data."""
        df_sorted = df.sort_values("timestamp").copy()
        df_sorted["hour"] = pd.to_datetime(df_sorted["timestamp"], unit="s").dt.hour
        df_sorted["day_of_week"] = pd.to_datetime(df_sorted["timestamp"], unit="s").dt.dayofweek

        hourly_patterns = df_sorted.groupby("hour").agg({
            "signal_confidence": "mean",
            "sac_action": "mean"
        }).to_dict()

        return {
            "hourly_signal_confidence": hourly_patterns["signal_confidence"],
            "hourly_sac_action": hourly_patterns["sac_action"],
            "best_signal_hour": max(hourly_patterns["signal_confidence"].items(), key=lambda x: x[1])[0],
            "best_sac_hour": max(hourly_patterns["sac_action"].items(), key=lambda x: x[1])[0],
        }

    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation coefficient strength."""
        if pd.isna(correlation):
            return "no_data"
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.3:
            return "moderate"
        elif abs_corr >= 0.1:
            return "weak"
        else:
            return "very_weak"
