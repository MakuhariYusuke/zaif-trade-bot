"""
PerformanceTracker Component.

This component is responsible for tracking performance metrics.
Follows Single Responsibility Principle by focusing only on performance tracking.
"""

import statistics
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
