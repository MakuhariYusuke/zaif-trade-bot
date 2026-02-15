"""
PatternStatistics Component.

This component is responsible for tracking pattern recognition statistics.
Follows Single Responsibility Principle by focusing only on pattern statistics.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, TypedDict, cast

from ztb.utils.logging_utils import get_logger

from .history_helpers import append_with_compaction
from .interfaces import IPatternStatistics

if TYPE_CHECKING:
    from ..action_signal_guide import ActionSignal
    from ..types import PatternStats, SignalMetadata


class DetectionHistoryEntry(TypedDict):
    timestamp: float
    pattern_type: str
    detected: bool
    metadata: "SignalMetadata"


class PatternCombinationEntry(TypedDict):
    patterns: tuple[str, ...]
    frequency: int


class PatternCombinationStats(TypedDict):
    combinations: list[PatternCombinationEntry]
    total_combinations: int
    min_frequency: int


class TemporalPatternStats(TypedDict):
    pattern_type: str
    window_hours: int
    total_detections: int
    success_rate: float
    detection_frequency_per_hour: float
    time_span_hours: float


class TemporalPatternError(TypedDict):
    error: str


class OverallStatistics(TypedDict, total=False):
    total_detections: int
    total_successful_detections: int
    overall_success_rate: float
    unique_patterns: int
    uptime_hours: float
    detection_rate_per_hour: float
    most_frequent_pattern: tuple[str, int] | None
    least_frequent_pattern: tuple[str, int] | None
    top_patterns_by_success: list[tuple[str, float]]
    bottom_patterns_by_success: list[tuple[str, float]]


class PatternStatistics(IPatternStatistics):
    """
    Tracks statistics for pattern recognition operations.

    This class monitors:
    - Pattern detection frequencies
    - Pattern success rates
    - Signal quality metrics
    - Pattern combinations and correlations
    - Historical performance trends
    """

    def __init__(self, max_history_size: int = 10000) -> None:
        """
        Initialize PatternStatistics.

        Args:
            max_history_size: Maximum number of historical records to keep
        """
        self.max_history_size = max_history_size
        self.logger = get_logger("ztb.trading.strategies.pattern_statistics")

        # Pattern detection statistics
        self.pattern_counts: dict[str, int] = defaultdict(int)
        self.pattern_success_counts: dict[str, int] = defaultdict(int)
        self.pattern_failure_counts: dict[str, int] = defaultdict(int)

        # Signal quality tracking
        self.pattern_strengths: dict[str, list[float]] = defaultdict(list)
        self.pattern_confidences: dict[str, list[float]] = defaultdict(list)
        self.pattern_accuracies: dict[str, list[bool]] = defaultdict(list)

        # Pattern combinations
        self.pattern_combinations: dict[tuple[str, ...], int] = defaultdict(int)
        self.pattern_correlations: dict[tuple[str, str], float] = {}

        # Historical data
        self.detection_history: deque[DetectionHistoryEntry] = deque(
            maxlen=max_history_size
        )
        self.temporal_patterns: dict[str, list[tuple[int, bool]]] = defaultdict(list)

        # Performance metrics
        self.start_time = time.time()
        self.total_detections = 0
        self.total_successful_detections = 0

    def record_pattern_detection(
        self,
        pattern_type: str,
        detected: bool,
        metadata: "SignalMetadata | None" = None,
    ) -> None:
        """
        Record pattern detection attempt.

        Args:
            pattern_type: Type of pattern detected
            detected: Whether pattern was successfully detected
            metadata: Additional metadata about the detection
        """
        self.pattern_counts[pattern_type] += 1
        self.total_detections += 1

        if detected:
            self.pattern_success_counts[pattern_type] += 1
            self.total_successful_detections += 1
        else:
            self.pattern_failure_counts[pattern_type] += 1

        now = time.time()
        timestamp = int(now)

        # Record in history
        history_entry: DetectionHistoryEntry = {
            "timestamp": now,
            "pattern_type": pattern_type,
            "detected": detected,
            "metadata": cast("SignalMetadata", metadata or {}),
        }
        self.detection_history.append(history_entry)

        # Record temporal pattern
        append_with_compaction(
            self.temporal_patterns[pattern_type],
            (timestamp, detected),
            high_water=1000,
            retain=500,
        )

    def record_pattern_signal(self, pattern_type: str, signal: "ActionSignal") -> None:
        """
        Record pattern signal metrics.

        Args:
            pattern_type: Type of pattern
            signal: Signal object with strength and confidence
        """
        try:
            strength = float(getattr(signal, "strength", 0.0))
            confidence = float(getattr(signal, "confidence", 0.0))

            append_with_compaction(
                self.pattern_strengths[pattern_type],
                strength,
                high_water=500,
                retain=250,
            )
            append_with_compaction(
                self.pattern_confidences[pattern_type],
                confidence,
                high_water=500,
                retain=250,
            )

        except Exception as exc:
            self.logger.warning(f"Failed to record pattern signal metrics: {exc}")

    def update_pattern_strength_stats(self, pattern_type: str, strength: float) -> None:
        """
        Update pattern strength statistics.

        Args:
            pattern_type: Type of pattern
            strength: Signal strength value
        """
        append_with_compaction(
            self.pattern_strengths[pattern_type],
            float(strength),
            high_water=500,
            retain=250,
        )

    def record_pattern_accuracy(self, pattern_type: str, accurate: bool) -> None:
        """
        Record pattern prediction accuracy.

        Args:
            pattern_type: Type of pattern
            accurate: Whether the pattern prediction was accurate
        """
        append_with_compaction(
            self.pattern_accuracies[pattern_type],
            bool(accurate),
            high_water=1000,
            retain=500,
        )

    def record_pattern_combination(self, pattern_types: list[str]) -> None:
        """
        Record combination of patterns detected together.

        Args:
            pattern_types: List of pattern types detected simultaneously
        """
        if len(pattern_types) > 1:
            # Sort for consistent key
            combination_key = tuple(sorted(pattern_types))
            self.pattern_combinations[combination_key] += 1

    def get_pattern_statistics(
        self, pattern_type: str | None = None
    ) -> "PatternStats":
        """
        Get comprehensive pattern statistics.

        Args:
            pattern_type: Specific pattern type, or None for all

        Returns:
            Dictionary with pattern statistics
        """
        if pattern_type:
            return cast("PatternStats", self._get_single_pattern_stats(pattern_type))

        all_stats = {
            pattern_name: self._get_single_pattern_stats(pattern_name)
            for pattern_name in self.pattern_counts.keys()
        }
        return cast("PatternStats", all_stats)

    def get_detection_frequencies(self) -> dict[str, float]:
        """
        Get detection frequencies for all patterns.

        Returns:
            Dictionary mapping pattern types to detection frequencies
        """
        if self.total_detections == 0:
            return {}

        return {
            pattern_type: count / self.total_detections
            for pattern_type, count in self.pattern_counts.items()
        }

    def get_success_rates(self) -> dict[str, float]:
        """
        Get success rates for all patterns.

        Returns:
            Dictionary mapping pattern types to success rates
        """
        success_rates: dict[str, float] = {}
        for pattern_type, total_count in self.pattern_counts.items():
            success_count = self.pattern_success_counts[pattern_type]
            success_rates[pattern_type] = (
                (success_count / total_count * 100) if total_count > 0 else 0.0
            )

        return success_rates

    def get_accuracy_rates(self) -> dict[str, float]:
        """
        Get accuracy rates for all patterns.

        Returns:
            Dictionary mapping pattern types to accuracy rates
        """
        accuracy_rates: dict[str, float] = {}
        for pattern_type, accuracies in self.pattern_accuracies.items():
            if accuracies:
                accuracy_rates[pattern_type] = sum(accuracies) / len(accuracies) * 100
            else:
                accuracy_rates[pattern_type] = 0.0

        return accuracy_rates

    def get_pattern_combinations(
        self, min_frequency: int = 2
    ) -> PatternCombinationStats:
        """
        Get frequently occurring pattern combinations.

        Args:
            min_frequency: Minimum frequency to include

        Returns:
            Dictionary with pattern combination statistics
        """
        filtered_combinations = {
            combination: frequency
            for combination, frequency in self.pattern_combinations.items()
            if frequency >= min_frequency
        }

        # Sort by frequency
        sorted_combinations = sorted(
            filtered_combinations.items(), key=lambda item: item[1], reverse=True
        )

        return {
            "combinations": [
                {"patterns": combination, "frequency": frequency}
                for combination, frequency in sorted_combinations
            ],
            "total_combinations": len(filtered_combinations),
            "min_frequency": min_frequency,
        }

    def get_temporal_patterns(
        self, pattern_type: str, window_hours: int = 24
    ) -> TemporalPatternStats | TemporalPatternError:
        """
        Get temporal pattern analysis for a specific pattern.

        Args:
            pattern_type: Pattern type to analyze
            window_hours: Time window in hours for analysis

        Returns:
            Dictionary with temporal pattern statistics
        """
        if pattern_type not in self.temporal_patterns:
            return {"error": f"No temporal data for pattern: {pattern_type}"}

        temporal_data = self.temporal_patterns[pattern_type]
        if not temporal_data:
            return {"error": f"Empty temporal data for pattern: {pattern_type}"}

        # Filter by time window
        current_time = time.time()
        window_seconds = window_hours * 3600
        window_start = current_time - window_seconds

        recent_data = [
            (timestamp, success)
            for timestamp, success in temporal_data
            if timestamp >= window_start
        ]

        if not recent_data:
            return {
                "error": (
                    f"No data in the last {window_hours} hours "
                    f"for pattern: {pattern_type}"
                )
            }

        # Calculate temporal statistics
        timestamps, successes = zip(*recent_data)
        success_rate = sum(successes) / len(successes) * 100

        # Calculate detection frequency (detections per hour)
        time_span_hours = (max(timestamps) - min(timestamps)) / 3600
        detection_frequency = (
            len(recent_data) / time_span_hours if time_span_hours > 0 else 0.0
        )

        return {
            "pattern_type": pattern_type,
            "window_hours": window_hours,
            "total_detections": len(recent_data),
            "success_rate": float(success_rate),
            "detection_frequency_per_hour": float(detection_frequency),
            "time_span_hours": float(time_span_hours),
        }

    def get_overall_statistics(self) -> OverallStatistics:
        """
        Get overall pattern statistics summary.

        Returns:
            Dictionary with overall statistics
        """
        uptime_hours = (time.time() - self.start_time) / 3600

        stats: OverallStatistics = {
            "total_detections": self.total_detections,
            "total_successful_detections": self.total_successful_detections,
            "overall_success_rate": (
                self.total_successful_detections / self.total_detections * 100
            )
            if self.total_detections > 0
            else 0.0,
            "unique_patterns": len(self.pattern_counts),
            "uptime_hours": float(uptime_hours),
            "detection_rate_per_hour": self.total_detections / uptime_hours
            if uptime_hours > 0
            else 0.0,
            "most_frequent_pattern": max(
                self.pattern_counts.items(), key=lambda item: item[1]
            )
            if self.pattern_counts
            else None,
            "least_frequent_pattern": min(
                self.pattern_counts.items(), key=lambda item: item[1]
            )
            if self.pattern_counts
            else None,
        }

        # Add top patterns by success rate
        success_rates = self.get_success_rates()
        if success_rates:
            stats["top_patterns_by_success"] = sorted(
                success_rates.items(), key=lambda item: item[1], reverse=True
            )[:5]
            stats["bottom_patterns_by_success"] = sorted(
                success_rates.items(), key=lambda item: item[1]
            )[:5]

        return stats

    def reset_statistics(self) -> None:
        """
        Reset all pattern statistics.
        """
        self.pattern_counts.clear()
        self.pattern_success_counts.clear()
        self.pattern_failure_counts.clear()
        self.pattern_strengths.clear()
        self.pattern_confidences.clear()
        self.pattern_accuracies.clear()
        self.pattern_combinations.clear()
        self.pattern_correlations.clear()
        self.detection_history.clear()
        self.temporal_patterns.clear()

        self.total_detections = 0
        self.total_successful_detections = 0
        self.start_time = time.time()

        self.logger.info("Pattern statistics reset")

    def _get_single_pattern_stats(self, pattern_type: str) -> dict[str, object]:
        """
        Get statistics for a single pattern type.

        Args:
            pattern_type: Pattern type

        Returns:
            Dictionary with pattern statistics
        """
        total_count = self.pattern_counts.get(pattern_type, 0)
        success_count = self.pattern_success_counts.get(pattern_type, 0)

        stats: dict[str, object] = {
            "total_detections": total_count,
            "successful_detections": success_count,
            "failed_detections": self.pattern_failure_counts.get(pattern_type, 0),
            "success_rate": (success_count / total_count * 100)
            if total_count > 0
            else 0.0,
        }

        # Add signal quality metrics
        if pattern_type in self.pattern_strengths and self.pattern_strengths[pattern_type]:
            strengths = self.pattern_strengths[pattern_type]
            stats["signal_strength"] = {
                "avg": statistics.mean(strengths),
                "min": min(strengths),
                "max": max(strengths),
                "samples": len(strengths),
            }

        if (
            pattern_type in self.pattern_confidences
            and self.pattern_confidences[pattern_type]
        ):
            confidences = self.pattern_confidences[pattern_type]
            stats["signal_confidence"] = {
                "avg": statistics.mean(confidences),
                "min": min(confidences),
                "max": max(confidences),
                "samples": len(confidences),
            }

        # Add accuracy metrics
        if pattern_type in self.pattern_accuracies and self.pattern_accuracies[pattern_type]:
            accuracies = self.pattern_accuracies[pattern_type]
            accuracy_rate = sum(accuracies) / len(accuracies) * 100
            stats["accuracy_rate"] = accuracy_rate
            stats["accuracy_samples"] = len(accuracies)

        return stats
