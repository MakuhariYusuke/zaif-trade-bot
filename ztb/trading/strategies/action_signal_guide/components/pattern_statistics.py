"""
PatternStatistics Component.

This component is responsible for tracking pattern recognition statistics.
Follows Single Responsibility Principle by focusing only on pattern statistics.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import statistics

from ztb.utils.logging_utils import get_logger

from .interfaces import IPatternStatistics


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

    def __init__(self, max_history_size: int = 10000):
        """
        Initialize PatternStatistics.

        Args:
            max_history_size: Maximum number of historical records to keep
        """
        self.max_history_size = max_history_size
        self.logger = get_logger("ztb.trading.strategies.pattern_statistics")

        # Pattern detection statistics
        self.pattern_counts: Dict[str, int] = defaultdict(int)
        self.pattern_success_counts: Dict[str, int] = defaultdict(int)
        self.pattern_failure_counts: Dict[str, int] = defaultdict(int)

        # Signal quality tracking
        self.pattern_strengths: Dict[str, List[float]] = defaultdict(list)
        self.pattern_confidences: Dict[str, List[float]] = defaultdict(list)
        self.pattern_accuracies: Dict[str, List[bool]] = defaultdict(list)

        # Pattern combinations
        self.pattern_combinations: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.pattern_correlations: Dict[Tuple[str, str], float] = {}

        # Historical data
        self.detection_history: List[Dict[str, Any]] = []
        self.temporal_patterns: Dict[str, List[Tuple[int, bool]]] = defaultdict(list)  # (timestamp, success)

        # Performance metrics
        self.start_time = time.time()
        self.total_detections = 0
        self.total_successful_detections = 0

    def record_pattern_detection(self, pattern_type: str, detected: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
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

        # Record in history
        history_entry = {
            'timestamp': time.time(),
            'pattern_type': pattern_type,
            'detected': detected,
            'metadata': metadata or {}
        }
        self.detection_history.append(history_entry)

        # Maintain history size limit
        if len(self.detection_history) > self.max_history_size:
            self.detection_history = self.detection_history[-self.max_history_size:]

        # Record temporal pattern
        timestamp = int(time.time())
        self.temporal_patterns[pattern_type].append((timestamp, detected))

        # Keep only recent temporal data (last 1000 entries per pattern)
        if len(self.temporal_patterns[pattern_type]) > 1000:
            self.temporal_patterns[pattern_type] = self.temporal_patterns[pattern_type][-500:]

    def record_pattern_signal(self, pattern_type: str, signal: Any) -> None:
        """
        Record pattern signal metrics.

        Args:
            pattern_type: Type of pattern
            signal: Signal object with strength and confidence
        """
        try:
            strength = getattr(signal, 'strength', 0.0)
            confidence = getattr(signal, 'confidence', 0.0)

            self.pattern_strengths[pattern_type].append(strength)
            self.pattern_confidences[pattern_type].append(confidence)

            # Keep only recent samples (last 500 per pattern)
            if len(self.pattern_strengths[pattern_type]) > 500:
                self.pattern_strengths[pattern_type] = self.pattern_strengths[pattern_type][-250:]
            if len(self.pattern_confidences[pattern_type]) > 500:
                self.pattern_confidences[pattern_type] = self.pattern_confidences[pattern_type][-250:]

        except Exception as e:
            self.logger.warning(f"Failed to record pattern signal metrics: {e}")

    def update_pattern_strength_stats(self, pattern_type: str, strength: float) -> None:
        """
        Update pattern strength statistics.

        Args:
            pattern_type: Type of pattern
            strength: Signal strength value
        """
        self.pattern_strengths[pattern_type].append(strength)

        # Keep only recent samples (last 500 per pattern)
        if len(self.pattern_strengths[pattern_type]) > 500:
            self.pattern_strengths[pattern_type] = self.pattern_strengths[pattern_type][-250:]

    def record_pattern_accuracy(self, pattern_type: str, accurate: bool) -> None:
        """
        Record pattern prediction accuracy.

        Args:
            pattern_type: Type of pattern
            accurate: Whether the pattern prediction was accurate
        """
        self.pattern_accuracies[pattern_type].append(accurate)

        # Keep only recent samples
        if len(self.pattern_accuracies[pattern_type]) > 1000:
            self.pattern_accuracies[pattern_type] = self.pattern_accuracies[pattern_type][-500:]

    def record_pattern_combination(self, pattern_types: List[str]) -> None:
        """
        Record combination of patterns detected together.

        Args:
            pattern_types: List of pattern types detected simultaneously
        """
        if len(pattern_types) > 1:
            # Sort for consistent key
            combination_key = tuple(sorted(pattern_types))
            self.pattern_combinations[combination_key] += 1

    def get_pattern_statistics(self, pattern_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive pattern statistics.

        Args:
            pattern_type: Specific pattern type, or None for all

        Returns:
            Dictionary with pattern statistics
        """
        if pattern_type:
            return self._get_single_pattern_stats(pattern_type)
        else:
            return {pt: self._get_single_pattern_stats(pt) for pt in self.pattern_counts.keys()}

    def get_detection_frequencies(self) -> Dict[str, float]:
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

    def get_success_rates(self) -> Dict[str, float]:
        """
        Get success rates for all patterns.

        Returns:
            Dictionary mapping pattern types to success rates
        """
        success_rates = {}
        for pattern_type, total_count in self.pattern_counts.items():
            success_count = self.pattern_success_counts[pattern_type]
            success_rates[pattern_type] = (success_count / total_count * 100) if total_count > 0 else 0.0

        return success_rates

    def get_accuracy_rates(self) -> Dict[str, float]:
        """
        Get accuracy rates for all patterns.

        Returns:
            Dictionary mapping pattern types to accuracy rates
        """
        accuracy_rates = {}
        for pattern_type, accuracies in self.pattern_accuracies.items():
            if accuracies:
                accuracy_rates[pattern_type] = (sum(accuracies) / len(accuracies) * 100)
            else:
                accuracy_rates[pattern_type] = 0.0

        return accuracy_rates

    def get_pattern_combinations(self, min_frequency: int = 2) -> Dict[str, Any]:
        """
        Get frequently occurring pattern combinations.

        Args:
            min_frequency: Minimum frequency to include

        Returns:
            Dictionary with pattern combination statistics
        """
        filtered_combinations = {
            combo: freq for combo, freq in self.pattern_combinations.items()
            if freq >= min_frequency
        }

        # Sort by frequency
        sorted_combinations = sorted(filtered_combinations.items(), key=lambda x: x[1], reverse=True)

        return {
            'combinations': [{'patterns': combo, 'frequency': freq} for combo, freq in sorted_combinations],
            'total_combinations': len(filtered_combinations),
            'min_frequency': min_frequency
        }

    def get_temporal_patterns(self, pattern_type: str, window_hours: int = 24) -> Dict[str, Any]:
        """
        Get temporal pattern analysis for a specific pattern.

        Args:
            pattern_type: Pattern type to analyze
            window_hours: Time window in hours for analysis

        Returns:
            Dictionary with temporal pattern statistics
        """
        if pattern_type not in self.temporal_patterns:
            return {'error': f'No temporal data for pattern: {pattern_type}'}

        temporal_data = self.temporal_patterns[pattern_type]
        if not temporal_data:
            return {'error': f'Empty temporal data for pattern: {pattern_type}'}

        # Filter by time window
        current_time = time.time()
        window_seconds = window_hours * 3600
        window_start = current_time - window_seconds

        recent_data = [(ts, success) for ts, success in temporal_data if ts >= window_start]

        if not recent_data:
            return {'error': f'No data in the last {window_hours} hours for pattern: {pattern_type}'}

        # Calculate temporal statistics
        timestamps, successes = zip(*recent_data)
        success_rate = sum(successes) / len(successes) * 100

        # Calculate detection frequency (detections per hour)
        time_span_hours = (max(timestamps) - min(timestamps)) / 3600
        detection_frequency = len(recent_data) / time_span_hours if time_span_hours > 0 else 0

        return {
            'pattern_type': pattern_type,
            'window_hours': window_hours,
            'total_detections': len(recent_data),
            'success_rate': success_rate,
            'detection_frequency_per_hour': detection_frequency,
            'time_span_hours': time_span_hours
        }

    def get_overall_statistics(self) -> Dict[str, Any]:
        """
        Get overall pattern statistics summary.

        Returns:
            Dictionary with overall statistics
        """
        uptime_hours = (time.time() - self.start_time) / 3600

        stats = {
            'total_detections': self.total_detections,
            'total_successful_detections': self.total_successful_detections,
            'overall_success_rate': (self.total_successful_detections / self.total_detections * 100) if self.total_detections > 0 else 0,
            'unique_patterns': len(self.pattern_counts),
            'uptime_hours': uptime_hours,
            'detection_rate_per_hour': self.total_detections / uptime_hours if uptime_hours > 0 else 0,
            'most_frequent_pattern': max(self.pattern_counts.items(), key=lambda x: x[1]) if self.pattern_counts else None,
            'least_frequent_pattern': min(self.pattern_counts.items(), key=lambda x: x[1]) if self.pattern_counts else None,
        }

        # Add top patterns by success rate
        success_rates = self.get_success_rates()
        if success_rates:
            stats['top_patterns_by_success'] = sorted(success_rates.items(), key=lambda x: x[1], reverse=True)[:5]
            stats['bottom_patterns_by_success'] = sorted(success_rates.items(), key=lambda x: x[1])[:5]

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

    def _get_single_pattern_stats(self, pattern_type: str) -> Dict[str, Any]:
        """
        Get statistics for a single pattern type.

        Args:
            pattern_type: Pattern type

        Returns:
            Dictionary with pattern statistics
        """
        total_count = self.pattern_counts.get(pattern_type, 0)
        success_count = self.pattern_success_counts.get(pattern_type, 0)

        stats = {
            'total_detections': total_count,
            'successful_detections': success_count,
            'failed_detections': self.pattern_failure_counts.get(pattern_type, 0),
            'success_rate': (success_count / total_count * 100) if total_count > 0 else 0,
        }

        # Add signal quality metrics
        if pattern_type in self.pattern_strengths and self.pattern_strengths[pattern_type]:
            strengths = self.pattern_strengths[pattern_type]
            stats['signal_strength'] = {
                'avg': statistics.mean(strengths),
                'min': min(strengths),
                'max': max(strengths),
                'samples': len(strengths)
            }

        if pattern_type in self.pattern_confidences and self.pattern_confidences[pattern_type]:
            confidences = self.pattern_confidences[pattern_type]
            stats['signal_confidence'] = {
                'avg': statistics.mean(confidences),
                'min': min(confidences),
                'max': max(confidences),
                'samples': len(confidences)
            }

        # Add accuracy metrics
        if pattern_type in self.pattern_accuracies and self.pattern_accuracies[pattern_type]:
            accuracies = self.pattern_accuracies[pattern_type]
            accuracy_rate = sum(accuracies) / len(accuracies) * 100
            stats['accuracy_rate'] = accuracy_rate
            stats['accuracy_samples'] = len(accuracies)

        return stats