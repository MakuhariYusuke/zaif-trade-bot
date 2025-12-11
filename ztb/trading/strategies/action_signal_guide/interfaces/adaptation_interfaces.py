"""
Real-time Adaptation Interfaces for Action Signal Guide.

This module defines interfaces for streaming data processing and real-time parameter adaptation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Any, Callable, Dict, List

import pandas as pd


class AdaptationTrigger(Enum):
    """Triggers for parameter adaptation."""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    MARKET_REGIME_CHANGE = "market_regime_change"
    TIME_BASED = "time_based"
    SIGNAL_QUALITY_DROP = "signal_quality_drop"
    VOLATILITY_SPIKE = "volatility_spike"


class StreamingDataType(Enum):
    """Types of streaming data."""

    MARKET_DATA = "market_data"
    SIGNAL_DATA = "signal_data"
    PERFORMANCE_DATA = "performance_data"
    EXTERNAL_FEED = "external_feed"


@dataclass
class StreamingDataPoint:
    """Single data point in streaming context."""

    data_type: StreamingDataType
    data: Any
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class AdaptationDecision:
    """Decision made by adaptation system."""

    trigger: AdaptationTrigger
    parameters_to_update: Dict[str, Any]
    confidence: float
    expected_impact: Dict[str, float]
    timestamp: float


@dataclass
class ProcessingResult:
    """Result of data processing operation."""

    success: bool
    processed_count: int
    processing_time: float
    quality_score: float
    metadata: Dict[str, Any]


@dataclass
class PerformanceMetrics:
    """Performance metrics for adaptation monitoring."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    processing_time: float
    memory_usage: float
    throughput: float
    timestamp: float


@dataclass
class FeedbackLoopData:
    """Data for feedback loop processing."""

    input_signals: List[Any]
    output_actions: List[Any]
    market_outcomes: List[float]
    performance_metrics: Dict[str, Any]
    adaptation_history: List[AdaptationDecision]


class IStreamingProcessor(ABC):
    """Interface for processing streaming market data and signals."""

    @abstractmethod
    def process_streaming_data(self, data_queue: Queue) -> None:
        """
        Process streaming data from queue.

        Args:
            data_queue: Queue containing streaming data points
        """
        pass

    @abstractmethod
    def register_data_handler(
        self,
        data_type: StreamingDataType,
        handler: Callable[[StreamingDataPoint], None],
    ) -> None:
        """
        Register handler for specific data type.

        Args:
            data_type: Type of data to handle
            handler: Handler function
        """
        pass

    @abstractmethod
    def get_processed_data(
        self, data_type: StreamingDataType, lookback_period: int = 100
    ) -> pd.DataFrame:
        """
        Get processed data for specified type and period.

        Args:
            data_type: Type of processed data to retrieve
            lookback_period: Number of periods to look back

        Returns:
            Processed data DataFrame
        """
        pass


class IAdaptiveThresholds(ABC):
    """Interface for adaptive threshold management."""

    @abstractmethod
    def update_thresholds(
        self, performance_data: Dict[str, Any], market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update adaptive thresholds based on performance and market conditions.

        Args:
            performance_data: Current performance metrics
            market_conditions: Current market conditions

        Returns:
            Updated threshold values
        """
        pass

    @abstractmethod
    def get_current_thresholds(self) -> Dict[str, Any]:
        """Get current adaptive threshold values."""
        pass

    @abstractmethod
    def reset_thresholds_to_default(self) -> None:
        """Reset thresholds to default values."""
        pass

    @abstractmethod
    def get_threshold_history(self) -> List[Dict[str, Any]]:
        """Get historical threshold values."""
        pass


class IPerformanceMonitor(ABC):
    """Interface for real-time performance monitoring."""

    @abstractmethod
    def monitor_performance(
        self, current_metrics: Dict[str, Any], baseline_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Monitor current performance against baseline.

        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics

        Returns:
            Performance analysis results
        """
        pass

    @abstractmethod
    def detect_performance_anomalies(
        self, metrics_history: List[Dict[str, Any]], threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect performance anomalies using statistical methods.

        Args:
            metrics_history: Historical performance metrics
            threshold: Anomaly detection threshold (standard deviations)

        Returns:
            List of detected anomalies
        """
        pass

    @abstractmethod
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        pass


class IFeedbackLoop(ABC):
    """Interface for feedback loop processing and learning."""

    @abstractmethod
    def process_feedback(self, feedback_data: FeedbackLoopData) -> AdaptationDecision:
        """
        Process feedback data and generate adaptation decisions.

        Args:
            feedback_data: Feedback loop data

        Returns:
            Adaptation decision
        """
        pass

    @abstractmethod
    def apply_adaptation(self, decision: AdaptationDecision) -> bool:
        """
        Apply adaptation decision to the system.

        Args:
            decision: Adaptation decision to apply

        Returns:
            True if adaptation was successful
        """
        pass

    @abstractmethod
    def validate_adaptation(
        self, decision: AdaptationDecision, validation_period: int = 100
    ) -> Dict[str, Any]:
        """
        Validate adaptation decision effectiveness.

        Args:
            decision: Adaptation decision to validate
            validation_period: Validation period in data points

        Returns:
            Validation results
        """
        pass

    @abstractmethod
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback loop processing statistics."""
        pass


class IRealTimeOptimizer(ABC):
    """Interface for real-time optimization algorithms."""

    @abstractmethod
    def optimize_realtime(
        self, current_state: Dict[str, Any], constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform real-time optimization given current state and constraints.

        Args:
            current_state: Current system state
            constraints: Optimization constraints

        Returns:
            Optimized parameters
        """
        pass

    @abstractmethod
    def update_optimization_model(self, new_data: Dict[str, Any]) -> None:
        """
        Update optimization model with new data.

        Args:
            new_data: New data for model update
        """
        pass

    @abstractmethod
    def get_optimization_trajectory(self) -> List[Dict[str, Any]]:
        """Get optimization parameter trajectory."""
        pass


class IAdaptiveController(ABC):
    """Interface for adaptive control systems."""

    @abstractmethod
    def control_system_parameters(
        self, system_state: Dict[str, Any], reference_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Control system parameters using adaptive control theory.

        Args:
            system_state: Current system state
            reference_performance: Reference performance targets

        Returns:
            Control actions (parameter adjustments)
        """
        pass

    @abstractmethod
    def estimate_system_dynamics(
        self, state_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Estimate system dynamics from historical state data.

        Args:
            state_history: Historical system states

        Returns:
            Estimated system dynamics parameters
        """
        pass

    @abstractmethod
    def get_control_performance(self) -> Dict[str, Any]:
        """Get adaptive control performance metrics."""
        pass
