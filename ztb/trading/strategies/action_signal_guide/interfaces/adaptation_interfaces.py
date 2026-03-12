"""
Real-time Adaptation Interfaces for Action Signal Guide.

Defines interfaces for streaming data processing and real-time parameter
adaptation while keeping payload typing consistent across modules.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Callable

import pandas as pd

from ztb.trading.strategies.action_signal_guide.interfaces.common_types import (
    ConstraintMap,
    IActionSignalGuideInterface,
    MetadataMap,
    MetricsMap,
    ObjectList,
    PayloadMap,
    PayloadRecords,
)

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
    data: object
    timestamp: float
    metadata: MetadataMap

@dataclass
class AdaptationDecision:
    """Decision made by adaptation system."""

    trigger: AdaptationTrigger
    parameters_to_update: PayloadMap
    confidence: float
    expected_impact: dict[str, float]
    timestamp: float

@dataclass
class ProcessingResult:
    """Result of data processing operation."""

    success: bool
    processed_count: int
    processing_time: float
    quality_score: float
    metadata: MetadataMap

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

    input_signals: ObjectList
    output_actions: ObjectList
    market_outcomes: list[float]
    performance_metrics: MetricsMap
    adaptation_history: list[AdaptationDecision]

class IStreamingProcessor(IActionSignalGuideInterface):
    """Interface for processing streaming market data and signals."""

    @abstractmethod
    def process_streaming_data(self, data_queue: Queue[StreamingDataPoint]) -> None:
        """Process streaming data from queue."""

    @abstractmethod
    def register_data_handler(
        self,
        data_type: StreamingDataType,
        handler: Callable[[StreamingDataPoint], None],
    ) -> None:
        """Register handler for specific data type."""

    @abstractmethod
    def get_processed_data(
        self, data_type: StreamingDataType, lookback_period: int = 100
    ) -> pd.DataFrame:
        """Get processed data for specified type and period."""

class IAdaptiveThresholds(IActionSignalGuideInterface):
    """Interface for adaptive threshold management."""

    @abstractmethod
    def update_thresholds(
        self, performance_data: MetricsMap, market_conditions: PayloadMap
    ) -> PayloadMap:
        """Update adaptive thresholds based on performance and market conditions."""

    @abstractmethod
    def get_current_thresholds(self) -> PayloadMap:
        """Get current adaptive threshold values."""

    @abstractmethod
    def reset_thresholds_to_default(self) -> None:
        """Reset thresholds to default values."""

    @abstractmethod
    def get_threshold_history(self) -> PayloadRecords:
        """Get historical threshold values."""

class IPerformanceMonitor(IActionSignalGuideInterface):
    """Interface for real-time performance monitoring."""

    @abstractmethod
    def monitor_performance(
        self, current_metrics: MetricsMap, baseline_metrics: MetricsMap
    ) -> PayloadMap:
        """Monitor current performance against baseline."""

    @abstractmethod
    def detect_performance_anomalies(
        self, metrics_history: PayloadRecords, threshold: float = 2.0
    ) -> PayloadRecords:
        """Detect performance anomalies using statistical methods."""

    @abstractmethod
    def generate_performance_report(self) -> PayloadMap:
        """Generate comprehensive performance report."""

class IFeedbackLoop(IActionSignalGuideInterface):
    """Interface for feedback loop processing and learning."""

    @abstractmethod
    def process_feedback(self, feedback_data: FeedbackLoopData) -> AdaptationDecision:
        """Process feedback data and generate adaptation decisions."""

    @abstractmethod
    def apply_adaptation(self, decision: AdaptationDecision) -> bool:
        """Apply adaptation decision to the system."""

    @abstractmethod
    def validate_adaptation(
        self, decision: AdaptationDecision, validation_period: int = 100
    ) -> PayloadMap:
        """Validate adaptation decision effectiveness."""

    @abstractmethod
    def get_feedback_statistics(self) -> MetricsMap:
        """Get feedback loop processing statistics."""

class IRealTimeOptimizer(IActionSignalGuideInterface):
    """Interface for real-time optimization algorithms."""

    @abstractmethod
    def optimize_realtime(
        self, current_state: PayloadMap, constraints: ConstraintMap
    ) -> PayloadMap:
        """Perform real-time optimization given current state and constraints."""

    @abstractmethod
    def update_optimization_model(self, new_data: PayloadMap) -> None:
        """Update optimization model with new data."""

    @abstractmethod
    def get_optimization_trajectory(self) -> PayloadRecords:
        """Get optimization parameter trajectory."""

class IAdaptiveController(IActionSignalGuideInterface):
    """Interface for adaptive control systems."""

    @abstractmethod
    def control_system_parameters(
        self, system_state: PayloadMap, reference_performance: MetricsMap
    ) -> PayloadMap:
        """Control system parameters using adaptive control theory."""

    @abstractmethod
    def estimate_system_dynamics(self, state_history: PayloadRecords) -> PayloadMap:
        """Estimate system dynamics from historical state data."""

    @abstractmethod
    def get_control_performance(self) -> MetricsMap:
        """Get adaptive control performance metrics."""

