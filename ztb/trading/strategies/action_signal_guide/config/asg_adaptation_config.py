"""
Real-time Adaptation Configuration for Action Signal Guide.

This module provides configuration management for real-time adaptation components.
"""

from dataclasses import dataclass, field
from typing import TypedDict

from ..interfaces.adaptation_interfaces import AdaptationTrigger

class TriggerConditionsPayload(TypedDict):
    """Adaptation trigger conditions."""

    performance_degradation: float
    regime_stability: int
    time_interval: int
    signal_quality_drop: float
    volatility_spike: float

class ProcessingLimitsPayload(TypedDict):
    """Processing/resource limits."""

    buffer_size: int
    max_queue_size: int
    max_workers: int
    max_memory: int
    max_optimization_time: int

class AdaptationSchedulePayload(TypedDict):
    """Adaptation schedule details."""

    processing_interval: float
    threshold_reset_frequency: int
    baseline_update_frequency: int
    cooldown_period: int

@dataclass
class StreamingProcessorConfig:
    """Configuration for streaming data processing."""

    enabled: bool = True
    buffer_size: int = 1000
    processing_interval: float = 0.1  # seconds
    max_queue_size: int = 5000
    enable_parallel_processing: bool = True
    max_workers: int = 4
    data_retention_period: int = 3600  # seconds

@dataclass
class AdaptiveThresholdsConfig:
    """Configuration for adaptive thresholds."""

    enabled: bool = True
    adaptation_rate: float = 0.1
    min_threshold_change: float = 0.01
    max_threshold_change: float = 0.2
    reset_frequency: int = 1000  # Reset every N updates
    threshold_bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "strength": (0.1, 0.9),
            "confidence": (0.2, 0.95),
            "reliability": (0.1, 0.8),
            "market_alignment": (0.0, 1.0),
        }
    )

@dataclass
class PerformanceMonitorConfig:
    """Configuration for performance monitoring."""

    enabled: bool = True
    monitoring_window: int = 100
    alert_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "accuracy_drop": 0.05,
            "sharpe_ratio_drop": 0.2,
            "max_drawdown_increase": 0.02,
        }
    )
    baseline_update_frequency: int = 500
    anomaly_detection_sensitivity: float = 2.0

@dataclass
class FeedbackLoopConfig:
    """Configuration for feedback loop processing."""

    enabled: bool = True
    feedback_window: int = 50
    learning_rate: float = 0.01
    adaptation_confidence_threshold: float = 0.7
    validation_period: int = 100
    max_consecutive_adaptations: int = 3
    cooldown_period: int = 50  # No adaptations for N periods after change

@dataclass
class RealTimeOptimizerConfig:
    """Configuration for real-time optimization."""

    enabled: bool = True
    optimization_algorithm: str = (
        "gradient_descent"  # gradient_descent, evolutionary, simulated_annealing
    )
    max_iterations: int = 50
    convergence_threshold: float = 1e-4
    step_size: float = 0.01
    bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "learning_rate": (1e-5, 1e-1),
            "adaptation_rate": (1e-3, 1e-1),
            "threshold": (0.1, 0.9),
        }
    )

@dataclass
class AdaptiveControllerConfig:
    """Configuration for adaptive control systems."""

    enabled: bool = True
    control_algorithm: str = "pid"  # pid, lqr, adaptive
    pid_gains: dict[str, float] = field(
        default_factory=lambda: {
            "kp": 0.5,  # Proportional gain
            "ki": 0.1,  # Integral gain
            "kd": 0.05,  # Derivative gain
        }
    )
    reference_tracking: bool = True
    disturbance_rejection: bool = True
    system_identification: bool = True

@dataclass
class AdaptationTriggerConfig:
    """Configuration for adaptation triggers."""

    enabled_triggers: list[AdaptationTrigger] = field(
        default_factory=lambda: [
            AdaptationTrigger.PERFORMANCE_DEGRADATION,
            AdaptationTrigger.MARKET_REGIME_CHANGE,
            AdaptationTrigger.TIME_BASED,
        ]
    )
    performance_degradation_threshold: float = 0.05
    regime_change_stability_period: int = 20
    time_based_interval: int = 300  # seconds
    signal_quality_drop_threshold: float = 0.1
    volatility_spike_threshold: float = 2.0

@dataclass
class RealTimeAdaptationConfig:
    """Main configuration for real-time adaptation."""

    enabled: bool = True
    streaming_processor: StreamingProcessorConfig = field(
        default_factory=StreamingProcessorConfig
    )
    adaptive_thresholds: AdaptiveThresholdsConfig = field(
        default_factory=AdaptiveThresholdsConfig
    )
    performance_monitor: PerformanceMonitorConfig = field(
        default_factory=PerformanceMonitorConfig
    )
    feedback_loop: FeedbackLoopConfig = field(default_factory=FeedbackLoopConfig)
    realtime_optimizer: RealTimeOptimizerConfig = field(
        default_factory=RealTimeOptimizerConfig
    )
    adaptive_controller: AdaptiveControllerConfig = field(
        default_factory=AdaptiveControllerConfig
    )
    triggers: AdaptationTriggerConfig = field(default_factory=AdaptationTriggerConfig)

    # Global settings
    adaptation_mode: str = "conservative"  # conservative, moderate, aggressive
    max_memory_usage: int = 512  # MB
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_persistence: bool = True
    state_save_path: str = "state/adaptation_state"

    def __post_init__(self):
        """Initialize configuration based on adaptation mode."""
        if self.adaptation_mode == "conservative":
            self._set_conservative_mode()
        elif self.adaptation_mode == "moderate":
            self._set_moderate_mode()
        elif self.adaptation_mode == "aggressive":
            self._set_aggressive_mode()

    def _set_conservative_mode(self):
        """set conservative adaptation parameters."""
        self.feedback_loop.learning_rate = 0.005
        self.adaptive_thresholds.adaptation_rate = 0.05
        self.triggers.performance_degradation_threshold = 0.1
        self.feedback_loop.adaptation_confidence_threshold = 0.8

    def _set_moderate_mode(self):
        """set moderate adaptation parameters."""
        self.feedback_loop.learning_rate = 0.01
        self.adaptive_thresholds.adaptation_rate = 0.1
        self.triggers.performance_degradation_threshold = 0.07
        self.feedback_loop.adaptation_confidence_threshold = 0.75

    def _set_aggressive_mode(self):
        """set aggressive adaptation parameters."""
        self.feedback_loop.learning_rate = 0.02
        self.adaptive_thresholds.adaptation_rate = 0.2
        self.triggers.performance_degradation_threshold = 0.05
        self.feedback_loop.adaptation_confidence_threshold = 0.7

    def get_trigger_conditions(self) -> TriggerConditionsPayload:
        """Get trigger conditions as dictionary."""
        return {
            "performance_degradation": self.triggers.performance_degradation_threshold,
            "regime_stability": self.triggers.regime_change_stability_period,
            "time_interval": self.triggers.time_based_interval,
            "signal_quality_drop": self.triggers.signal_quality_drop_threshold,
            "volatility_spike": self.triggers.volatility_spike_threshold,
        }

    def get_processing_limits(self) -> ProcessingLimitsPayload:
        """Get processing limits as dictionary."""
        return {
            "buffer_size": self.streaming_processor.buffer_size,
            "max_queue_size": self.streaming_processor.max_queue_size,
            "max_workers": self.streaming_processor.max_workers,
            "max_memory": self.max_memory_usage,
            "max_optimization_time": self.realtime_optimizer.max_iterations,
        }

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues: list[str] = []

        # Check learning rates
        if not 0 < self.feedback_loop.learning_rate <= 0.1:
            issues.append("learning_rate should be between 0 and 0.1")

        if not 0 < self.adaptive_thresholds.adaptation_rate <= 1:
            issues.append("adaptation_rate should be between 0 and 1")

        # Check thresholds
        for param, (
            min_val,
            max_val,
        ) in self.adaptive_thresholds.threshold_bounds.items():
            if min_val >= max_val:
                issues.append(f"Invalid bounds for {param}: min >= max")

        # Check buffer sizes
        if self.streaming_processor.buffer_size < 100:
            issues.append("buffer_size should be at least 100")

        if (
            self.streaming_processor.max_queue_size
            < self.streaming_processor.buffer_size
        ):
            issues.append("max_queue_size should be >= buffer_size")

        # Check performance thresholds
        for (
            threshold_name,
            threshold_value,
        ) in self.performance_monitor.alert_thresholds.items():
            if threshold_value < 0:
                issues.append(f"{threshold_name} threshold should be >= 0")

        return issues

    def get_adaptation_schedule(self) -> AdaptationSchedulePayload:
        """Get adaptation schedule configuration."""
        return {
            "processing_interval": self.streaming_processor.processing_interval,
            "threshold_reset_frequency": self.adaptive_thresholds.reset_frequency,
            "baseline_update_frequency": self.performance_monitor.baseline_update_frequency,
            "cooldown_period": self.feedback_loop.cooldown_period,
        }
