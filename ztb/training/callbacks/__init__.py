"""
Advanced Callbacks for Training.

訓練プロセスを高度に制御するためのコールバック集。
"""

# Legacy callbacks (for backward compatibility)
from ztb.training.callbacks.advanced_callbacks import (
    BestModelSaveCallback,
    EarlyStoppingCallback,
)
from ztb.training.callbacks.core.callback_implementations import (
    CheckpointCallback,
    LoggingCallback,
    MetricsCallback,
    ProgressCallback,
)

# Core callback system
from ztb.training.callbacks.core.modern_callback_system import (
    BaseCallback,
    CallbackContext,
    CallbackEvent,
    CallbackManager,
    CallbackPriority,
    CallbackResult,
)

# Monitoring system
from ztb.training.callbacks.monitoring.metrics_collector import (
    MetricDefinition,
    MetricsCollector,
    MetricValue,
    get_global_metrics_collector,
)
from ztb.training.callbacks.monitoring.real_time_monitor import (
    MonitorAlert,
    MonitorConfig,
    RealTimeMonitor,
    create_high_cpu_alert,
    create_high_memory_alert,
    create_training_stuck_alert,
    get_global_monitor,
)

__all__ = [
    # Core system
    "CallbackManager",
    "CallbackEvent",
    "CallbackPriority",
    "BaseCallback",
    "CallbackContext",
    "CallbackResult",
    # Implementations
    "ProgressCallback",
    "CheckpointCallback",
    "MetricsCallback",
    "LoggingCallback",
    # Monitoring
    "MetricsCollector",
    "MetricDefinition",
    "MetricValue",
    "get_global_metrics_collector",
    "RealTimeMonitor",
    "MonitorConfig",
    "MonitorAlert",
    "get_global_monitor",
    "create_high_cpu_alert",
    "create_high_memory_alert",
    "create_training_stuck_alert",
    # Legacy
    "EarlyStoppingCallback",
    "BestModelCallback",
]
