"""
Protocol definitions for performance metrics.

This module provides protocols that define common interfaces
for different types of performance metrics across the system.
"""

from typing import Any, Protocol

class PerformanceMetricsProtocol(Protocol):
    """Protocol for all performance metrics classes."""

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary representation."""
        ...

class TradableMetricsProtocol(PerformanceMetricsProtocol):
    """Protocol for trading-related performance metrics."""

    # Common trading metrics
    win_rate: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float

class SystemMetricsProtocol(PerformanceMetricsProtocol):
    """Protocol for system performance metrics."""

    # Common system metrics
    avg_latency_ms: float
    memory_usage_gb: float
    cpu_usage_percent: float

class MLMetricsProtocol(PerformanceMetricsProtocol):
    """Protocol for machine learning performance metrics."""

    # Common ML metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
