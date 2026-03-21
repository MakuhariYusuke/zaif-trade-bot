"""Compatibility shim for shared fill-record metric helpers.

Canonical implementation lives in :mod:`ztb.metrics.record_metrics`.
"""

from ztb.metrics.record_metrics import (
    BaseMetrics,
    ExtendedMetrics,
    MetricRecord,
    MetricsAccumulator,
    compute_base_metrics,
    compute_extended_metrics,
)

__all__ = [
    "MetricRecord",
    "BaseMetrics",
    "ExtendedMetrics",
    "MetricsAccumulator",
    "compute_base_metrics",
    "compute_extended_metrics",
]
