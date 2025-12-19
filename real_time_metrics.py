"""Compatibility shim to expose RealTimeMetrics at top-level for tests.

Some tests import `RealTimeMetrics` directly from `real_time_metrics`. The
implementation lives under `ztb.trading.production.real_time_metrics` in the
package; re-export it here for import-time compatibility.
"""
from ztb.trading.production.real_time_metrics import RealTimeMetrics

__all__ = ["RealTimeMetrics"]
