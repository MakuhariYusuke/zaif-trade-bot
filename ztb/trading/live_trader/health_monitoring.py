"""Health monitoring implementation for live trading."""

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

import psutil

from ztb.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ztb.trading.live_trader.live_trader import LiveTrader

class HealthMonitoring:
    """Handles health status monitoring for live trading."""

    def __init__(self, live_trader: "LiveTrader"):
        """Initialize health monitoring with reference to live trader."""
        self.live_trader = live_trader
        self.logger = get_logger(__name__)

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status for monitoring."""
        try:
            # Basic status
            status = "healthy"
            issues = []

            # Check model
            if self.live_trader.model is None:
                status = "critical"
                issues.append("Model not loaded")

            # Check exchange adapter
            if self.live_trader.exchange_adapter is None:
                status = "critical"
                issues.append("Exchange adapter not available")

            # Check position validity
            if not (-1 <= self.live_trader.position <= 1):
                status = "error"
                issues.append(f"Invalid position: {self.live_trader.position}")

            # Check price data freshness
            price_age_seconds = 0
            if hasattr(self.live_trader, "_last_price_time"):
                price_age_seconds = (
                    datetime.now() - self.live_trader._last_price_time
                ).total_seconds()
                if price_age_seconds > 300:  # 5 minutes
                    status = "warning"
                    issues.append(f"Price data stale: {price_age_seconds:.0f}s old")

            # Check memory usage
            try:
                process = psutil.Process(os.getpid())
                memory_percent = process.memory_percent()
                if memory_percent > 90:
                    status = "critical"
                    issues.append(f"High memory usage: {memory_percent:.1f}%")
                elif memory_percent > 80:
                    if status == "healthy":
                        status = "warning"
                    issues.append(f"Elevated memory usage: {memory_percent:.1f}%")
            except ImportError:
                issues.append("psutil not available for memory monitoring")
            except Exception as e:
                issues.append(f"Memory check failed: {e}")

            # Check feature computation
            try:
                test_features = self.live_trader._compute_features()
                if len(test_features) == 0:
                    status = "error"
                    issues.append("Feature computation returning empty array")
            except Exception as e:
                status = "error"
                issues.append(f"Feature computation failed: {e}")

            # Check API connectivity (if adapter available)
            api_status = "unknown"
            if self.live_trader.exchange_adapter:
                try:
                    # Quick connectivity test
                    test_price = self.live_trader._get_current_price()
                    if test_price > 0:
                        api_status = "connected"
                    else:
                        api_status = "error"
                        if status == "healthy":
                            status = "warning"
                        issues.append("API connectivity test failed")
                except Exception as e:
                    api_status = "error"
                    if status == "healthy":
                        status = "warning"
                    issues.append(f"API connectivity error: {e}")

            return {
                "status": status,
                "issues": issues,
                "position": self.live_trader.position,
                "total_pnl": self.live_trader.total_pnl,
                "trades_count": self.live_trader.trades_count,
                "last_price": self.live_trader._last_valid_price,
                "price_age_seconds": price_age_seconds,
                "model_loaded": self.live_trader.model is not None,
                "adapter_available": self.live_trader.exchange_adapter is not None,
                "api_status": api_status,
                "memory_percent": memory_percent
                if "memory_percent" in locals()
                else None,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "status": "critical",
                "issues": [f"Health check failed: {e}"],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
