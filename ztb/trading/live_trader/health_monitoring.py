"""Health monitoring implementation for live trading."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict
import asyncio
import psutil
import os
import time

from ztb.utils.logging_utils import get_logger
from ztb.utils.health_monitor import HealthChecker, HealthStatus, HealthCheckResult

if TYPE_CHECKING:
    from ztb.trading.live_trader.live_trader import LiveTrader


class HealthMonitoring:
    """Handles health status monitoring for live trading."""

    def __init__(self, live_trader: 'LiveTrader'):
        """Initialize health monitoring with reference to live trader."""
        self.live_trader = live_trader
        self.logger = get_logger(__name__)
        self.health_checker = HealthChecker()
        self.health_checker.setup_default_checks()

        # Register trading-specific health checks
        self.health_checker.register_check("trading_model", self._check_trading_model)
        self.health_checker.register_check("exchange_adapter", self._check_exchange_adapter)
        self.health_checker.register_check("position_validity", self._check_position_validity)
        self.health_checker.register_check("price_data_freshness", self._check_price_data_freshness)
        self.health_checker.register_check("feature_computation", self._check_feature_computation)
        self.health_checker.register_check("api_connectivity", self._check_api_connectivity)

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status for monitoring."""
        try:
            # Use the new health checker for comprehensive monitoring
            summary = self.health_checker.get_health_summary()

            # Add trading-specific information
            trading_info = {
                "position": self.live_trader.position,
                "total_pnl": self.live_trader.total_pnl,
                "trades_count": self.live_trader.trades_count,
                "last_price": getattr(self.live_trader, '_last_valid_price', None),
                "model_loaded": self.live_trader.model is not None,
                "adapter_available": self.live_trader.exchange_adapter is not None,
            }

            # Merge with health summary
            summary.update(trading_info)

            # Convert status enum to string for backward compatibility
            summary["status"] = summary["overall_status"]

            return summary

        except Exception as e:
            return {
                "status": "critical",
                "issues": [f"Health check failed: {e}"],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _check_trading_model(self) -> HealthCheckResult:
        """Check if trading model is loaded and healthy."""
        start_time = time.time()
        try:
            if self.live_trader.model is None:
                return HealthCheckResult(
                    name="trading_model",
                    status=HealthStatus.UNHEALTHY,
                    message="Trading model not loaded",
                    details={},
                    timestamp=time.time(),
                    duration=time.time() - start_time
                )
            return HealthCheckResult(
                name="trading_model",
                status=HealthStatus.HEALTHY,
                message="Trading model loaded successfully",
                details={"model_type": type(self.live_trader.model).__name__},
                timestamp=time.time(),
                duration=time.time() - start_time
            )
        except Exception as e:
            return HealthCheckResult(
                name="trading_model",
                status=HealthStatus.UNHEALTHY,
                message=f"Trading model check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                duration=time.time() - start_time
            )

    def _check_exchange_adapter(self) -> HealthCheckResult:
        """Check if exchange adapter is available."""
        start_time = time.time()
        try:
            if self.live_trader.exchange_adapter is None:
                return HealthCheckResult(
                    name="exchange_adapter",
                    status=HealthStatus.UNHEALTHY,
                    message="Exchange adapter not available",
                    details={},
                    timestamp=time.time(),
                    duration=time.time() - start_time
                )
            return HealthCheckResult(
                name="exchange_adapter",
                status=HealthStatus.HEALTHY,
                message="Exchange adapter available",
                details={"adapter_type": type(self.live_trader.exchange_adapter).__name__},
                timestamp=time.time(),
                duration=time.time() - start_time
            )
        except Exception as e:
            return HealthCheckResult(
                name="exchange_adapter",
                status=HealthStatus.UNHEALTHY,
                message=f"Exchange adapter check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                duration=time.time() - start_time
            )

    def _check_position_validity(self) -> HealthCheckResult:
        """Check if current position is valid."""
        start_time = time.time()
        try:
            position = self.live_trader.position
            if not (-1 <= position <= 1):
                return HealthCheckResult(
                    name="position_validity",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Invalid position: {position}",
                    details={"position": position},
                    timestamp=time.time(),
                    duration=time.time() - start_time
                )
            return HealthCheckResult(
                name="position_validity",
                status=HealthStatus.HEALTHY,
                message="Position is valid",
                details={"position": position},
                timestamp=time.time(),
                duration=time.time() - start_time
            )
        except Exception as e:
            return HealthCheckResult(
                name="position_validity",
                status=HealthStatus.UNHEALTHY,
                message=f"Position validity check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                duration=time.time() - start_time
            )

    def _check_price_data_freshness(self) -> HealthCheckResult:
        """Check if price data is fresh."""
        start_time = time.time()
        try:
            price_age_seconds = 0
            last_price_time = getattr(self.live_trader, '_last_price_time', None)
            if last_price_time:
                price_age_seconds = (datetime.now() - last_price_time).total_seconds()
                if price_age_seconds > 300:  # 5 minutes
                    return HealthCheckResult(
                        name="price_data_freshness",
                        status=HealthStatus.DEGRADED,
                        message=f"Price data stale: {price_age_seconds:.0f}s old",
                        details={"age_seconds": price_age_seconds},
                        timestamp=time.time(),
                        duration=time.time() - start_time
                    )
            return HealthCheckResult(
                name="price_data_freshness",
                status=HealthStatus.HEALTHY,
                message="Price data is fresh",
                details={"age_seconds": price_age_seconds},
                timestamp=time.time(),
                duration=time.time() - start_time
            )
        except Exception as e:
            return HealthCheckResult(
                name="price_data_freshness",
                status=HealthStatus.UNHEALTHY,
                message=f"Price data freshness check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                duration=time.time() - start_time
            )

    def _check_feature_computation(self) -> HealthCheckResult:
        """Check if feature computation is working."""
        start_time = time.time()
        try:
            test_features = self.live_trader._compute_features()
            if len(test_features) == 0:
                return HealthCheckResult(
                    name="feature_computation",
                    status=HealthStatus.UNHEALTHY,
                    message="Feature computation returning empty array",
                    details={"features_length": len(test_features)},
                    timestamp=time.time(),
                    duration=time.time() - start_time
                )
            return HealthCheckResult(
                name="feature_computation",
                status=HealthStatus.HEALTHY,
                message="Feature computation working",
                details={"features_length": len(test_features)},
                timestamp=time.time(),
                duration=time.time() - start_time
            )
        except Exception as e:
            return HealthCheckResult(
                name="feature_computation",
                status=HealthStatus.UNHEALTHY,
                message=f"Feature computation failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                duration=time.time() - start_time
            )

    def _check_api_connectivity(self) -> HealthCheckResult:
        """Check API connectivity."""
        start_time = time.time()
        try:
            if not self.live_trader.exchange_adapter:
                return HealthCheckResult(
                    name="api_connectivity",
                    status=HealthStatus.UNKNOWN,
                    message="No exchange adapter available",
                    details={},
                    timestamp=time.time(),
                    duration=time.time() - start_time
                )

            # Quick connectivity test
            test_price = self.live_trader._get_current_price()
            if asyncio.iscoroutine(test_price):
                test_price = asyncio.run(test_price)

            if test_price > 0:
                return HealthCheckResult(
                    name="api_connectivity",
                    status=HealthStatus.HEALTHY,
                    message="API connectivity successful",
                    details={"test_price": test_price},
                    timestamp=time.time(),
                    duration=time.time() - start_time
                )
            else:
                return HealthCheckResult(
                    name="api_connectivity",
                    status=HealthStatus.DEGRADED,
                    message="API connectivity test returned invalid price",
                    details={"test_price": test_price},
                    timestamp=time.time(),
                    duration=time.time() - start_time
                )
        except Exception as e:
            return HealthCheckResult(
                name="api_connectivity",
                status=HealthStatus.UNHEALTHY,
                message=f"API connectivity check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                duration=time.time() - start_time
            )