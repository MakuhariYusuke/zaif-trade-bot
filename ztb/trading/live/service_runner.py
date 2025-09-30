#!/usr/bin/env python3
"""
24/7 Trading Service Runner

A continuous trading service that runs the trading bot with automatic restarts,
health monitoring, and comprehensive logging.
"""

import argparse
import logging
import signal
import sys
import time
from typing import Optional

from ztb.ops.monitoring.health_monitor import HealthMonitor
from ztb.utils.errors import TradingBotError


class TradingService:
    """24/7 trading service with automatic restarts and monitoring."""

    def __init__(self, config_path: Optional[str] = None, log_level: str = "INFO"):
        self.config_path = config_path
        self.log_level = log_level
        self.running = False
        self.restart_count = 0
        self.max_restarts = 10
        self.restart_delay = 60  # seconds
        self.health_monitor = HealthMonitor("trading_service")
        self.last_health_report = 0
        self.health_report_interval = 300  # 5 minutes

        # Set up logging
        self._setup_logging()

        # Handle signals
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _setup_logging(self):
        """Set up comprehensive logging."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("trading_service.log", mode="a"),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _run_trading_cycle(self) -> bool:
        """
        Run a single trading cycle.

        Returns:
            True if cycle completed successfully, False if it crashed
        """
        try:
            self.logger.info("Starting trading cycle...")

            # Import and run the main trading logic
            # This would typically call the main trading functions
            # For now, we'll simulate a trading cycle
            self._simulate_trading_cycle()

            self.logger.info("Trading cycle completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Trading cycle failed: {e}", exc_info=True)
            return False

    def _simulate_trading_cycle(self):
        """Simulate a trading cycle (replace with actual trading logic)."""
        # This is a placeholder - in real implementation, this would:
        # 1. Load configuration
        # 2. Initialize trading components
        # 3. Run trading loops
        # 4. Handle orders and positions
        # 5. Monitor health and performance

        self.logger.info("Simulating trading operations...")

        # Simulate some work
        time.sleep(5)

        # Check health
        if not self._check_health():
            raise TradingBotError("Health check failed")

        self.logger.info("Trading simulation completed")

    def _check_health(self) -> bool:
        """Perform health checks."""
        try:
            health_status = self.health_monitor.check_overall_health()

            # Log health status periodically
            current_time = time.time()
            if current_time - self.last_health_report > self.health_report_interval:
                self.logger.info(f"Health status: {health_status['status']}")
                for check_name, check_result in health_status["checks"].items():
                    if not check_result.get("healthy", True):
                        self.logger.warning(
                            f"Health check failed: {check_name} - {check_result}"
                        )
                self.last_health_report = current_time

            # Check if restart is needed
            if self.health_monitor.should_restart(health_status):
                self.logger.warning("Health monitor recommends restart")
                return False

            return health_status["status"] == "healthy"

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def _should_restart(self, cycle_success: bool) -> bool:
        """Determine if we should restart after a cycle."""
        if cycle_success:
            # Successful cycle - restart for continuous operation
            return True

        # Failed cycle - check restart limits
        self.restart_count += 1
        if self.restart_count >= self.max_restarts:
            self.logger.error(f"Maximum restarts ({self.max_restarts}) exceeded")
            return False

        self.logger.warning(
            f"Restarting after failure (attempt {self.restart_count}/{self.max_restarts})"
        )
        return True

    def run(self):
        """Run the 24/7 trading service."""
        self.logger.info("Starting 24/7 Trading Service")
        self.logger.info(f"Config: {self.config_path}")
        self.logger.info(f"Log level: {self.log_level}")

        self.running = True

        try:
            while self.running:
                # Run a trading cycle
                cycle_success = self._run_trading_cycle()

                # Check if we should continue
                if not self._should_restart(cycle_success):
                    break

                # Wait before next cycle (or restart)
                if self.running:
                    self.logger.info(
                        f"Waiting {self.restart_delay} seconds before next cycle..."
                    )
                    time.sleep(self.restart_delay)

        except Exception as e:
            self.logger.critical(
                f"Critical error in trading service: {e}", exc_info=True
            )

        finally:
            self.logger.info("Trading service stopped")
            self._cleanup()

    def _cleanup(self):
        """Clean up resources."""
        self.logger.info("Performing cleanup...")
        # Close connections, save state, etc.
        self.logger.info("Cleanup completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="24/7 Trading Service Runner")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=10,
        help="Maximum number of restarts on failure",
    )
    parser.add_argument(
        "--restart-delay", type=int, default=60, help="Delay between restarts (seconds)"
    )

    args = parser.parse_args()

    # Create and run service
    service = TradingService(config_path=args.config, log_level=args.log_level)

    # Override defaults if specified
    if args.max_restarts != 10:
        service.max_restarts = args.max_restarts
    if args.restart_delay != 60:
        service.restart_delay = args.restart_delay

    service.run()


if __name__ == "__main__":
    main()
