#!/usr/bin/env python3
"""
Main entry point for live trading.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ztb.utils.path_utils import get_project_root

project_root = get_project_root()
sys.path.insert(0, str(project_root))

from ztb.trading.live_trader.config import (
    HealthServerHandle,
    LiveTradingOptions,
    MetricsServerHandle,
    _build_argument_parser,
    _start_health_server,
    _start_metrics_server,
)
from ztb.trading.live_trader.live_trader import LiveTrader
from ztb.trading.live_trader.utils import _configure_live_logging


def main() -> None:
    """Main entry point for live trading."""
    try:
        _main_impl()
    except Exception as e:
        print(f"Error in main: {e}")
        raise


def _main_impl() -> None:
    """Implementation of main function."""
    print("Entering _main_impl")
    parser = _build_argument_parser()
    args = parser.parse_args()

    print(f"args.dry_run: {args.dry_run}")
    options = LiveTradingOptions.from_cli_args(args)
    runtime_logger = _configure_live_logging(options.log_level)

    print("After configure logging")
    # Start servers if enabled
    metrics_handle: MetricsServerHandle | None = _start_metrics_server(
        options, runtime_logger
    )
    health_handle: HealthServerHandle | None = _start_health_server(
        options,
        lambda: {"status": "initializing"},
        runtime_logger,  # Placeholder
    )

    # Initialize trader
    trader = LiveTrader(options)

    # Update health provider with trader
    if health_handle:
        # Could update health provider here if needed
        pass

    # Run trading loop
    try:
        asyncio.run(trader.run_trading_loop(options.duration_hours))
    except KeyboardInterrupt:
        runtime_logger.info("Trading interrupted by user")
    except Exception as e:
        runtime_logger.error(f"Failed to run trading loop: {e}")
        raise
    finally:
        # Keep handles referenced for potential future cleanup logic
        _ = (metrics_handle, health_handle)


if __name__ == "__main__":
    main()
