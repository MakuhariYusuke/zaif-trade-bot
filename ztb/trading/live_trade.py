#!/usr/bin/env python3
"""
DEPRECATED: Live Trading Bot for BTC/JPY using Trained SAC Model.

This module has been refactored and split into multiple modules under ztb.trading.live_trader.

Please use the new modular structure instead:

- ztb.trading.live_trader.config: Configuration classes
- ztb.trading.live_trader.live_trader: LiveTrader class
- ztb.trading.live_trader.main: Main entry point
- ztb.trading.live_trader.utils: Utility functions

Example usage:
    from ztb.trading.live_trader import LiveTrader, main

    # or
    from ztb.trading.live_trader.live_trader import LiveTrader
    from ztb.trading.live_trader.main import main
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "ztb.trading.live_trade module is deprecated. "
    "Please use ztb.trading.live_trader instead. "
    "The old module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new modules for backward compatibility
from ztb.trading.live_trader.main import main

# Keep some constants for backward compatibility
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = -1

# For backward compatibility, keep the old main function
def _deprecated_main():
    """Deprecated main function. Use ztb.trading.live_trader.main.main() instead."""
    warnings.warn(
        "live_trade._deprecated_main() is deprecated. Use ztb.trading.live_trader.main.main() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
