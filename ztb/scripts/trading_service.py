#!/usr/bin/env python3
"""
DEPRECATED: Trading service runner.

This module is deprecated. Use ztb.live.service_runner instead.
"""

import sys
import warnings

warnings.warn(
    "ztb.scripts.trading_service is deprecated. Use ztb.live.service_runner instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import and re-export from the new location
try:
    from ztb.live.service_runner import TradingService, main

    __all__ = ["TradingService", "main"]
except ImportError as e:
    print(f"Error importing service_runner: {e}", file=sys.stderr)
    TradingService = None  # type: ignore
    main = None  # type: ignore
    __all__ = []
