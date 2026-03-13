#!/usr/bin/env python3
"""
DEPRECATED: Trading service runner.

This module is deprecated. Use ztb.live.service_runner instead.
"""

import sys
import warnings
from pathlib import Path

from ztb.utils.file_utils import get_project_root

# Add project root to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))

warnings.warn(
    "ztb.scripts.trading_service is deprecated. Use ztb.live.service_runner instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import and re-export from the new location
try:
    pass

    __all__ = []
except ImportError as e:
    print(f"Error importing service_runner: {e}", file=sys.stderr)
    __all__ = []
