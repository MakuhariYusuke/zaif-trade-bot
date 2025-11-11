"""
Common timeframe definitions for multi-timeframe feature calculations
"""

from enum import Enum


class Timeframe(Enum):
    """Enumeration of supported timeframes for feature calculations."""

    M1 = "1min"  # 1-minute equivalent
    M5 = "5min"  # 5-minute equivalent
    M15 = "15min"  # 15-minute equivalent
    H1 = "1hour"  # 1-hour equivalent
    H4 = "4hour"  # 4-hour equivalent
    D1 = "1day"  # 1-day equivalent


from typing import Any, Dict


def get_timeframe_params(timeframe: Timeframe) -> Dict[str, Any]:
    """
    Get common timeframe parameters for various calculations.

    Args:
        timeframe: Timeframe enum value

    Returns:
        Dictionary with common parameters for the timeframe
    """
    # Common parameters that can be used across different indicators
    base_params = {
        Timeframe.M1: {"short_period": 5, "medium_period": 15, "long_period": 30},
        Timeframe.M5: {"short_period": 10, "medium_period": 30, "long_period": 60},
        Timeframe.M15: {"short_period": 20, "medium_period": 60, "long_period": 120},
        Timeframe.H1: {"short_period": 50, "medium_period": 150, "long_period": 300},
        Timeframe.H4: {"short_period": 100, "medium_period": 300, "long_period": 600},
        Timeframe.D1: {"short_period": 200, "medium_period": 600, "long_period": 1200},
    }

    return base_params.get(timeframe, base_params[Timeframe.D1])
