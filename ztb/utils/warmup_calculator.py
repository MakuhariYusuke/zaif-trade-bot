"""
Automatic warmup calculation based on feature lookback periods.

This module calculates the minimum warmup period required for all features
to have sufficient historical data for computation.
"""

import math
from typing import Dict, Union


def get_max_lookback() -> int:
    """
    Get the maximum lookback period from all registered features.

    Returns:
        Maximum lookback period required across all features.

    Examples:
        >>> get_max_lookback()
        200  # If SMA_200 is the longest lookback feature
    """
    # Known lookback periods for common indicators
    # These should ideally come from feature metadata
    lookback_map = {
        "sma_5": 5,
        "sma_10": 10,
        "sma_20": 20,
        "sma_50": 50,
        "sma_100": 100,
        "sma_200": 200,
        "ema_12": 12,
        "ema_26": 26,
        "rsi_14": 14,
        "rsi_28": 28,
        "macd": 26,  # EMA_26 is the longest component
        "bollinger": 20,  # Typical period
        "atr_14": 14,
        "ichimoku": 52,  # Senkou Span B uses 52 periods
        "stochastic_14": 14,
        "adx_14": 14,
        "cci_20": 20,
        "williams_r_14": 14,
        "obv": 1,  # Only needs current volume
        "vwap": 1,  # Daily VWAP resets
        "price_return": 1,
        "volume_return": 1,
    }

    # Get maximum lookback
    max_lookback = max(lookback_map.values())

    return max_lookback


def calculate_warmup(
    safety_margin: float = 0.1,
) -> int:
    """
    Calculate the recommended warmup period with safety margin.

    Args:
        safety_margin: Additional buffer as fraction of max_lookback (default 10%).

    Returns:
        Recommended warmup period (ceiling of max_lookback * (1 + safety_margin)).

    Examples:
        >>> calculate_warmup()
        220  # For max_lookback=200 with 10% margin

        >>> calculate_warmup(safety_margin=0.2)
        240  # For max_lookback=200 with 20% margin
    """
    max_lookback = get_max_lookback()
    warmup = math.ceil(max_lookback * (1.0 + safety_margin))
    return warmup


def get_warmup_with_metadata(
    safety_margin: float = 0.1,
) -> Dict[str, Union[int, float]]:
    """
    Get warmup calculation with detailed metadata.

    Args:
        safety_margin: Additional buffer as fraction of max_lookback (default 10%).

    Returns:
        Dictionary containing:
            - max_lookback: Maximum lookback period found
            - safety_margin: Margin applied
            - warmup: Final recommended warmup period

    Examples:
        >>> get_warmup_with_metadata()
        {'max_lookback': 200, 'safety_margin': 0.1, 'warmup': 220}
    """
    max_lookback = get_max_lookback()
    warmup = calculate_warmup(safety_margin)

    return {
        "max_lookback": max_lookback,
        "safety_margin": safety_margin,
        "warmup": warmup,
    }


def validate_warmup(
    provided_warmup: int,
    safety_margin: float = 0.1,
) -> bool:
    """
    Validate that provided warmup is sufficient.

    Args:
        provided_warmup: Warmup period to validate.
        safety_margin: Required safety margin (default 10%).

    Returns:
        True if provided warmup >= calculated minimum warmup, False otherwise.

    Examples:
        >>> validate_warmup(220)
        True  # 220 >= 220 (for max_lookback=200)

        >>> validate_warmup(100)
        False  # 100 < 220
    """
    required_warmup = calculate_warmup(safety_margin)
    return provided_warmup >= required_warmup
