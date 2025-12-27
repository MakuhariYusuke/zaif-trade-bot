#!/usr/bin/env python3
"""
format_utils.py
Common formatting utilities for consistent output formatting across the codebase.
"""

from typing import Any, Dict


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if minutes > 0:
            return f"{hours}h {minutes}m {secs:.1f}s"
        else:
            return f"{hours}h {secs:.1f}s"


def format_number(num: float, precision: int = 2) -> str:
    """
    Format number with appropriate suffix (K, M, B).

    Args:
        num: Number to format
        precision: Decimal precision

    Returns:
        Formatted number string
    """
    if abs(num) >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def format_percentage(value: float, precision: int = 2) -> str:
    """
    Format value as percentage.

    Args:
        value: Value to format (0.1 = 10%)
        precision: Decimal precision

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"


def format_currency(amount: float, currency: str = "$", precision: int = 2) -> str:
    """
    Format amount as currency.

    Args:
        amount: Monetary amount
        currency: Currency symbol
        precision: Decimal precision

    Returns:
        Formatted currency string
    """
    return f"{currency}{amount:.{precision}f}"


def format_metric_summary(results: Dict[str, Any]) -> str:
    """
    Format metric results as summary string.

    Args:
        results: Dictionary of metric results

    Returns:
        Formatted summary string
    """
    summary_parts = []
    for key, value in results.items():
        if isinstance(value, float):
            if "pct" in key.lower() or "rate" in key.lower():
                summary_parts.append(f"{key}: {value:.2f}%")
            elif "ratio" in key.lower():
                summary_parts.append(f"{key}: {value:.3f}")
            else:
                summary_parts.append(f"{key}: {value:.4f}")
        else:
            summary_parts.append(f"{key}: {value}")

    return " | ".join(summary_parts)