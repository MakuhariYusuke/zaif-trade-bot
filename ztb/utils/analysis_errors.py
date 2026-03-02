"""Analysis-specific error handling utilities.

This module provides error handling patterns specifically designed for
analysis operations, including safe execution of analysis functions
with appropriate logging and fallback behaviors.
"""

import logging
from typing import Any, Callable

from ztb.utils.errors import TradingBotError, safe_operation

class AnalysisError(TradingBotError):
    """Base exception for analysis-related errors."""

    pass

class DataAnalysisError(AnalysisError):
    """Raised when data analysis operations fail."""

    pass

class PerformanceAnalysisError(AnalysisError):
    """Raised when performance analysis operations fail."""

    pass

class PatternAnalysisError(AnalysisError):
    """Raised when pattern analysis operations fail."""

    pass

def safe_analysis_operation(
    operation: Callable[..., Any],
    *args: Any,
    logger: logging.Logger | None = None,
    fallback_result: Any = None,
    context: str = "",
    error_types: tuple[type[Exception], ...] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute an analysis operation safely with analysis-specific error handling.

    This is a wrapper around safe_operation that provides analysis-specific
    defaults and logging patterns.

    Args:
        operation: The analysis function to execute
        *args: Positional arguments to pass to the operation
        logger: Logger instance for error reporting
        fallback_result: Value to return on error (default: None)
        context: Context string for error messages
        error_types: Specific exception types to catch (default: all)
        **kwargs: Keyword arguments to pass to the operation

    Returns:
        Result of the operation or fallback_result on error
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Default to catching analysis-related errors plus general exceptions
    if error_types is None:
        error_types = (AnalysisError, ValueError, TypeError, KeyError, AttributeError)

    return safe_operation(
        operation,
        *args,
        logger=logger,
        fallback=fallback_result,
        context=f"Analysis operation failed: {context}",
        error_types=error_types,
        **kwargs,
    )

def safe_data_analysis(
    operation: Callable[..., Any],
    *args: Any,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> Any:
    """Execute data analysis operations with data-specific error handling.

    Specialized for data processing operations that might fail due to
    data quality issues or missing data.
    """
    return safe_analysis_operation(
        operation,
        *args,
        logger=logger,
        context="Data analysis",
        error_types=(DataAnalysisError, ValueError, KeyError, AttributeError),
        **kwargs,
    )

def safe_performance_analysis(
    operation: Callable[..., Any],
    *args: Any,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> Any:
    """Execute performance analysis operations with performance-specific error handling.

    Specialized for performance calculations that might fail due to
    mathematical errors or invalid trading data.
    """
    return safe_analysis_operation(
        operation,
        *args,
        logger=logger,
        context="Performance analysis",
        error_types=(
            PerformanceAnalysisError,
            ValueError,
            ZeroDivisionError,
            OverflowError,
        ),
        **kwargs,
    )

def safe_pattern_analysis(
    operation: Callable[..., Any],
    *args: Any,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> Any:
    """Execute pattern analysis operations with pattern-specific error handling.

    Specialized for pattern recognition operations that might fail due to
    statistical computation errors or insufficient data.
    """
    return safe_analysis_operation(
        operation,
        *args,
        logger=logger,
        context="Pattern analysis",
        error_types=(PatternAnalysisError, ValueError, RuntimeError, StatisticsError),
        **kwargs,
    )

class StatisticsError(AnalysisError):
    """Raised when statistical computations fail."""

    pass
