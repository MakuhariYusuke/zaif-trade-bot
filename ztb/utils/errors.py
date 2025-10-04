"""
Unified exception types for the trading bot.

Provides standardized error handling without bare except clauses.
"""

from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar('T'), Dict, Optional


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(TradingBotError):
    """Configuration-related errors."""

    pass


class ValidationError(TradingBotError):
    """Data validation errors."""

    pass


class NetworkError(TradingBotError):
    """Network and API communication errors."""

    pass


class DatabaseError(TradingBotError):
    """Database operation errors."""

    pass


class TradingError(TradingBotError):
    """Trading operation errors."""

    pass


class IdempotencyError(TradingBotError):
    """Idempotency-related errors."""

    pass


class LockError(TradingBotError):
    """Locking and concurrency errors."""

    pass


def handle_error(logger: Any, error: Exception, context: str = "", reraise: bool = True) -> None:
    """
    Unified error handling with consistent logging.

    Args:
        logger: Logger instance for error reporting
        error: The exception that occurred
        context: Additional context about where the error occurred
        reraise: Whether to re-raise the exception after logging
    """
    error_msg = f"{context}: {str(error)}" if context else str(error)
    logger.error(error_msg, exc_info=True)

    if reraise:
        raise error


def safe_operation(logger: Any, operation: Callable[[], Any], context: str = "", default_result: Any = None) -> Any:
    """
    Execute an operation safely with unified error handling.

    Args:
        logger: Logger instance for error reporting
        operation: Callable to execute
        context: Context for error messages
        default_result: Value to return on error

    Returns:
        Result of operation or default_result on error
    """
    try:
        return operation()
    except Exception as e:
        error_msg = f"{context}: {str(e)}" if context else str(e)
        logger.error(error_msg, exc_info=True)
        return default_result
