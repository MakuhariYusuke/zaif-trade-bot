"""Utility error types and a backwards-compatible safe_operation.

This module provides a lightweight TradingBotError hierarchy and a
flexible `safe_operation` helper. Many call-sites in the codebase call
`safe_operation` with different argument orders and keywords (historic
API drift). To reduce widespread mypy "call-arg" issues and to be
backwards-compatible, `safe_operation` accepts either style and will
forward arbitrary args/kwargs to the wrapped operation.
"""

from typing import Any, Callable, Dict, Optional


class ZTBError(Exception):
    """Base exception for all Zaif Trade Bot errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class TradingBotError(ZTBError):
    """Base exception for all trading bot errors (legacy compatibility)."""

    pass


class InsufficientFundsError(TradingBotError):
    """Raised when trading operation fails due to insufficient funds."""

    pass


class OrderError(TradingBotError):
    """Base class for order-related errors."""

    pass


class OrderNotFoundError(OrderError):
    """Raised when trying to cancel or query a non-existent order."""

    pass


class MinimumSizeError(OrderError):
    """Raised when order size is below exchange minimum requirements."""

    pass


class ValidationError(ZTBError):
    """Data validation errors."""


class ConfigurationError(ZTBError):
    """Configuration-related errors."""


class SchemaError(ZTBError):
    """Schema-related errors."""


class ModelError(ZTBError):
    """Model loading and inference errors."""


class NetworkError(ZTBError):
    """Network and API communication errors with retry information."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        retry_count: int = 0,
        max_retries: Optional[int] = None,
    ):
        super().__init__(message, details)
        self.url = url
        self.status_code = status_code
        self.retry_count = retry_count
        self.max_retries = max_retries


class DatabaseError(ZTBError):
    """Database operation errors."""


class TradingError(ZTBError):
    """Trading operation errors with position and order context."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        position: Optional[float] = None,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
    ):
        super().__init__(message, details)
        self.position = position
        self.order_id = order_id
        self.symbol = symbol


class IdempotencyError(TradingBotError):
    """Idempotency-related errors."""


class LockError(TradingBotError):
    """Locking and concurrency errors."""


def handle_error(
    logger: Any, error: Exception, context: str = "", reraise: bool = True
) -> None:
    """Log and optionally re-raise an exception.

    This helper keeps calling code concise while making sure exceptions
    are not lost.
    """
    error_msg = f"{context}: {str(error)}" if context else str(error)
    try:
        logger.error(error_msg, exc_info=True)
    except Exception:
        # Be defensive: logging should not crash the error handler
        pass

    if reraise:
        raise error


def safe_operation(*args: Any, **kwargs: Any) -> Any:
    """Execute an operation safely, supporting multiple calling styles.

    Supported invocation patterns (both are used in the codebase):
    - safe_operation(logger, operation=callable, ...)
    - safe_operation(operation_callable, fallback=..., df=..., verbose=True)

    The helper will:
    - detect the operation callable and optional logger
    - accept both `default_result` and `fallback` as the return-on-error
    - accept `context` or `operation_name` as a context string for logging
    - forward any remaining kwargs to the operation when calling it
    """
    # Normalize common control keywords
    default_result = kwargs.pop("default_result", kwargs.pop("fallback", None))
    context = kwargs.pop("context", kwargs.pop("operation_name", ""))
    reraise_critical = kwargs.pop("reraise_critical", False)
    error_types = kwargs.pop("error_types", None)

    logger: Optional[Any] = None
    operation: Optional[Callable[..., Any]] = None
    call_args: tuple[Any, ...] = ()

    # Determine operation and logger from positional args or kwargs
    if args:
        if callable(args[0]):
            operation = args[0]
            call_args = args[1:]
            logger = kwargs.pop("logger", None)
        else:
            # First positional arg is likely a logger
            logger = args[0]
            operation = kwargs.pop("operation", None)
            call_args = args[1:]
    else:
        operation = kwargs.pop("operation", None)
        logger = kwargs.pop("logger", None)

    if operation is None:
        raise ValueError("safe_operation requires an operation callable")

    # Remaining kwargs are intended to be forwarded to the operation call
    call_kwargs = kwargs

    try:
        result = operation(*call_args, **call_kwargs)
        return result
    except Exception as e:  # noqa: BLE001 - we intentionally catch exceptions to return default
        # If caller asked to only handle specific types, re-raise others
        if error_types is not None and not isinstance(e, tuple(error_types)):
            raise

        # Re-raise critical trading errors if requested
        if reraise_critical and isinstance(e, TradingBotError):
            raise

        # Log if logger is available
        msg = f"{context}: {e}" if context else str(e)
        if logger is not None:
            try:
                logger.error(msg, exc_info=True)
            except Exception:
                pass

    return default_result
