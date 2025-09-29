"""
Unified exception types for the trading bot.

Provides standardized error handling without bare except clauses.
"""

from typing import Any, Dict, Optional


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
