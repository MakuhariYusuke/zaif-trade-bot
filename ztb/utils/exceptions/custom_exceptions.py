"""Custom exception classes for improved error handling."""

from typing import Any, Dict, Optional


class ZTBBaseException(Exception):
    """Base exception class for ZTB system."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(ZTBBaseException):
    """Raised when there's a configuration error."""
    pass


class ValidationError(ZTBBaseException):
    """Raised when validation fails."""
    pass


class TrainingError(ZTBBaseException):
    """Raised when training-related errors occur."""
    pass


class TradingError(ZTBBaseException):
    """Raised when trading-related errors occur."""
    pass


class ModelError(ZTBBaseException):
    """Raised when model-related errors occur."""
    pass


class DataError(ZTBBaseException):
    """Raised when data-related errors occur."""
    pass


class RiskError(ZTBBaseException):
    """Raised when risk management errors occur."""
    pass


class EnvironmentError(ZTBBaseException):
    """Raised when environment-related errors occur."""
    pass


class NetworkError(ZTBBaseException):
    """Raised when network/API errors occur."""
    pass


class ResourceError(ZTBBaseException):
    """Raised when resource-related errors occur (memory, disk, etc.)."""
    pass


# Specific training exceptions
class EarlyStoppingError(TrainingError):
    """Raised when early stopping is triggered."""
    pass


class OverfittingError(TrainingError):
    """Raised when overfitting is detected."""
    pass


class TrainingInstabilityError(TrainingError):
    """Raised when training becomes unstable."""
    pass


class ConvergenceError(TrainingError):
    """Raised when training fails to converge."""
    pass


# Specific trading exceptions
class InsufficientFundsError(TradingError):
    """Raised when insufficient funds for trading."""
    pass


class InvalidOrderError(TradingError):
    """Raised when order parameters are invalid."""
    pass


class MarketDataError(DataError):
    """Raised when market data is invalid or unavailable."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    pass


class ModelSaveError(ModelError):
    """Raised when model saving fails."""
    pass


# Specific validation exceptions
class RewardValidationError(ValidationError):
    """Raised when reward validation fails."""
    pass


class ActionValidationError(ValidationError):
    """Raised when action validation fails."""
    pass


class StateValidationError(ValidationError):
    """Raised when state validation fails."""
    pass