"""
Compatibility shim for circuit_breakers module.

Re-exports symbols from the canonical ztb.utils.circuit_breaker module.
This shim allows gradual migration from old imports.

TODO: Remove this shim after migration to ztb.utils.circuit_breaker is complete.
"""

from ztb.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenException,
    CircuitState,
    get_circuit_breaker,
    reset_all_circuit_breakers,
)

# Re-export all symbols for backward compatibility
__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenException",
    "CircuitState",
    "get_circuit_breaker",
    "reset_all_circuit_breakers",
]