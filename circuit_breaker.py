# Compatibility shim: some legacy tests import `circuit_breaker` as a top-level module
# Delegate to the canonical implementation under `ztb.utils.circuit_breaker`.
try:
    from ztb.utils.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitState,
    )
except Exception:  # pragma: no cover - minimal no-op fallback for collection
    class CircuitState:
        CLOSED = "closed"
        OPEN = "open"

    class CircuitBreakerConfig:
        def __init__(self, *args, **kwargs):
            pass

    class CircuitBreaker:
        def __init__(self, name: str, config: CircuitBreakerConfig):
            self.name = name
            self.config = config
            self.state = CircuitState.CLOSED

        def record_failure(self):
            self.state = CircuitState.OPEN

        def record_success(self):
            self.state = CircuitState.CLOSED

__all__ = ["CircuitBreaker", "CircuitBreakerConfig", "CircuitState"]
