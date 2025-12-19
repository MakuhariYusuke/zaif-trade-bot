"""Minimal alert_system shim used by some integration tests."""


class AlertSystem:
    def __init__(self, *args, **kwargs):
        pass

    def send_alert(self, message: str, level: str = "info") -> None:
        # no-op for tests
        return None

__all__ = ["AlertSystem"]
