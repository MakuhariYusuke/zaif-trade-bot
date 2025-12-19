# Compatibility shim: provide a top-level `emergency_stop` module used by some legacy tests.
try:
    from ztb.trading.risk_overlay import EmergencyStop  # type: ignore
except Exception:  # pragma: no cover - minimal fallback for collection
    class EmergencyStop:
        def __init__(self, *args, **kwargs):
            self.triggered = False

        def get_emergency_status(self):
            return {"triggered": self.triggered}

__all__ = ["EmergencyStop"]
