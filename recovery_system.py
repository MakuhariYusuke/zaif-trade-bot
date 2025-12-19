"""Top-level shim exposing RecoverySystem for tests that import it directly.

The real implementation lives under `ztb.trading.production.recovery_system`;
re-export it here so `from recovery_system import RecoverySystem` works during
test collection.
"""
from ztb.trading.production.recovery_system import RecoverySystem, RecoveryStatus

__all__ = ["RecoverySystem", "RecoveryStatus"]
