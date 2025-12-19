"""Top-level shim exposing RollbackManager for integration tests.

Re-exports implementation from `ztb.trading.production.rollback_manager`.
"""
from ztb.trading.production.rollback_manager import RollbackManager

__all__ = ["RollbackManager"]
