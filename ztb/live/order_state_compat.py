"""
Compatibility shim for order_state module.

Re-exports symbols from the canonical ztb.trading.order_state_machine module.
This shim allows gradual migration from old imports.

TODO: Remove this shim after migration to ztb.trading.order_state_machine is complete.
"""

from ztb.trading.order_state_machine import (
    IdempotencyManager,
    OrderData,
    OrderEvent,
    OrderState,
    OrderStateMachine,
    get_idempotency_manager,
)

# Re-export all symbols for backward compatibility
__all__ = [
    "IdempotencyManager",
    "OrderData",
    "OrderEvent",
    "OrderState",
    "OrderStateMachine",
    "get_idempotency_manager",
]