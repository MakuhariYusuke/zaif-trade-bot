"""Compatibility shim to expose RiskBasedAllocator at top-level for tests.

Re-exports the implementation from `ztb.trading.production.risk_based_allocator`.
"""
from ztb.trading.production.risk_based_allocator import (
    RiskBasedAllocator,
    AllocationDecision,
)

__all__ = ["RiskBasedAllocator", "AllocationDecision"]
