"""Top-level shim exposing ResultComparator used in legacy integration tests.

Re-exports `ResultComparator` from `ztb.trading.production.result_comparator`.
"""
from ztb.trading.production.result_comparator import ResultComparator

__all__ = ["ResultComparator"]
