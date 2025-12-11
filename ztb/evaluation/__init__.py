"""Compatibility package for evaluation modules.

Re-exports evaluation functionality from ztb.analysis.evaluation to support
older imports such as `ztb.evaluation.unified_evaluation` used in tests.
"""

from . import unified_evaluation  # noqa: F401

__all__ = ["unified_evaluation"]
